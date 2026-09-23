// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Capture front-end: 16 kHz microphone samples to speech segments.
//!
//! The chain, and why it is in this order:
//!
//! ```text
//!   webview capture ──► silero ─────────► segmenter
//!   16 kHz mono          speech            utterances, chunk pauses
//!   320-sample frames    probability       and turn endpoints
//!                        per 32 ms frame
//! ```
//!
//! Echo cancellation, noise suppression and gain control are not here, deliberately. The
//! webview owns the device, so it is the only side that knows what was just played and can
//! subtract it; running Chromium's chain and then a second one over the same audio pumps
//! noise and strips the quiet consonants the recogniser depends on. What arrives here is
//! already clean, and the only question left is when a person is speaking.
//!
//! Two things below are load-bearing:
//!
//! - **The pre-roll ring holds audio the VAD has not accepted yet.** By the time speech is
//!   confirmed, its first syllable is already in the past, so segments are cut from the ring
//!   rather than from the moment of detection.
//! - **Recurrent VAD state is reset at every turn boundary.** Carrying it forward makes the
//!   model open the next utterance already half-convinced someone is speaking.

use std::{collections::VecDeque, path::Path};

use ort::{session::Session, value::Value};

use crate::tts::TTS_SAMPLE_RATE;

/// The rate everything here runs at, and the rate the webview is asked to send.
///
/// Fixed rather than configurable: Silero and Qwen3-ASR both expect it, and a mismatch is
/// silent rather than loud - the models still return numbers, they are just wrong.
pub const SAMPLE_RATE: u32 = 16_000;

/// Speech probability at which a frame counts as speech.
///
/// Silero's own recommended threshold. One number, used everywhere a frame is judged, so the
/// segmenter and anything downstream agree about what speech is.
pub const SPEECH_PROBABILITY: f32 = 0.5;

/// Fresh samples per Silero inference, and the unit every duration here is counted in.
///
/// 512 at 16 kHz is 32 ms, which is the window Silero is published with and trained on.
///
/// This used to be 576, taken from the width of the model's input tensor and described as fixed
/// by the exported graph. The graph does take 576 - but 64 of them are the tail of the previous
/// window, which the reference wrapper carries over and feeds back in. Passing 576 *fresh*
/// samples instead advanced 36 ms per step and gave the model no context to start from, so every
/// frame was judged from a standing start and the frame clock ran 12.5% slow against the audio.
pub const VAD_FRAME_SAMPLES: usize = 512;

/// Samples of the previous window handed back to the model with the next one.
const VAD_CONTEXT_SAMPLES: usize = 64;

/// What the graph actually takes: the retained context followed by the new frame.
const VAD_INPUT_SAMPLES: usize = VAD_CONTEXT_SAMPLES + VAD_FRAME_SAMPLES;

const VAD_STATE_LEN: usize = 128;

// ============================================================================================
// Silero VAD
// ============================================================================================

/// Silero's published export, compiled into the binary.
///
/// It is 2.2 MB, it never changes between runs, and ONNX Runtime can build a session straight
/// from memory - so there is no file to install, no path to get wrong, and no way to end up
/// running a different voice detector than the one this was measured against.
const SILERO_MODEL: &[u8] = include_bytes!("silero_vad.onnx");

/// Silero VAD, holding its own recurrent state.
///
/// The state is carried from frame to frame, which is what lets the model use context, but
/// state left over from one turn causes false starts in the next, so [`SileroVad::reset`] exists
/// and the capture pipeline calls it at every turn boundary.
///
/// Silero's published export carries both LSTM halves in one `2 x 1 x 128` tensor and returns
/// the next one as `stateN`; older exports split it into separate `h` and `c` tensors. This
/// drives the published one.
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    /// The last samples of the previous frame, which begin the next inference.
    context: Vec<f32>,
}

impl SileroVad {
    /// The export compiled into this binary. This is what the running app uses.
    pub fn embedded() -> Result<Self, AudioError> {
        let session = Session::builder()
            .map_err(|error| AudioError::Onnx(error.to_string()))?
            .commit_from_memory(SILERO_MODEL)
            .map_err(|error| AudioError::Onnx(error.to_string()))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * VAD_STATE_LEN],
            context: vec![0.0; VAD_CONTEXT_SAMPLES],
        })
    }

    /// A model from disk, for comparing exports against the compiled-in one.
    pub fn load(model: impl AsRef<Path>) -> Result<Self, AudioError> {
        let model = model.as_ref();
        if !model.is_file() {
            return Err(AudioError::MissingModel(model.display().to_string()));
        }
        let session = Session::builder()
            .map_err(|error| AudioError::Onnx(error.to_string()))?
            .commit_from_file(model)
            .map_err(|error| AudioError::Onnx(error.to_string()))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * VAD_STATE_LEN],
            context: vec![0.0; VAD_CONTEXT_SAMPLES],
        })
    }

    /// Clears recurrent state.
    ///
    /// Without this between utterances the LSTM carries dirty state forward and turn two starts
    /// with the model already half-convinced someone is speaking.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        // The audio either side of a segment boundary is unrelated, so carrying the tail of the
        // last utterance into the first frame of the next one is the same mistake as carrying
        // the recurrent state.
        self.context.fill(0.0);
    }

    /// Speech probability for exactly one frame.
    pub fn probability(&mut self, frame: &[f32]) -> Result<f32, AudioError> {
        if frame.len() != VAD_FRAME_SAMPLES {
            return Err(AudioError::FrameSize {
                expected: VAD_FRAME_SAMPLES,
                actual: frame.len(),
            });
        }
        // The model reads the tail of the previous frame before this one. Without it every
        // window is judged from silence, which is not how it was trained.
        let mut window = Vec::with_capacity(VAD_INPUT_SAMPLES);
        window.extend_from_slice(&self.context);
        window.extend_from_slice(frame);
        self.context
            .copy_from_slice(&frame[VAD_FRAME_SAMPLES - VAD_CONTEXT_SAMPLES..]);
        let input = Value::from_array(([1usize, VAD_INPUT_SAMPLES], window))
            .map_err(|error| AudioError::Onnx(error.to_string()))?;
        let state = Value::from_array(([2usize, 1, VAD_STATE_LEN], self.state.clone()))
            .map_err(|error| AudioError::Onnx(error.to_string()))?;
        // The graph switches its front end on this value; Zen only ever feeds 16 kHz.
        let rate = Value::from_array(([1usize], vec![i64::from(SAMPLE_RATE)]))
            .map_err(|error| AudioError::Onnx(error.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs! { "input" => input, "state" => state, "sr" => rate })
            .map_err(|error| AudioError::Onnx(error.to_string()))?;

        let probability = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|error| AudioError::Onnx(error.to_string()))?
            .1
            .first()
            .copied()
            .ok_or_else(|| AudioError::Onnx("output was empty".into()))?;

        // State must be copied out before the next call or the model runs open-loop.
        self.state.copy_from_slice(
            outputs["stateN"]
                .try_extract_tensor::<f32>()
                .map_err(|error| AudioError::Onnx(error.to_string()))?
                .1,
        );

        Ok(probability)
    }
}

// ============================================================================================
// Segmentation
// ============================================================================================

/// How long a pause has to run before the turn is handed over.
///
/// A fixed number cannot be right for everyone. Some people finish a sentence and stop; others
/// think out loud with a beat in the middle of every thought. Set it short and the second kind
/// gets cut off mid-sentence and has to start again; set it long and the first kind waits for
/// a reply that was ready a second ago. Neither is a setting anyone should have to discover.
///
/// [`EndpointPolicy`] measures the speaker instead, from two kinds of evidence:
///
/// - **A pause inside a turn** - silence, then more words before the endpoint fired. The
///   endpoint already tolerated it, so this only says it must not fall below that.
/// - **A turn that ended and was resumed straight away.** This is the important one, and the
///   only signal that says the endpoint was *too short*: a pause longer than the endpoint ends
///   the turn by definition, so it can never be observed from inside one. What can be observed
///   is the speaker carrying on a moment later - which means the silence was a breath, not an
///   ending, and their real pause was the endpoint plus however long they took to resume.
///
/// It relaxes geometrically over uneventful turns, so one long hesitation does not slow the
/// rest of the conversation for good, and is clamped at both ends. What it learned is reported to
/// the page and handed back at the start of the next session, so it is not relearned from
/// scratch every time Zen starts.
///
/// There is no hand-set mode. A fixed silence always cuts somebody off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointPolicy {
    /// Never wait less than this, however brisk the speaker.
    floor_frames: usize,
    /// Never wait more than this, however hesitant.
    ceiling_frames: usize,
    /// Held above the longest within-turn pause seen, so a pause of the same length again does
    /// not end the turn.
    margin_frames: usize,
    /// Current estimate.
    current_frames: usize,
}

impl EndpointPolicy {
    /// A neutral starting point: long enough for an ordinary mid-sentence beat, short enough
    /// that a decisive speaker is not left waiting.
    pub fn adaptive() -> Self {
        Self {
            floor_frames: SegmenterConfig::frames_for_ms(600),
            ceiling_frames: SegmenterConfig::frames_for_ms(2_000),
            margin_frames: SegmenterConfig::frames_for_ms(150),
            current_frames: SegmenterConfig::frames_for_ms(900),
        }
    }

    /// The adaptive policy, starting from a pause learned in an earlier session.
    ///
    /// Clamped into the policy's own range, so a remembered value can never push it outside the
    /// bounds a fresh one keeps to.
    pub fn adaptive_from_ms(ms: usize) -> Self {
        let mut policy = Self::adaptive();
        policy.current_frames =
            SegmenterConfig::frames_for_ms(ms).clamp(policy.floor_frames, policy.ceiling_frames);
        policy
    }

    /// A policy that never moves. Tests only: segmentation boundaries are much easier to reason
    /// about when the endpoint does not shift under them. A range with no width is exactly that,
    /// so it needs no separate code path.
    #[cfg(test)]
    pub fn fixed(frames: usize) -> Self {
        Self {
            floor_frames: frames,
            ceiling_frames: frames,
            margin_frames: 0,
            current_frames: frames,
        }
    }

    /// Frames of silence that end a turn right now.
    pub fn frames(self) -> usize {
        self.current_frames
    }

    /// The shortest wait the policy will ever settle on.
    pub fn floor(self) -> usize {
        self.floor_frames
    }

    /// The longest wait the policy will ever settle on.
    pub fn ceiling(self) -> usize {
        self.ceiling_frames
    }

    /// Raise the endpoint clear of a pause the speaker has demonstrated they take.
    ///
    /// Only ever raises. Being cut off mid-sentence costs the speaker the whole utterance and
    /// their place in the thought; waiting a fraction of a second longer costs a fraction of a
    /// second, so the two errors are not worth trading symmetrically.
    fn observe(&mut self, pause: usize) {
        let wanted = (pause + self.margin_frames).clamp(self.floor_frames, self.ceiling_frames);
        self.current_frames = self.current_frames.max(wanted);
    }

    /// One turn that ended and stayed ended: evidence the endpoint is not too short.
    ///
    /// Gives back an eighth of whatever the wait stands above its floor, and never less than a
    /// frame. That is a half-life of about five decisive turns. It used to fall by one frame per
    /// turn, which after a single long pause (measured: the wait learned 1.47 s while reading
    /// aloud) took some eighteen clean turns to come back under a second - that whole time, every
    /// reply started half a second later than it needed to. A fraction rather than a fixed step
    /// also keeps one brisk reply from undoing most of what was learned.
    fn relax(&mut self) {
        let excess = self.current_frames.saturating_sub(self.floor_frames);
        self.current_frames = self
            .current_frames
            .saturating_sub((excess / 8).max(1))
            .max(self.floor_frames);
    }
}

/// Tuning for turning a stream of speech probabilities into utterances.
#[derive(Debug, Clone, Copy)]
pub struct SegmenterConfig {
    /// Probability at or above which a frame counts as speech.
    pub enter_threshold: f32,
    /// Probability below which an in-progress utterance counts as silent.
    ///
    /// Deliberately lower than [`Self::enter_threshold`]. A single threshold makes the decision
    /// chatter around the boundary, cutting a segment in the middle of a word whenever the
    /// speaker's level dips; the gap between the two is what stops that.
    pub exit_threshold: f32,
    /// Consecutive speech frames required before an utterance is declared open.
    ///
    /// Rejects keyboard clicks, door knocks, and lip smacks, which spike for one frame.
    pub onset_frames: usize,
    /// Silence that closes a *sub-chunk* while the turn stays open.
    ///
    /// A short pause between phrases. The piece just spoken is handed to transcription
    /// immediately, so by the time the speaker actually finishes, most of their words have
    /// already been through the recogniser.
    pub chunk_pause_frames: usize,
    /// Silence that ends the *turn*.
    ///
    /// This is the endpoint: input stops and the model starts. It has to be long enough that a
    /// speaker gathering their thoughts is not cut off. It also buys time: the phrase before the
    /// silence was handed to recognition at the chunk pause, so that work runs while this timer
    /// does. Recognition costs about 300 ms plus 630 ms per second of audio, so a short last
    /// phrase is done by the time the endpoint fires and a long one is well on its way.
    /// [`EndpointPolicy`] sets the length without anyone having to choose a number.
    pub endpoint: EndpointPolicy,
    /// Audio kept from *before* the detected onset.
    ///
    /// Silero reports speech a frame or two after it starts, and confirmation costs more frames
    /// still. Without this padding every segment loses its first consonant, which is exactly the
    /// part a transcriber needs.
    pub preroll_frames: usize,
    /// Audio kept after the hangover expires, so trailing consonants survive.
    pub postroll_frames: usize,
    /// How soon after a turn ends a new utterance still counts as the speaker carrying on.
    ///
    /// Past this it is a genuinely new thing to say, and the endpoint that ended the last turn
    /// was right. Roughly a second: long enough to cover drawing breath, short enough that a
    /// considered follow-up is not mistaken for an interrupted one.
    pub resume_window_frames: usize,
    /// Utterances shorter than this are discarded as noise.
    pub minimum_frames: usize,
    /// A piece this long is handed to the recogniser at the speaker's next breath, while they are
    /// still talking.
    ///
    /// Pieces used to be cut only at a [`Self::chunk_pause_frames`] pause, so someone speaking
    /// fluently was recognised only once they had stopped. Measured in the running app
    /// (2026-09-23): a 14.5 s passage whose only gap was 96 ms long went to the recogniser whole
    /// when it ended, recognition then took 11.8 s, and the answer began 14.4 s after the speaker
    /// stopped. Recognition runs faster than speech, so handed over in pieces it keeps up instead.
    pub rolling_frames: usize,
    /// Hard cut, for speech that reaches this without a breath.
    pub maximum_frames: usize,
    /// Audio repeated across a hard cut, so a word it lands in is heard whole at least once.
    pub overlap_frames: usize,
    /// Hard end of a turn, however much speech is still arriving.
    ///
    /// A turn this long is not a person finishing a thought - it is a television, a call on
    /// speakerphone, or a microphone left open. Without it the turn grows until some other
    /// limit trips and the speaker is told to start again, having said everything already.
    pub max_turn_frames: usize,
}

impl SegmenterConfig {
    pub const fn frames_for_ms(ms: usize) -> usize {
        // 512 samples at 16 kHz is 32 ms.
        let frame_ms = VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize;
        let frames = ms / frame_ms;
        if frames == 0 {
            1
        } else {
            frames
        }
    }
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        // Pieces are cut to the lengths the recogniser itself is sized for, so one definition
        // governs both the live cut and the planner that splits anything longer.
        let chunks = ChunkConfig::default();
        Self {
            enter_threshold: SPEECH_PROBABILITY,
            exit_threshold: 0.35,
            onset_frames: 2,                              // 64 ms candidate onset
            chunk_pause_frames: Self::frames_for_ms(400), // ~400 ms
            endpoint: EndpointPolicy::adaptive(),
            preroll_frames: Self::frames_for_ms(300), // ~300 ms
            postroll_frames: Self::frames_for_ms(200), // ~200 ms
            resume_window_frames: Self::frames_for_ms(1_000), // ~1 s
            minimum_frames: Self::frames_for_ms(160), // retain brief answers such as yes/no
            rolling_frames: Self::frames_for_ms(chunks.target_ms), // 6 s
            maximum_frames: Self::frames_for_ms(chunks.maximum_ms), // 10 s
            overlap_frames: Self::frames_for_ms(chunks.overlap_ms), // 200 ms
            max_turn_frames: Self::frames_for_ms(60_000), // 60 s
        }
    }
}

/// A completed utterance, ready for transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub samples: Vec<f32>,
    /// Speech probability per VAD frame, retained so the segment can be split at its own pauses
    /// later without re-running the detector.
    pub probabilities: Vec<f32>,
    /// Why the segment ended. A [`SegmentEnd::MaximumLength`] cut is mid-sentence by definition,
    /// so a consumer can choose to keep the transcript open rather than treating it as a turn.
    pub end: SegmentEnd,
    /// It begins with audio repeated from the piece before, which was cut with no breath to cut
    /// at. Words heard on both sides of that seam were said once.
    pub overlaps_previous: bool,
}

impl SpeechSegment {
    pub fn duration_ms(&self) -> usize {
        self.samples.len() * 1000 / SAMPLE_RATE as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentEnd {
    /// The speaker stopped and stayed stopped. A real endpoint.
    Silence,
    /// A short pause between phrases. The turn is still open and more is expected.
    Pause,
    /// The hard length limit fired; the speaker is probably still going.
    MaximumLength,
    /// The stream ended (device closed, or the caller flushed).
    StreamEnd,
}

/// What one frame produced.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SegmentOutcome {
    /// Audio ready to transcribe, if a piece completed on this frame.
    pub chunk: Option<SpeechSegment>,
    /// The speaker has finished. Stop taking input and start answering.
    pub turn_ended: bool,
}

/// A gap long enough to cut a long stretch of speech at: over 98 ms, the gap Silero's own
/// `get_speech_timestamps` splits over-long speech at (its `min_silence_at_max_speech`). Four
/// 32 ms frames is the first count past it.
const BREATH_FRAMES: usize = 98 / (VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize) + 1;

/// Frames in a segment that actually carry speech, as opposed to padding.
///
/// A segment is padded at both ends so a transcriber gets the leading and trailing consonants,
/// and the minimum-length rule used to be applied to the padded total. That made the rule far
/// weaker than it reads: with the padding this app uses, 72 ms of speech was enough to produce a
/// 468 ms segment and be accepted as an utterance. A cough, a door, a keyboard - anything with
/// one loud frame in it - cleared a bar meant to require a third of a second of speech.
///
/// After onset, count at the exit threshold so quiet syllables accepted by hysteresis count.
fn voiced_frames(probabilities: &[f32], threshold: f32) -> usize {
    probabilities
        .iter()
        .filter(|probability| **probability >= threshold)
        .count()
}

/// Turns a stream of `(frame, probability)` pairs into utterances.
///
/// Kept free of audio I/O so the policy can be tested against synthetic probability sequences,
/// which is the only practical way to cover the boundary cases.
pub struct Segmenter {
    config: SegmenterConfig,
    /// Frames not yet accepted into an utterance. This is the pre-roll.
    pending: VecDeque<Vec<f32>>,
    /// Frames belonging to the utterance currently open.
    active: Vec<Vec<f32>>,
    /// Probability for each frame in `active`, kept in step with it.
    active_probabilities: Vec<f32>,
    /// Pre-roll protects consonants but cannot count as evidence confirming an onset.
    active_start: usize,
    pending_probabilities: VecDeque<f32>,
    speech_run: usize,
    silence_run: usize,
    /// Longest silence in the current turn that was followed by more speech. The endpoint
    /// already tolerated it, so it only says the endpoint must not fall below it.
    longest_pause: usize,
    /// Frames of silence since the last turn ended, while none is open. A speaker who resumes
    /// after only a few of these was not finished, and the endpoint that cut them was short.
    since_turn_end: Option<usize>,
    /// Frames of the turn currently open, counted across the pieces it was cut into.
    turn_frames: usize,
    /// The piece being gathered repeats audio from the one before it: see
    /// [`SpeechSegment::overlaps_previous`].
    overlap_pending: bool,
    open: bool,
}

impl Segmenter {
    pub fn new(config: SegmenterConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            active: Vec::new(),
            active_probabilities: Vec::new(),
            active_start: 0,
            pending_probabilities: VecDeque::new(),
            speech_run: 0,
            silence_run: 0,
            longest_pause: 0,
            since_turn_end: None,
            turn_frames: 0,
            overlap_pending: false,
            open: false,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Feeds one frame, reporting both completed audio and whether the turn ended.
    pub fn push_turn(&mut self, frame: Vec<f32>, probability: f32) -> SegmentOutcome {
        if self.open {
            self.push_open(frame, probability)
        } else {
            SegmentOutcome {
                chunk: self.push_closed(frame, probability),
                turn_ended: false,
            }
        }
    }

    fn push_closed(&mut self, frame: Vec<f32>, probability: f32) -> Option<SpeechSegment> {
        if let Some(gap) = self.since_turn_end.as_mut() {
            *gap = gap.saturating_add(1);
        }
        self.pending.push_back(frame);
        self.pending_probabilities.push_back(probability);
        while self.pending.len() > self.config.preroll_frames.max(1) {
            self.pending.pop_front();
            self.pending_probabilities.pop_front();
        }

        if probability >= self.config.enter_threshold {
            self.speech_run += 1;
        } else {
            self.speech_run = 0;
        }

        if self.speech_run >= self.config.onset_frames {
            // Someone who carries on a moment after being cut off was mid-thought, and the
            // silence that ended their turn was part of that thought. Their real pause was the
            // endpoint plus the gap, and the endpoint has to clear it or the next sentence is
            // cut in the same place. This is the only view of a pause too long to observe from
            // inside a turn.
            if let Some(gap) = self.since_turn_end.take() {
                let resumed = gap.saturating_sub(self.speech_run);
                if resumed <= self.config.resume_window_frames {
                    let pause = self.config.endpoint.frames().saturating_add(resumed);
                    self.config.endpoint.observe(pause);
                }
            }
            // Adopt the whole pre-roll, which already contains the frames that triggered onset.
            self.active = self.pending.drain(..).collect();
            self.active_probabilities = self.pending_probabilities.drain(..).collect();
            self.active_start = self
                .active_probabilities
                .len()
                .saturating_sub(self.speech_run);
            self.open = true;
            self.silence_run = 0;
        }
        None
    }

    fn push_open(&mut self, frame: Vec<f32>, probability: f32) -> SegmentOutcome {
        self.turn_frames += 1;
        self.active.push(frame);
        self.active_probabilities.push(probability);

        if probability >= self.config.exit_threshold {
            // Speech resumed, so whatever silence preceded it was a pause inside the turn,
            // not the end of one. That is the only honest measurement of how long this
            // speaker leaves between thoughts.
            self.longest_pause = self.longest_pause.max(self.silence_run);
            self.silence_run = 0;
        } else {
            self.silence_run += 1;
        }

        // Past the cap, end on the next real gap rather than mid-word. Cutting a sentence in
        // half loses it entirely: the speaker carries straight on, that continuation opens a
        // new turn, and the half already captured is dropped with the utterance it belonged to.
        // A person pauses within a second or two of passing a minute; a television does not,
        // which is what the hard stop below is for.
        let endpoint = if self.turn_frames >= self.config.max_turn_frames {
            self.config
                .endpoint
                .frames()
                .min(SegmenterConfig::frames_for_ms(250))
        } else {
            self.config.endpoint.frames()
        };

        // The turn is over. Everything still buffered goes out with it.
        if self.silence_run >= endpoint {
            let chunk = self.close(SegmentEnd::Silence);
            return SegmentOutcome {
                chunk,
                turn_ended: true,
            };
        }

        // Nothing resembling a gap has arrived even after the endpoint was tightened, so this
        // is not a person: a television, a call on speakerphone, a microphone left open.
        // Answer what was heard rather than listening forever.
        if self.turn_frames >= self.config.max_turn_frames * 2 {
            let chunk = self.close(SegmentEnd::MaximumLength);
            return SegmentOutcome {
                chunk,
                turn_ended: true,
            };
        }

        let voiced = voiced_frames(
            &self.active_probabilities[self.active_start..],
            self.config.exit_threshold,
        );

        // Long enough, and the speaker has just drawn breath: hand this much over now, while they
        // carry on, instead of recognising all of it after they stop. The breath is silence, so
        // it is given to both sides - the end of this piece, and the pre-roll of the next, where
        // the first sound after it would otherwise land a frame before the detector rates it
        // speech.
        if self.active.len() >= self.config.rolling_frames
            && self.silence_run == BREATH_FRAMES
            && voiced > self.config.minimum_frames
        {
            let chunk = self.cut(SegmentEnd::Pause, self.active.len(), BREATH_FRAMES);
            return SegmentOutcome {
                chunk,
                turn_ended: false,
            };
        }

        // No breath came. Cut anyway, where the voice was quietest since the piece could first
        // have been cut - the likeliest place for a boundary between words - and repeat a little
        // across the seam, because it may still fall inside one.
        if self.active.len() >= self.config.maximum_frames {
            let from = self.config.rolling_frames.min(self.active.len() - 1);
            let split = from + quietest(&self.active_probabilities[from..]);
            let chunk = self.cut(
                SegmentEnd::MaximumLength,
                split.max(1),
                self.config.overlap_frames,
            );
            return SegmentOutcome {
                chunk,
                turn_ended: false,
            };
        }

        // A pause between phrases. Emit exactly once, on the frame the threshold is crossed -
        // `==` not `>=`, or every further silent frame would emit another empty piece.
        if self.silence_run == self.config.chunk_pause_frames && voiced > self.config.minimum_frames
        {
            let chunk = self.cut(SegmentEnd::Pause, self.active.len(), 0);
            return SegmentOutcome {
                chunk,
                turn_ended: false,
            };
        }

        SegmentOutcome {
            chunk: None,
            turn_ended: false,
        }
    }

    /// Hands the first `split` frames of the piece to the recogniser while the turn stays open.
    /// The last `carry` of them also begin the next piece.
    fn cut(&mut self, end: SegmentEnd, split: usize, carry: usize) -> Option<SpeechSegment> {
        let mut frames = std::mem::take(&mut self.active);
        let mut probabilities = std::mem::take(&mut self.active_probabilities);
        let start = std::mem::take(&mut self.active_start);
        let split = split.min(frames.len());
        let carry = carry.min(split);
        self.active = frames[split - carry..].to_vec();
        self.active_probabilities = probabilities[split - carry..].to_vec();
        frames.truncate(split);
        probabilities.truncate(split);
        // Audio carried across a cut with no breath in it may hold part of a word, heard again
        // at the start of the next piece; carried silence repeats nothing.
        let overlaps_previous = std::mem::replace(
            &mut self.overlap_pending,
            end == SegmentEnd::MaximumLength && carry > 0,
        );
        // Trim the pause that triggered the cut, keeping the post-roll. Handing the recogniser
        // the silence costs real time - it charges by audio length - and buys nothing.
        if end == SegmentEnd::Pause {
            let keep = frames
                .len()
                .saturating_sub(self.silence_run.saturating_sub(self.config.postroll_frames))
                .max(1);
            frames.truncate(keep);
            probabilities.truncate(keep);
        }
        if voiced_frames(
            &probabilities[start.min(probabilities.len())..],
            self.config.exit_threshold,
        ) < self.config.minimum_frames
        {
            return None;
        }
        let mut samples = Vec::with_capacity(frames.len() * VAD_FRAME_SAMPLES);
        for frame in frames {
            samples.extend_from_slice(&frame);
        }
        Some(SpeechSegment {
            samples,
            probabilities,
            end,
            overlaps_previous,
        })
    }

    /// Ends any open utterance, for a closing device or an explicit flush.
    pub fn flush(&mut self) -> Option<SpeechSegment> {
        if self.open {
            self.close(SegmentEnd::StreamEnd)
        } else {
            None
        }
    }

    fn close(&mut self, end: SegmentEnd) -> Option<SpeechSegment> {
        let mut frames = std::mem::take(&mut self.active);
        let mut probabilities = std::mem::take(&mut self.active_probabilities);
        let start = std::mem::take(&mut self.active_start);
        let overlaps_previous = std::mem::take(&mut self.overlap_pending);
        self.open = false;
        self.speech_run = 0;
        self.turn_frames = 0;
        let trailing = self.silence_run;
        self.silence_run = 0;
        if end == SegmentEnd::Silence {
            // Only a turn that ended on its own says anything about endpoint length. A cut at
            // the maximum length, or a flush at shutdown, does not.
            if self.longest_pause > 0 {
                self.config.endpoint.observe(self.longest_pause);
            } else {
                self.config.endpoint.relax();
            }
            // Start counting the gap: if the speaker resumes shortly, this ending was wrong.
            self.since_turn_end = Some(0);
        }
        self.longest_pause = 0;
        self.pending.clear();
        self.pending_probabilities.clear();

        // Trim the silence that proved the utterance was over, but keep the post-roll so the
        // last consonant is not clipped.
        if end == SegmentEnd::Silence {
            let keep = frames
                .len()
                .saturating_sub(trailing.saturating_sub(self.config.postroll_frames));
            frames.truncate(keep.max(1));
            probabilities.truncate(keep.max(1));
        }

        if voiced_frames(
            &probabilities[start.min(probabilities.len())..],
            self.config.exit_threshold,
        ) < self.config.minimum_frames
        {
            return None;
        }

        let mut samples = Vec::with_capacity(frames.len() * VAD_FRAME_SAMPLES);
        for frame in frames {
            samples.extend_from_slice(&frame);
        }
        Some(SpeechSegment {
            samples,
            probabilities,
            end,
            overlaps_previous,
        })
    }
}

/// Index of the frame the detector was least sure was speech: in unbroken speech, the likeliest
/// boundary between words. The first, if several are equally quiet.
fn quietest(probabilities: &[f32]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index)
}

// ============================================================================================
// Semantic chunking
// ============================================================================================

/// How to cut a long utterance into pieces the recogniser handles well.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Preferred chunk length. Splitting looks for a pause near here.
    pub target_ms: usize,
    /// Hard limit. If no pause is found before this, the chunk is cut anyway.
    pub maximum_ms: usize,
    /// Never produce a piece shorter than this; a fragment transcribes worse than the whole.
    pub minimum_ms: usize,
    /// Probability below which a frame is a candidate split point.
    pub pause_threshold: f32,
    /// Audio repeated on both sides of a cut.
    ///
    /// A transcriber uses surrounding audio for context, so a word sitting exactly on a boundary
    /// would otherwise be half-heard on each side and recognised on neither.
    pub overlap_ms: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_ms: 6_000,
            maximum_ms: 10_000,
            minimum_ms: 1_500,
            pause_threshold: 0.35,
            overlap_ms: 200,
        }
    }
}

/// One piece of an utterance, as a sample range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
    /// True when the cut landed on a real pause rather than the hard limit: what the planner's
    /// tests check it by. Joining needs no such distinction here - every seam the planner makes
    /// overlaps, so every one is joined with `asr::join_overlapping`.
    pub split_on_pause: bool,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn duration_ms(&self) -> usize {
        self.len() * 1000 / SAMPLE_RATE as usize
    }
}

/// Splits an utterance at its quietest internal moments.
///
/// Long utterances are the ones that hurt: the recogniser is less reliable on a long input, and
/// one long job returns nothing until all of it is done. Pieces are recognised one after
/// another, each bounded, and each shows its words as soon as it finishes.
///
/// Cutting on *pauses* rather than on a timer is what keeps that from costing accuracy. An
/// arbitrary cut lands mid-word about as often as not, and both halves then transcribe badly.
pub fn plan_chunks(
    total_samples: usize,
    frame_probabilities: &[f32],
    config: &ChunkConfig,
) -> Vec<Chunk> {
    let per_ms = SAMPLE_RATE as usize / 1000;
    // Public configuration must guarantee forward progress, including zero/overflow inputs.
    let maximum_ms = config.maximum_ms.clamp(1, 30_000);
    let minimum_ms = config.minimum_ms.clamp(1, maximum_ms);
    let target = config.target_ms.clamp(minimum_ms, maximum_ms) * per_ms;
    let maximum = maximum_ms * per_ms;
    let minimum = minimum_ms * per_ms;
    let overlap = config.overlap_ms.min(minimum_ms / 2) * per_ms;

    if total_samples == 0 {
        return Vec::new();
    }
    if total_samples <= maximum {
        return vec![Chunk {
            start: 0,
            end: total_samples,
            split_on_pause: true,
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < total_samples {
        let remaining = total_samples - start;
        if remaining <= maximum {
            chunks.push(Chunk {
                start,
                end: total_samples,
                split_on_pause: true,
            });
            break;
        }

        // Look for the deepest pause between the earliest acceptable cut and the hard limit,
        // preferring one near the target length.
        let window_start = start + minimum.min(remaining);
        let window_end = (start + maximum).min(total_samples);
        let (cut, on_pause) = find_pause(
            frame_probabilities,
            window_start,
            window_end,
            start + target,
            config.pause_threshold,
        );

        chunks.push(Chunk {
            start,
            end: cut,
            split_on_pause: on_pause,
        });
        // Step back by the overlap so a word on the seam is heard whole at least once.
        start = cut.saturating_sub(overlap).max(cut.min(start + 1));
    }

    chunks
}

/// Finds the quietest frame in a sample range, biased toward `preferred`.
fn find_pause(
    probabilities: &[f32],
    window_start: usize,
    window_end: usize,
    preferred: usize,
    threshold: f32,
) -> (usize, bool) {
    if probabilities.is_empty() || window_end <= window_start {
        return (window_end, false);
    }
    let first = window_start / VAD_FRAME_SAMPLES;
    let last = (window_end / VAD_FRAME_SAMPLES).min(probabilities.len());
    if last <= first {
        return (window_end, false);
    }

    let mut best: Option<(usize, f32)> = None;
    for (index, &probability) in probabilities.iter().enumerate().take(last).skip(first) {
        if probability > threshold {
            continue;
        }
        let position = index * VAD_FRAME_SAMPLES;
        // Distance from the target length, scaled so a slightly deeper pause far from the target
        // does not beat a good pause right where the cut belongs.
        let distance = position.abs_diff(preferred) as f32 / SAMPLE_RATE as f32;
        let score = probability + distance * 0.05;
        if best.is_none_or(|(_, current)| score < current) {
            best = Some((position, score));
        }
    }

    match best {
        // `>=`, not `>`: the window already starts a whole minimum chunk past the cut before
        // it, so a pause sitting exactly on its first frame is real progress and a real pause.
        // Requiring one strictly inside threw that away and cut at the hard limit instead -
        // mid-word, and marked untrustworthy - despite having found the breath it was after.
        Some((position, _)) if position >= window_start => (position, true),
        // No pause anywhere in range: this speaker has not drawn breath. Cut at the limit and
        // mark it, so the consumer knows the seam is untrustworthy.
        _ => (window_end, false),
    }
}

// ============================================================================================
// Errors
// ============================================================================================

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("model file is missing: {0}")]
    MissingModel(String),
    #[error("onnx runtime error: {0}")]
    Onnx(String),
    #[error("expected a {expected}-sample frame but got {actual}")]
    FrameSize { expected: usize, actual: usize },
    #[error("resampler error: {0}")]
    Resample(String),
}

// ============================================================================================
// Output loudness
// ============================================================================================

/// Four-times oversampling filter, one row per fractional instant between two samples.
///
/// This is the measurement ITU-R BS.1770-4 defines for true peak: reconstruct the waveform at
/// quarter-sample steps and take the peak of that, rather than of the samples. The standard
/// tabulates its coefficients for 48 kHz input, so these are the same design - a windowed sinc,
/// twelve taps per phase, unity gain at DC - solved for the synthesiser's 24 kHz. The window is
/// Kaiser with beta 9, picked by sweeping the band: at that value the detector reads a full
/// scale tone at a quarter of the sampling rate - the classic worst case, where the samples sit
/// at 0.707 and the wave between them reaches 1.0 - as 1.00000.
const TRUE_PEAK_PHASES: usize = 4;
const TRUE_PEAK_TAPS: usize = 12;
const TRUE_PEAK_FILTER: [[f32; TRUE_PEAK_TAPS]; TRUE_PEAK_PHASES] = [
    // The sample itself: nothing to reconstruct at a whole-sample instant.
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        -0.000604485,
        0.004496868,
        -0.018047016,
        0.053939503,
        -0.14949417,
        0.89368963,
        0.28078428,
        -0.08898283,
        0.03197014,
        -0.009475104,
        0.001847544,
        -0.000124349,
    ],
    [
        -0.000423694,
        0.00416595,
        -0.018690377,
        0.05899672,
        -0.16212623,
        0.61807764,
        0.61807764,
        -0.16212623,
        0.05899672,
        -0.018690377,
        0.00416595,
        -0.000423694,
    ],
    [
        -0.000124349,
        0.001847544,
        -0.009475104,
        0.03197014,
        -0.08898283,
        0.28078428,
        0.89368963,
        -0.14949417,
        0.053939503,
        -0.018047016,
        0.004496868,
        -0.000604485,
    ],
];

/// Makeup gain and a true-peak limiter for synthesised speech.
///
/// Two things separate this from holding each sample under a number.
///
/// The first is what it measures. A sample-peak limiter protects the samples it is handed, but
/// nothing plays those samples: the page resamples 24 kHz to whatever the device runs at, and
/// what the listener hears are the peaks of the waveform reconstructed between them. Those sit
/// above the samples either side, so audio limited to just under full scale comes back over it
/// and clips downstream, past anything this code can reach. Measured over a two-minute session,
/// the engine peaked at 0.9799 and sent nothing at full scale while the browser's own output
/// clipped 224 samples across 39 events, clustered on the loudest syllables and heard as a
/// crackle over the voice. So the detector oversamples before it looks. Being a measurement
/// rather than an allowance, it holds for any output rate and any content, which guessing at
/// headroom does not.
///
/// The second is how it responds. Bending each sample toward a ceiling is a waveshaper: it
/// changes the shape of the wave, and the harmonics that adds are heard as roughness on exactly
/// the loudest, most sustained sounds - in speech, the vowels. Following the peak and applying
/// one smooth gain to the passage leaves the shape alone and only moves the level, which a
/// listener does not notice. Look-ahead is what makes that possible: the reduction a peak needs
/// is known a millisecond before the peak arrives, so the gain slides down to meet it instead
/// of stepping. Release is slow, so the gain does not lurch back up between syllables and pump.
pub struct Loudness {
    gain: f32,
    /// Gain reduction currently applied, where 1.0 is none.
    envelope: f32,
    /// Amplified samples taken in but not yet handed back, oldest first.
    window: [f32; Self::WINDOW],
    /// The gain each instant still inside the look-ahead will need, oldest first.
    pending: [f32; Self::LOOKAHEAD + 1],
}

impl Loudness {
    /// True-peak ceiling, -1 dBTP.
    ///
    /// The broadcast figure, and the reasoning behind it applies here unchanged. Four-times
    /// oversampling does not see quite all of the waveform: swept across the band, the detector
    /// above reads at worst 0.436 dB low, up near the top of the range where a 24 kHz stream has
    /// little energy anyway. Applying the envelope per sample rather than per window costs a
    /// little more. One decibel covers both with room to spare. Unlike the sample-peak ceiling
    /// this replaces, it bounds what the listener's device will actually have to reproduce.
    const CEILING: f32 = 0.891_251; // 10^(-1/20)
    /// Gain for Zen's voice: -15.1 LUFS, a decibel and a half louder than the -16.6 of 1.65,
    /// which sat at the -16 that spoken-word streaming is normalised to. At that level Zen
    /// sounded quiet on laptop speakers, so it was raised (2026-09-23).
    ///
    /// Measured by ITU-R BS.1770-4 over eight reply-shaped passages, 93 s of speech
    /// (`examples/loudness.rs`, 2026-09-18). The synthesiser itself gives -20.9 LUFS with true
    /// peaks at -3.4 dBTP, so any gain past about 1.5 meets the ceiling somewhere. What matters
    /// is how often, because a limiter working on every loud syllable is heard as loud and flat
    /// however correct its ceiling:
    ///
    /// | gain | loudness   | held down > 1 dB | deepest |
    /// |------|------------|------------------|---------|
    /// | 1.65 | -16.6 LUFS | 0.4 % of speech  | 1.9 dB  |
    /// | 2.00 | -15.1 LUFS | 7.4 %            | 3.6 dB  |
    /// | 2.20 | -14.5 LUFS | 15.2 %           | 4.4 dB  |
    /// | 2.40 | -14.1 LUFS | 25.6 %           | 5.2 dB  |
    ///
    /// At 2.0 the limiter holds the loudest 7.4 % of speech down by more than a decibel, which
    /// is still heard as a voice rather than as a limiter. Past it each step buys little - 2.2
    /// adds 0.6 dB for twice the squeezing - and by 2.4 it is loud and flat.
    ///
    /// Louder playback is also more of Zen's own voice for the echo canceller to remove. 2.0 was
    /// first tried on 2026-09-18 and taken back when Zen cut its own reply off on laptop
    /// speakers; the capture chain had no anti-aliasing filter then. Re-measured on the same
    /// laptop with Windows at full volume (2026-09-19, three long replies at each gain): no
    /// self-interruption at 1.65 or 2.0, and Silero's highest score on what the canceller left
    /// of Zen's voice was 0.06 and 0.15, against the 0.5 it takes to interrupt.
    pub const DEFAULT_GAIN: f32 = 2.0;
    /// Range a caller may ask for. Below 0.5 the assistant is inaudible; above 4.0 the limiter
    /// is holding the level down through every syllable and the result is loud and flat.
    pub const MIN_GAIN: f32 = 0.5;
    pub const MAX_GAIN: f32 = 4.0;
    /// Look-ahead, 1 ms at 24 kHz: how long the gain gets to reach a peak before it lands.
    const LOOKAHEAD: usize = TTS_SAMPLE_RATE as usize / 1000;
    /// Samples held at once: the look-ahead, the instant being measured, and the taps the
    /// detector needs beyond it.
    const WINDOW: usize = Self::LOOKAHEAD + TRUE_PEAK_TAPS / 2 + 1;
    /// Where the detector's filter starts, so its centre lands one look-ahead ahead of the
    /// sample about to be handed back.
    const TAP_BASE: usize = Self::LOOKAHEAD + 1 - TRUE_PEAK_TAPS / 2;
    /// Delay from a sample going in to that sample coming out.
    pub const LATENCY: usize = Self::WINDOW - 1;
    /// Enough per sample to cross the whole range of gain reduction within the look-ahead, so
    /// the envelope always arrives at its target before the sample needing it does. Linear for
    /// that reason: a smoothed attack approaches its target without reaching it, and whatever it
    /// has not covered by the deadline is an overshoot.
    const ATTACK_STEP: f32 = 1.0 / Self::LOOKAHEAD as f32;
    /// Per-sample recovery, about 150 ms at 24 kHz. Slow enough not to pump between words,
    /// fast enough that a single loud syllable does not hold the whole reply down.
    const RELEASE: f32 = 1.0 / (0.150 * TTS_SAMPLE_RATE as f32);

    pub fn new(gain: f32) -> Self {
        Self {
            gain: gain.clamp(Self::MIN_GAIN, Self::MAX_GAIN),
            envelope: 1.0,
            window: [0.0; Self::WINDOW],
            pending: [1.0; Self::LOOKAHEAD + 1],
        }
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Hands back what the look-ahead is still holding, and starts the next passage clean.
    ///
    /// The delay this limiter runs at is not free. Without draining it the last 1.25 ms of
    /// every phrase is never sent at all, and the ending fade the page applies then lands on a
    /// waveform already missing its end - which is the one place a fade exists to guarantee
    /// silence. Call this when synthesis for a phrase has finished. On a barge-in call
    /// [`Self::reset`] instead: that audio is not wanted.
    pub fn finish(&mut self) -> Vec<f32> {
        let mut tail = vec![0.0; Self::LATENCY];
        self.process(&mut tail);
        self.reset();
        tail
    }

    /// Drops everything held back and starts again with the limiter idle.
    ///
    /// Called at the start of every phrase, so gain reduction earned at the end of one does not
    /// start the next one quiet, and on a barge-in, where the audio still held is not wanted.
    /// A phrase that ended normally has already been drained by [`Self::finish`].
    pub fn reset(&mut self) {
        self.envelope = 1.0;
        self.window = [0.0; Self::WINDOW];
        self.pending = [1.0; Self::LOOKAHEAD + 1];
    }

    /// The peak of the waveform these samples stand for, including between them.
    fn true_peak(&self) -> f32 {
        let mut peak = 0.0f32;
        for phase in &TRUE_PEAK_FILTER {
            let mut instant = 0.0f32;
            for (tap, weight) in phase.iter().enumerate() {
                instant += weight * self.window[Self::TAP_BASE + tap];
            }
            peak = peak.max(instant.abs());
        }
        peak
    }

    /// Applies gain in place, holding the reconstructed peak under the ceiling by moving the
    /// level rather than by reshaping the wave.
    ///
    /// Samples come back delayed by [`Self::LATENCY`], which is the look-ahead the detector
    /// needs. The count is unchanged and the delay is constant, so a stream stays seamless
    /// across calls; only the boundary between one passage and the next needs [`Self::reset`].
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            self.window.copy_within(1.., 0);
            self.window[Self::WINDOW - 1] = *sample * self.gain;
            // Read after the shift, so the sample going out is the one whose own instant is
            // still the oldest entry in `pending` rather than one older than all of them.
            let due = self.window[0];

            let peak = self.true_peak();
            let required = if peak > Self::CEILING {
                Self::CEILING / peak
            } else {
                1.0
            };
            self.pending.copy_within(1.., 0);
            self.pending[Self::LOOKAHEAD] = required;

            // The sample going out now has to already be under whatever the loudest instant
            // still ahead of it will need, so the reduction is in place before that peak lands.
            let target = self.pending.iter().copied().fold(1.0f32, f32::min);
            self.envelope = if target < self.envelope {
                (self.envelope - Self::ATTACK_STEP).max(target)
            } else {
                (self.envelope + (target - self.envelope) * Self::RELEASE).min(1.0)
            };
            // The envelope already holds this under the ceiling; the clamp is only a guard
            // against a non-finite sample reaching the conversion to 16-bit.
            *sample = (due * self.envelope).clamp(-1.0, 1.0);
        }
    }
}

impl Default for Loudness {
    fn default() -> Self {
        Self::new(Self::DEFAULT_GAIN)
    }
}

// ============================================================================================
// Pipeline
// ============================================================================================

/// What the capture chain reports: someone has started speaking, a piece of what they said is
/// ready for recognition, or they have finished.
pub enum CaptureEvent {
    Started,
    Segment(SpeechSegment),
    Ended,
}

/// The whole capture chain: 16 kHz microphone samples in, speech segments out.
///
/// Echo cancellation, noise suppression and gain control belong to the webview, which owns the
/// device and knows what it just played; this side owns only the decision of when someone is
/// speaking. Running the WebRTC chain again here would process the same audio twice, which
/// pumps noise and strips quiet consonants the recogniser needs.
pub struct CapturePipeline {
    vad: SileroVad,
    segmenter: Segmenter,
    /// Samples not yet forming a whole VAD frame.
    pending: Vec<f32>,
    /// A candidate may open the segmenter without interrupting the current reply.
    started: bool,
}

impl CapturePipeline {
    pub fn new(segmenter_config: SegmenterConfig) -> Result<Self, AudioError> {
        Ok(Self {
            vad: SileroVad::embedded()?,
            segmenter: Segmenter::new(segmenter_config),
            pending: Vec::new(),
            started: false,
        })
    }

    /// Feeds 16 kHz mono samples, returning the turn boundaries and utterances they completed.
    pub fn push_events(&mut self, samples: &[f32]) -> Result<Vec<CaptureEvent>, AudioError> {
        self.pending.extend_from_slice(samples);
        let mut events = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= VAD_FRAME_SAMPLES {
            let frame = self.pending[offset..offset + VAD_FRAME_SAMPLES].to_vec();
            offset += VAD_FRAME_SAMPLES;
            let probability = self.vad.probability(&frame)?;
            self.process_frame(frame, probability, &mut events);
        }
        self.pending.drain(..offset);
        Ok(events)
    }

    fn process_frame(&mut self, frame: Vec<f32>, probability: f32, events: &mut Vec<CaptureEvent>) {
        let outcome = self.segmenter.push_turn(frame, probability);
        let confirmed = self.segmenter.is_open()
            && voiced_frames(
                &self.segmenter.active_probabilities[self.segmenter.active_start..],
                self.segmenter.config.exit_threshold,
            ) >= self.segmenter.config.minimum_frames;
        if !self.started && (confirmed || outcome.chunk.is_some()) {
            self.started = true;
            events.push(CaptureEvent::Started);
        }
        if let Some(segment) = outcome.chunk {
            events.push(CaptureEvent::Segment(segment));
        }
        if outcome.turn_ended {
            self.vad.reset();
            if self.started {
                events.push(CaptureEvent::Ended);
            }
            self.started = false;
        }
    }

    pub fn flush(&mut self) -> Option<SpeechSegment> {
        // Samples left over from the last partial frame have never been shown to the detector,
        // because it only reads whole frames - so without this they are simply dropped. That is
        // up to 32 ms off the end of the last thing said, which is exactly where a final
        // consonant lives.
        //
        // They are appended to the segment rather than padded up to a frame and pushed through
        // the detector: the segment has already been accepted, so its opinion of these samples
        // would change nothing, and padding would put silence the speaker never made into audio
        // the recogniser is charged by the length of. With nothing open there is no utterance
        // for them to belong to and they are discarded, which is what they are.
        let remainder = std::mem::take(&mut self.pending);
        self.started = false;
        let mut segment = self.segmenter.flush()?;
        segment.samples.extend_from_slice(&remainder);
        Some(segment)
    }

    /// Starts the endpoint from a pause learned earlier, without rebuilding the pipeline.
    pub fn set_endpoint(&mut self, endpoint: EndpointPolicy) {
        self.segmenter.config.endpoint = endpoint;
        self.reset_capture();
    }

    /// What the endpoint currently is, in milliseconds. It moves as the speaker is measured, and
    /// the interface shows it so the behaviour is legible rather than mysterious.
    pub fn endpoint_ms(&self) -> usize {
        self.segmenter.config.endpoint.frames() * VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize
    }

    pub fn reset_capture(&mut self) {
        // `config` carries the learned endpoint, so rebuilding from it keeps what the
        // speaker has already demonstrated about their own pauses.
        self.segmenter = Segmenter::new(self.segmenter.config);
        self.vad.reset();
        self.pending.clear();
        self.started = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_probabilities(probabilities: &[f32]) -> Vec<CaptureEvent> {
        let mut pipeline = CapturePipeline::new(SegmenterConfig::default()).unwrap();
        let mut events = Vec::new();
        for &probability in probabilities {
            pipeline.process_frame(vec![0.1; VAD_FRAME_SAMPLES], probability, &mut events);
        }
        events
    }

    #[test]
    fn an_unconfirmed_spike_never_interrupts_the_reply() {
        for background in [0.01, 0.4] {
            let mut probabilities = vec![background; 12];
            probabilities.extend([0.9; 2]);
            probabilities.extend([0.01; 100]);
            assert!(
                capture_probabilities(&probabilities).is_empty(),
                "background {background}"
            );
        }
    }

    #[test]
    fn a_short_answer_reaches_asr_with_matching_turn_boundaries() {
        let mut probabilities = vec![0.9; 6]; // 192 ms, e.g. a brief yes/no.
        probabilities.extend([0.01; 100]);
        let events = capture_probabilities(&probabilities);
        assert!(matches!(events.first(), Some(CaptureEvent::Started)));
        assert!(matches!(events.last(), Some(CaptureEvent::Ended)));
        let segments: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                CaptureEvent::Segment(segment) => Some(segment),
                _ => None,
            })
            .collect();
        assert_eq!(segments.len(), 1);
        assert_eq!(voiced_frames(&segments[0].probabilities, 0.5), 6);
    }

    #[test]
    fn quieter_syllables_after_onset_count_as_speech() {
        let mut probabilities = vec![0.9; 2];
        probabilities.extend([0.4; 30]);
        probabilities.extend([0.01; 100]);
        let events = capture_probabilities(&probabilities);
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, CaptureEvent::Started))
                .count(),
            1
        );
        assert!(events.iter().any(|e| matches!(e, CaptureEvent::Segment(_))));
        assert!(matches!(events.last(), Some(CaptureEvent::Ended)));
    }

    #[test]
    fn malformed_chunk_limits_still_terminate_with_complete_coverage() {
        for value in [0, 1, usize::MAX] {
            let chunks = plan_chunks(
                16000,
                &[],
                &ChunkConfig {
                    target_ms: value,
                    maximum_ms: value,
                    minimum_ms: value,
                    overlap_ms: value,
                    pause_threshold: f32::NAN,
                },
            );
            assert!(!chunks.is_empty());
            assert_eq!(chunks[0].start, 0);
            assert_eq!(chunks.last().unwrap().end, 16000);
            assert!(chunks.iter().all(|c| c.start < c.end && c.end <= 16000));
            assert!(chunks
                .windows(2)
                .all(|pair| pair[1].start > pair[0].start && pair[1].start <= pair[0].end));
        }
    }

    /// The most a sample may sit above the ceiling: one step of float rounding.
    ///
    /// The limiter aims at the ceiling rather than under it, and the gain it applies is
    /// computed as the ceiling divided by the peak, so a sample landing exactly on the ceiling
    /// can come back a single representable step above it. That is arithmetic, not overshoot.
    const OVER: f32 = f32::EPSILON;

    /// Runs a passage through the limiter and lines the result back up with its input.
    ///
    /// The limiter holds samples back to see what the waveform does between them, so getting
    /// the whole passage out means pushing silence in behind it, and comparing like with like
    /// means dropping the delay off the front.
    fn through(loudness: &mut Loudness, input: &[f32]) -> Vec<f32> {
        let mut samples = input.to_vec();
        samples.resize(input.len() + Loudness::LATENCY, 0.0);
        loudness.process(&mut samples);
        samples.drain(..Loudness::LATENCY);
        samples
    }

    /// Reconstructs the waveform between samples the way a resampler does.
    ///
    /// Deliberately not the detector's filter: sixteen steps per sample and sixty-four taps
    /// against the detector's four and twelve, so what this measures is independent of what it
    /// is being used to check.
    fn reconstructed_peak(samples: &[f32]) -> f32 {
        const STEPS: usize = 16;
        const HALF: isize = 32;
        let mut peak = 0.0f32;
        for centre in HALF as usize..samples.len() - HALF as usize {
            for step in 0..STEPS {
                let offset = step as f64 / STEPS as f64;
                let mut instant = 0.0f64;
                for tap in -HALF + 1..=HALF {
                    let x = tap as f64 - offset;
                    let sinc = if x == 0.0 {
                        1.0
                    } else {
                        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                    };
                    let window = 0.5 * (1.0 + (std::f64::consts::PI * x / HALF as f64).cos());
                    instant += samples[(centre as isize + tap) as usize] as f64 * sinc * window;
                }
                peak = peak.max(instant.abs() as f32);
            }
        }
        peak
    }

    #[test]
    fn quiet_speech_is_amplified_linearly() {
        // Anything under the ceiling is scaled exactly and nothing else is done to it, which
        // is what keeps ordinary speech uncoloured.
        let got = through(&mut Loudness::new(2.0), &[0.05, -0.1, 0.2, -0.3]);
        for (got, want) in got.iter().zip([0.1, -0.2, 0.4, -0.6]) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn nothing_can_be_pushed_past_the_ceiling() {
        // The reason the limiter exists. Without it, enough gain to fix the average drives peaks
        // into the device clamp, and the loudest part of every sentence becomes distortion.
        for gain in [1.0, 1.8, 2.5, 4.0] {
            let ramp: Vec<f32> = (-20..=20).map(|i| i as f32 / 10.0).collect();
            for sample in through(&mut Loudness::new(gain), &ramp) {
                assert!(
                    sample.abs() <= Loudness::CEILING + OVER,
                    "gain {gain} produced {sample}"
                );
            }
        }
    }

    /// One second of a sine at `level`, at the synthesiser's rate.
    fn sine(level: f32, hz: f32) -> Vec<f32> {
        (0..TTS_SAMPLE_RATE as usize)
            .map(|i| {
                (level as f64
                    * (std::f64::consts::TAU * hz as f64 * i as f64 / TTS_SAMPLE_RATE as f64).sin())
                    as f32
            })
            .collect()
    }

    /// A tone at a quarter of the sampling rate, sampled on its zero crossings.
    ///
    /// The worst case for peaks between samples, and the one that motivates measuring them:
    /// every sample reads 0.707 of the amplitude and the wave between them reaches all of it.
    fn quarter_rate_tone(level: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                (level as f64
                    * (std::f64::consts::FRAC_PI_2 * i as f64 + std::f64::consts::FRAC_PI_4).sin())
                    as f32
            })
            .collect()
    }

    #[test]
    fn the_detector_sees_the_peak_between_samples_not_just_the_samples() {
        // Four-times oversampling recovers the whole amplitude of the worst case, so the number
        // the limiter works against is the one the listener will hear.
        let tone = quarter_rate_tone(1.0, 512);
        let sample_peak = tone.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            (sample_peak - 0.707).abs() < 0.01,
            "the samples themselves read {sample_peak}"
        );
        let mut detector = Loudness::new(1.0);
        let mut seen = 0.0f32;
        for (index, value) in tone.iter().enumerate() {
            let mut block = [*value];
            detector.process(&mut block);
            // Past the onset, where the step out of silence rings and legitimately overshoots.
            if index > Loudness::WINDOW * 2 {
                seen = seen.max(detector.true_peak());
            }
        }
        assert!(
            (seen - 1.0).abs() < 0.005,
            "the detector read the worst case as {seen}, not 1.0"
        );
    }

    #[test]
    fn what_the_page_reconstructs_stays_under_full_scale() {
        // The guarantee, checked the way it is meant: limit the signal, then reconstruct it the
        // way the page's resampler will, and confirm nothing in between the samples clips.
        //
        // The unlimited signal is the control. Its samples sit at 0.672 and any sample-peak
        // limiter would wave them through at a ceiling of 0.89 - and the listener would get
        // 1.109, which the device clamps, which is the crackle this replaced.
        let tone = quarter_rate_tone(0.95, 2048);
        let amplified: Vec<f32> = tone.iter().map(|s| s * Loudness::DEFAULT_GAIN).collect();
        let unprotected = reconstructed_peak(&amplified);
        assert!(
            unprotected > 1.0,
            "the control has to actually overshoot to mean anything: {unprotected}"
        );

        let limited = through(&mut Loudness::default(), &tone);
        let heard = reconstructed_peak(&limited);
        assert!(
            heard <= 1.0,
            "the listener would still hear clipping: {heard}"
        );
        assert!(
            heard > 0.8,
            "and the ceiling should be used, not hidden from: {heard}"
        );
    }

    /// How far a passage departs from a scaled copy of itself, as a fraction of its own level.
    ///
    /// A limiter that only moves the level leaves the shape alone, so the best-fitting scale
    /// accounts for nearly all of it. One that reshapes the wave cannot be fitted by any single
    /// scale, and the residue is the roughness a listener hears.
    fn shape_error(original: &[f32], processed: &[f32]) -> f32 {
        let (mut dot, mut square) = (0.0f64, 0.0f64);
        for (a, b) in original.iter().zip(processed) {
            dot += *a as f64 * *b as f64;
            square += *a as f64 * *a as f64;
        }
        let scale = dot / square;
        let (mut residue, mut energy) = (0.0f64, 0.0f64);
        for (a, b) in original.iter().zip(processed) {
            residue += (*b as f64 - scale * *a as f64).powi(2);
            energy += *b as f64 * *b as f64;
        }
        (residue / energy).sqrt() as f32
    }

    #[test]
    fn holding_a_peak_down_moves_the_level_without_reshaping_the_wave() {
        // The complaint this exists for: sustained loud sounds - vowels, held notes - came out
        // rough. Bending every sample toward a ceiling is a distortion effect, and speech peaks
        // sit far enough above the average that ordinary syllables were being bent.
        let tone = sine(0.55, 220.0);
        let limited = through(&mut Loudness::default(), &tone);
        assert!(
            limited.iter().all(|s| s.abs() <= Loudness::CEILING + OVER),
            "the ceiling must still hold"
        );
        let error = shape_error(&tone, &limited);
        assert!(
            error < 0.02,
            "the waveform was reshaped rather than turned down: {error}"
        );
    }

    #[test]
    fn a_louder_passage_still_comes_out_louder() {
        // Limiting must compress the range, never invert it.
        let level = |input: f32| {
            let samples = through(&mut Loudness::default(), &sine(input, 220.0));
            let energy: f64 = samples.iter().map(|s| *s as f64 * *s as f64).sum();
            (energy / samples.len() as f64).sqrt()
        };
        let (quiet, mid, loud) = (level(0.1), level(0.4), level(0.9));
        assert!(quiet < mid && mid < loud, "{quiet} {mid} {loud}");
    }

    #[test]
    fn the_level_recovers_after_a_peak_rather_than_staying_ducked() {
        // A limiter that never releases leaves everything after one loud syllable quiet.
        let mut input = sine(0.95, 220.0);
        input.extend(sine(0.2, 220.0));
        let quiet_alone = sine(0.2, 220.0);
        let samples = through(&mut Loudness::default(), &input);
        let tail = &samples[samples.len() - TTS_SAMPLE_RATE as usize / 4..];
        let reference: Vec<f32> = quiet_alone
            .iter()
            .rev()
            .take(tail.len())
            .rev()
            .copied()
            .collect();
        let level = |data: &[f32]| {
            (data.iter().map(|s| *s as f64 * *s as f64).sum::<f64>() / data.len() as f64).sqrt()
        };
        let recovered = level(tail) / (level(&reference) * Loudness::DEFAULT_GAIN as f64);
        assert!(
            recovered > 0.95,
            "the level was still held down a second later: {recovered}"
        );
    }

    #[test]
    fn the_gain_is_already_down_when_the_peak_arrives() {
        // What the look-ahead buys. Silence, then a loud passage starting without warning: the
        // reduction has to be in place on its first sample, not applied a moment after it.
        let mut input = vec![0.0f32; 256];
        input.extend(sine(0.95, 220.0).into_iter().take(2048));
        let limited = through(&mut Loudness::default(), &input);
        assert!(
            limited[..256].iter().all(|s| s.abs() < 1e-6),
            "silence must stay silent"
        );
        assert!(
            limited[256..]
                .iter()
                .all(|s| s.abs() <= Loudness::CEILING + OVER),
            "the onset overshot the ceiling"
        );
    }

    #[test]
    fn finishing_a_phrase_hands_back_its_last_millisecond() {
        // The look-ahead holds the end of every passage. Dropped, the page fades out audio
        // that is already missing its final samples, which defeats the point of the fade.
        let mut loudness = Loudness::default();
        let mut passage = vec![0.0f32; 480];
        // A mark right at the end, well inside what the look-ahead is still holding.
        passage[479] = 0.4;
        passage[470] = 0.4;
        loudness.process(&mut passage);
        assert!(
            passage.iter().all(|s| s.abs() < 1e-6),
            "the marks should still be inside the limiter, not in this block",
        );
        let tail = loudness.finish();
        assert_eq!(tail.len(), Loudness::LATENCY);
        let loudest = tail.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            (loudest - 0.4 * Loudness::DEFAULT_GAIN).abs() < 1e-3,
            "the end of the passage came back as {loudest}",
        );
    }

    #[test]
    fn finishing_leaves_the_limiter_ready_for_the_next_phrase() {
        let mut loudness = Loudness::default();
        let mut loud = sine(0.95, 220.0);
        loudness.process(&mut loud);
        let _ = loudness.finish();
        let mut next = vec![0.0f32; Loudness::LATENCY * 2];
        loudness.process(&mut next);
        assert!(
            next.iter().all(|s| *s == 0.0),
            "the previous phrase was still inside the limiter",
        );
    }

    #[test]
    fn a_reset_leaves_nothing_of_the_last_phrase_behind() {
        // Between phrases there is a pause, and a millisecond of the previous phrase arriving
        // after it would be heard as a tick rather than as speech.
        let mut loudness = Loudness::default();
        let mut loud = sine(0.95, 220.0);
        loudness.process(&mut loud);
        loudness.reset();
        let mut next = vec![0.0f32; Loudness::LATENCY * 2];
        loudness.process(&mut next);
        assert!(
            next.iter().all(|s| *s == 0.0),
            "the previous phrase leaked into the next one"
        );
    }

    #[test]
    fn the_waveform_keeps_its_shape_around_zero() {
        let samples = through(&mut Loudness::new(1.8), &[0.0, -0.0, 0.25, -0.25]);
        assert_eq!(samples[0], 0.0);
        assert!((samples[2] + samples[3]).abs() < 1e-6, "asymmetric");
    }

    #[test]
    fn the_measured_synthesiser_level_lands_where_the_default_gain_puts_it() {
        // Measured output: RMS about 0.09, -20.9 dBFS, and -20.9 LUFS by BS.1770-4 too
        // (`examples/loudness.rs`). The default gain lifts that average by 6 dB, to the -15.1
        // LUFS measured over whole passages.
        let rms = through(&mut Loudness::default(), &[0.09f32])[0];
        let db = 20.0 * rms.log10();
        assert!((-16.0..=-14.0).contains(&db), "average landed at {db} dBFS");
        // Ordinary peaks, about 0.53, now reach past the ceiling: the limiter takes them - the
        // 7.4 % of speech it works on at this gain - and none gets through.
        let peak = through(&mut Loudness::default(), &[0.53f32])[0];
        assert!(
            peak <= Loudness::CEILING + 1e-6,
            "a peak got past the ceiling: {peak}"
        );
        assert!(peak > 0.8, "real speech peaks should use the range: {peak}");
    }

    #[test]
    fn an_absurd_gain_request_is_clamped_rather_than_honoured() {
        assert_eq!(Loudness::new(99.0).gain(), Loudness::MAX_GAIN);
        assert_eq!(Loudness::new(0.0).gain(), Loudness::MIN_GAIN);
    }

    fn frame() -> Vec<f32> {
        vec![0.1; VAD_FRAME_SAMPLES]
    }

    fn config() -> SegmenterConfig {
        SegmenterConfig {
            enter_threshold: 0.5,
            exit_threshold: 0.35,
            onset_frames: 2,
            chunk_pause_frames: 3,
            endpoint: EndpointPolicy::fixed(6),
            resume_window_frames: 4,
            // Deliberately larger than `onset_frames`. If the two are equal, the onset frames
            // fill the pre-roll buffer entirely and no genuine pre-speech padding survives -
            // the setting looks enabled while doing nothing.
            preroll_frames: 4,
            postroll_frames: 1,
            minimum_frames: 3,
            // Out of reach unless a test brings it in, so a long script is not cut at a breath.
            rolling_frames: 1_000,
            maximum_frames: 50,
            overlap_frames: 2,
            max_turn_frames: 900,
        }
    }

    /// Drives the segmenter with a probability sequence, collecting whatever it emits.
    fn run(segmenter: &mut Segmenter, probabilities: &[f32]) -> Vec<SpeechSegment> {
        probabilities
            .iter()
            .filter_map(|probability| segmenter.push_turn(frame(), *probability).chunk)
            .collect()
    }

    #[test]
    fn silence_alone_never_opens_a_segment() {
        let mut segmenter = Segmenter::new(config());
        assert!(run(&mut segmenter, &[0.0; 40]).is_empty());
        assert!(!segmenter.is_open());
    }

    #[test]
    fn a_single_loud_frame_is_rejected_as_a_click() {
        // A door knock or key press spikes for one frame. Requiring consecutive onset frames is
        // the whole reason this does not open an utterance.
        let mut segmenter = Segmenter::new(config());
        assert!(run(&mut segmenter, &[0.0, 0.0, 0.9, 0.0, 0.0, 0.0]).is_empty());
        assert!(!segmenter.is_open());
    }

    #[test]
    fn a_short_pause_emits_a_piece_but_keeps_the_turn_open() {
        // The heart of the two-level design: a pause between phrases hands audio to the
        // recogniser immediately while the speaker is still mid-thought.
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 10]);
        script.extend([0.0; 3]);
        let segments = run(&mut segmenter, &script);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end, SegmentEnd::Pause);
        assert!(segmenter.is_open(), "a short pause must not end the turn");
    }

    #[test]
    fn a_long_pause_ends_the_turn() {
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 10]);
        let mut ended = false;
        for probability in script {
            ended |= segmenter.push_turn(frame(), probability).turn_ended;
        }
        assert!(!ended, "speech alone must not end the turn");
        for _ in 0..8 {
            ended |= segmenter.push_turn(frame(), 0.0).turn_ended;
        }
        assert!(ended, "sustained silence must end the turn");
        assert!(!segmenter.is_open());
    }

    #[test]
    fn the_speaker_resuming_after_a_short_pause_stays_in_one_turn() {
        // Someone gathering their thoughts mid-sentence must not have their turn ended under
        // them; that is the failure a single threshold cannot avoid.
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 8]);
        script.extend([0.0; 4]); // past the chunk pause, short of the endpoint
        script.extend([0.9; 8]);
        let mut ended = false;
        for probability in script {
            ended |= segmenter.push_turn(frame(), probability).turn_ended;
        }
        assert!(!ended, "a mid-thought pause must not end the turn");
        assert!(segmenter.is_open());
    }

    #[test]
    fn the_endpoint_pause_is_long_enough_to_hide_the_final_transcription() {
        // The endpoint is not an arbitrary comfort setting. Recognising a final phrase costs
        // roughly 450 ms plus 240 ms per second of audio, and it runs while this timer runs - so
        // the window has to be at least that long or the model waits after the endpoint fires.
        // The adaptive ceiling is what has to be able to hide it; the learned value moves
        // between the floor and there.
        let ceiling_frames = SegmenterConfig::default().endpoint.ceiling();
        let endpoint_ms = ceiling_frames * VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize;
        let worst_final_chunk_ms = 450 + 240 * 2;
        assert!(
            endpoint_ms >= worst_final_chunk_ms,
            "endpoint window of {endpoint_ms} ms cannot hide a {worst_final_chunk_ms} ms transcription"
        );
    }

    #[test]
    fn the_chunk_pause_is_shorter_than_the_endpoint() {
        // If these ever crossed, a piece would only ever be emitted as the turn ended and the
        // streaming behaviour would silently disappear.
        let config = SegmenterConfig::default();
        let floor_frames = config.endpoint.floor();
        assert!(config.chunk_pause_frames < floor_frames);
    }

    #[test]
    fn the_segment_includes_audio_from_before_the_onset_was_confirmed() {
        // Silero reports speech late and confirmation costs more frames still, so a segment that
        // began at the moment of detection would be missing its first consonant.
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 6]);
        script.extend([0.0; 6]);
        let segments = run(&mut segmenter, &script);
        let frames = segments[0].samples.len() / VAD_FRAME_SAMPLES;
        // 2 silent pre-roll frames + 6 speech frames + 1 post-roll frame.
        assert_eq!(
            frames, 9,
            "the two silent frames before onset must be carried into the segment"
        );
    }

    #[test]
    fn a_brief_dip_mid_word_does_not_split_the_utterance() {
        // The gap between enter and exit thresholds exists for exactly this: a speaker's level
        // dipping to 0.4 between syllables must not end their sentence.
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9, 0.9]);
        script.extend([0.0; 6]);
        let segments = run(&mut segmenter, &script);
        assert_eq!(segments.len(), 1, "the dip must not have cut the utterance");
    }

    #[test]
    fn a_pause_shorter_than_the_hangover_keeps_one_utterance_open() {
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 5]);
        script.extend([0.0, 0.0]); // below hangover of 3
        script.extend([0.9; 5]);
        script.extend([0.0; 6]);
        assert_eq!(run(&mut segmenter, &script).len(), 1);
    }

    #[test]
    fn a_long_pause_produces_two_separate_utterances() {
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 6]);
        script.extend([0.0; 8]);
        script.extend([0.9; 6]);
        script.extend([0.0; 8]);
        assert_eq!(run(&mut segmenter, &script).len(), 2);
    }

    #[test]
    fn an_utterance_too_short_to_be_speech_is_discarded() {
        let mut config = config();
        config.minimum_frames = 50;
        let mut segmenter = Segmenter::new(config);
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 4]);
        script.extend([0.0; 6]);
        assert!(run(&mut segmenter, &script).is_empty());
    }

    #[test]
    fn a_monologue_is_cut_at_the_maximum_rather_than_growing_without_bound() {
        let mut config = config();
        config.maximum_frames = 10;
        let mut segmenter = Segmenter::new(config);
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 40]);
        let segments = run(&mut segmenter, &script);
        assert!(!segments.is_empty());
        assert_eq!(segments[0].end, SegmentEnd::MaximumLength);
        for segment in &segments {
            assert!(segment.samples.len() <= 10 * VAD_FRAME_SAMPLES);
        }
    }

    /// Pieces roll over at the next breath after 20 frames, and are cut hard at 30.
    fn rolling() -> SegmenterConfig {
        let mut config = config();
        config.rolling_frames = 20;
        config.maximum_frames = 30;
        config.chunk_pause_frames = 8;
        config.endpoint = EndpointPolicy::fixed(12);
        config
    }

    #[test]
    fn a_long_stretch_is_handed_over_at_the_next_breath_while_the_turn_goes_on() {
        let mut segmenter = Segmenter::new(rolling());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 24]);
        script.extend([0.1; BREATH_FRAMES]);
        script.extend([0.9; 10]);
        script.extend([0.0; 12]);
        let segments = run(&mut segmenter, &script);
        assert_eq!(
            segments.len(),
            2,
            "one piece at the breath, the rest after it"
        );
        // pre-roll 2 + speech 24 + the breath trimmed to the post-roll of 1.
        assert_eq!(segments[0].samples.len(), 27 * VAD_FRAME_SAMPLES);
        assert_eq!(segments[0].end, SegmentEnd::Pause);
        // The breath again as pre-roll, the speech after it, and the post-roll.
        assert_eq!(
            segments[1].probabilities[..BREATH_FRAMES],
            [0.1; BREATH_FRAMES]
        );
        assert_eq!(
            segments[1].samples.len(),
            (BREATH_FRAMES + 10 + 1) * VAD_FRAME_SAMPLES
        );
        // Handed over at its own pause, as any last phrase is, before the endpoint.
        assert_eq!(segments[1].end, SegmentEnd::Pause);
        assert!(
            segments.iter().all(|s| !s.overlaps_previous),
            "silence repeats no words"
        );
    }

    #[test]
    fn a_breath_before_the_rolling_length_is_just_a_breath() {
        let mut segmenter = Segmenter::new(rolling());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 10]);
        script.extend([0.1; BREATH_FRAMES]);
        script.extend([0.9; 4]);
        script.extend([0.0; 12]);
        assert_eq!(run(&mut segmenter, &script).len(), 1);
    }

    #[test]
    fn with_no_breath_the_cut_lands_where_the_voice_was_quietest_and_the_seam_overlaps() {
        let mut segmenter = Segmenter::new(rolling());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 40]);
        script[25] = 0.6; // softer, but still speech: not a breath
        script.extend([0.0; 12]);
        let segments = run(&mut segmenter, &script);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].end, SegmentEnd::MaximumLength);
        assert_eq!(
            segments[0].samples.len(),
            25 * VAD_FRAME_SAMPLES,
            "cut at the soft frame"
        );
        assert!(!segments[0].overlaps_previous);
        // The next piece opens with the overlap - the two frames before the cut - then the
        // soft frame the cut landed on.
        assert!(segments[1].overlaps_previous);
        assert_eq!(segments[1].probabilities[..3], [0.9, 0.9, 0.6]);
    }

    #[test]
    fn flush_closes_an_open_utterance_and_labels_it_as_a_stream_end() {
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 8]);
        assert!(run(&mut segmenter, &script).is_empty());
        assert!(segmenter.is_open());
        let flushed = segmenter
            .flush()
            .expect("an open utterance must be emitted");
        assert_eq!(flushed.end, SegmentEnd::StreamEnd);
        assert!(segmenter.flush().is_none(), "flush must be idempotent");
    }

    #[test]
    fn trailing_silence_is_trimmed_but_the_postroll_survives() {
        // Handing the transcriber half a second of silence wastes time; clipping the final
        // consonant costs accuracy. The post-roll is the compromise.
        let mut segmenter = Segmenter::new(config());
        let mut script = vec![0.0, 0.0];
        script.extend([0.9; 10]);
        script.extend([0.0; 6]);
        let segment = &run(&mut segmenter, &script)[0];
        let frames = segment.samples.len() / VAD_FRAME_SAMPLES;
        // preroll(2 silent) + speech(10) + postroll(1) = 13. The segmenter saw 4 trailing silent
        // frames before closing and kept only one of them.
        assert_eq!(
            frames, 13,
            "expected trailing silence trimmed down to the post-roll"
        );
    }

    #[test]
    fn duration_is_reported_in_milliseconds_of_audio() {
        let segment = SpeechSegment {
            samples: vec![0.0; SAMPLE_RATE as usize],
            probabilities: Vec::new(),
            end: SegmentEnd::Silence,
            overlaps_previous: false,
        };
        assert_eq!(segment.duration_ms(), 1_000);
    }

    #[test]
    fn frames_for_ms_never_rounds_down_to_nothing() {
        // A caller asking for 10 ms of hangover must not silently get zero frames, which would
        // close every utterance on its first quiet frame.
        assert_eq!(SegmenterConfig::frames_for_ms(0), 1);
        assert_eq!(SegmenterConfig::frames_for_ms(10), 1);
        // 32 ms a frame, rounded down, with one as the floor.
        assert_eq!(SegmenterConfig::frames_for_ms(320), 10);
        assert_eq!(SegmenterConfig::frames_for_ms(351), 10);
        assert_eq!(SegmenterConfig::frames_for_ms(352), 11);
    }

    #[test]
    fn a_wrong_sized_frame_is_rejected_rather_than_reshaped() {
        // A frame is 512 fresh samples; the model sees 576 because it is given the 64 before
        // them as well. Quietly padding or truncating would produce a plausible-looking
        // probability from the wrong audio.
        let mut vad = SileroVad::embedded().expect("embedded silero");
        let error = vad.probability(&[0.0; 100]).unwrap_err();
        assert!(matches!(error, AudioError::FrameSize { expected: 512, .. }));
        // The width the graph takes is not the width a caller passes.
        assert!(vad.probability(&[0.0; 576]).is_err());
    }

    #[test]
    fn stopping_mid_frame_does_not_lose_the_end_of_the_word() {
        // The detector only reads whole frames, so whatever is left over when capture stops had
        // never been seen and was dropped - up to 32 ms, which is where a final consonant lives.
        let mut pipeline = CapturePipeline::new(SegmenterConfig {
            endpoint: EndpointPolicy::fixed(3),
            onset_frames: 1,
            minimum_frames: 1,
            ..SegmenterConfig::default()
        })
        .expect("pipeline");
        // Opened through the segmenter directly: what the real detector makes of a synthetic
        // tone is not the subject here, and driving it through the model would make this a test
        // of Silero rather than of what happens to the samples left over.
        for _ in 0..6 {
            pipeline
                .segmenter
                .push_turn(vec![0.3; VAD_FRAME_SAMPLES], 0.9);
        }
        assert!(pipeline.segmenter.is_open(), "the turn has to be open");
        let tail = vec![0.42f32; VAD_FRAME_SAMPLES - 1];
        pipeline.push_events(&tail).expect("push tail");
        assert_eq!(
            pipeline.pending.len(),
            tail.len(),
            "a part-frame tail waits in the buffer, unseen by the detector",
        );

        let segment = pipeline.flush().expect("an open turn is flushed");
        let kept = segment
            .samples
            .iter()
            .rev()
            .take_while(|sample| (**sample - 0.42).abs() < 1e-6)
            .count();
        assert_eq!(
            kept,
            tail.len(),
            "the part-frame tail has to reach the recogniser",
        );
    }

    #[test]
    fn a_brief_noise_cannot_pad_itself_up_to_an_utterance() {
        // Reproduced before the fix: two loud frames became a segment of fourteen once the
        // pre-roll and post-roll were added, and the minimum-length rule measured the padded
        // total. A cough cleared a bar meant to require a third of a second of speech.
        let mut config = SegmenterConfig {
            preroll_frames: 6,
            postroll_frames: 6,
            minimum_frames: 8,
            onset_frames: 1,
            ..SegmenterConfig::default()
        };
        config.endpoint = EndpointPolicy::fixed(3);
        let mut segmenter = Segmenter::new(config);
        let frame = || vec![0.1f32; VAD_FRAME_SAMPLES];

        let mut segments = Vec::new();
        for probability in [0.01; 10] {
            segments.extend(segmenter.push_turn(frame(), probability).chunk);
        }
        // A short burst of speech: two frames, well under the minimum.
        for probability in [0.9, 0.9] {
            segments.extend(segmenter.push_turn(frame(), probability).chunk);
        }
        for probability in [0.01; 20] {
            segments.extend(segmenter.push_turn(frame(), probability).chunk);
        }
        assert!(
            segments.is_empty(),
            "padding turned {} voiced frames into an accepted utterance",
            2,
        );
    }

    #[test]
    fn the_model_is_given_the_samples_before_the_frame_as_well() {
        // Silero is published with a 512-sample window and reads the 64 samples before it as
        // context. Feeding it 576 fresh samples instead - which is what the graph width invites -
        // judges every window from a standing start and steps 36 ms at a time through audio
        // that is being measured in 32 ms frames.
        let mut vad = SileroVad::embedded().expect("embedded silero");
        let tone: Vec<f32> = (0..VAD_FRAME_SAMPLES)
            .map(|i| (i as f32 * 0.31).sin() * 0.5)
            .collect();
        let quiet = vec![0.0f32; VAD_FRAME_SAMPLES];

        // The same frame, once with loud audio behind it and once from silence. The context is
        // the only difference, so the readings may not be identical.
        vad.probability(&tone).unwrap();
        let after_tone = vad.probability(&quiet).unwrap();
        vad.reset();
        let from_silence = vad.probability(&quiet).unwrap();
        assert_ne!(
            after_tone, from_silence,
            "the frame before this one has to reach the model",
        );
    }

    #[test]
    fn silero_separates_speech_from_silence() {
        // An end-to-end sanity check on the real model: digital silence must score low. The
        // model is compiled in, so this runs everywhere, CI included.
        let mut vad = SileroVad::embedded().expect("embedded silero");
        let silence = vec![0.0f32; VAD_FRAME_SAMPLES];
        let probability = vad.probability(&silence).expect("probability");
        assert!(
            probability < 0.2,
            "digital silence scored {probability}, which would open segments on nothing"
        );
    }

    // --- decimation --------------------------------------------------------------------------

    // --- filtering ---------------------------------------------------------------------------

    // --- rate conversion ---------------------------------------------------------------------

    // --- double-talk guard -------------------------------------------------------------------

    #[test]
    fn a_fixed_endpoint_holds_exactly_its_threshold() {
        let mut segmenter = Segmenter::new(SegmenterConfig {
            endpoint: EndpointPolicy::fixed(19),
            ..config()
        });
        for _ in 0..12 {
            segmenter.push_turn(frame(), 0.9);
        }
        assert!(segmenter.is_open());
        let mut ended = false;
        for _ in 0..18 {
            ended |= segmenter.push_turn(frame(), 0.05).turn_ended;
        }
        assert!(!ended, "18 silent frames is under the 19-frame endpoint");
        assert!(
            segmenter.push_turn(frame(), 0.05).turn_ended,
            "the 19th ends it"
        );
    }

    /// Feeds `frames` of confident speech.
    fn speak(segmenter: &mut Segmenter, frames: usize) {
        for _ in 0..frames {
            segmenter.push_turn(frame(), 0.9);
        }
    }

    /// Feeds silence until the turn ends, returning how many frames that took.
    fn silence_until_end(segmenter: &mut Segmenter) -> usize {
        for elapsed in 1..500 {
            if segmenter.push_turn(frame(), 0.05).turn_ended {
                return elapsed;
            }
        }
        panic!("the turn never ended");
    }

    /// Feeds `frames` of silence with no turn open.
    fn silence(segmenter: &mut Segmenter, frames: usize) {
        for _ in 0..frames {
            assert!(!segmenter.push_turn(frame(), 0.05).turn_ended);
        }
    }

    fn adaptive() -> Segmenter {
        Segmenter::new(SegmenterConfig {
            endpoint: EndpointPolicy::adaptive(),
            ..config()
        })
    }

    #[test]
    fn carrying_on_after_being_cut_off_lengthens_the_endpoint() {
        // The complaint this exists for. Someone thinking mid-sentence is cut off, carries on a
        // moment later, and has no idea there is a number to go and change. A pause longer than
        // the endpoint cannot be observed from inside a turn - it ends the turn by definition -
        // so the resumption is the only evidence available that the endpoint was too short.
        let mut segmenter = adaptive();
        let before = segmenter.config.endpoint.frames();
        speak(&mut segmenter, 8);
        silence_until_end(&mut segmenter);
        silence(&mut segmenter, 3);
        speak(&mut segmenter, 8); // "...what I meant was"
        let after = segmenter.config.endpoint.frames();
        assert!(
            after > before,
            "being cut off and carrying on must lengthen the endpoint, {before} -> {after}"
        );
        // And it must still clear that pause a turn later: one uneventful turn relaxes by a
        // single frame, which must not undo what the interruption taught it.
        silence_until_end(&mut segmenter);
        assert!(segmenter.config.endpoint.frames() > before);
    }

    #[test]
    fn a_considered_follow_up_is_not_mistaken_for_an_interruption() {
        // Answering after a real gap - or after Zen has spoken - says the last ending was
        // right. Treating that as a premature cut would make the endpoint creep up every turn
        // until every reply felt sluggish.
        let mut segmenter = adaptive();
        let before = segmenter.config.endpoint.frames();
        speak(&mut segmenter, 8);
        silence_until_end(&mut segmenter);
        let gap = segmenter.config.resume_window_frames + 5;
        silence(&mut segmenter, gap);
        speak(&mut segmenter, 8);
        assert!(
            segmenter.config.endpoint.frames() <= before,
            "a follow-up after a real gap must never lengthen the endpoint"
        );
    }

    #[test]
    fn a_learned_endpoint_relaxes_once_the_speaker_stops_needing_it() {
        let mut segmenter = adaptive();
        speak(&mut segmenter, 8);
        silence_until_end(&mut segmenter);
        silence(&mut segmenter, 3);
        speak(&mut segmenter, 8);
        let raised = segmenter.config.endpoint.frames();
        silence_until_end(&mut segmenter);

        let gap = segmenter.config.resume_window_frames + 5;
        for _ in 0..12 {
            silence(&mut segmenter, gap);
            speak(&mut segmenter, 8);
            silence_until_end(&mut segmenter);
        }
        let settled = segmenter.config.endpoint.frames();
        assert!(
            settled < raised,
            "uneventful turns must recover latency, {raised} -> {settled}"
        );

        // But never below the floor: a brisk stretch must not make it trigger-happy.
        for _ in 0..400 {
            silence(&mut segmenter, gap);
            speak(&mut segmenter, 8);
            silence_until_end(&mut segmenter);
        }
        let floor_frames = EndpointPolicy::adaptive().floor();
        assert_eq!(segmenter.config.endpoint.frames(), floor_frames);
    }

    #[test]
    fn the_turn_cap_is_longer_than_anyone_actually_speaks_for() {
        // The cap exists for a microphone left open, not for a long question. Set it near the
        // length of real speech and a genuine question gets split in two, with the speaker's
        // own second half interrupting the answer to the first.
        let config = SegmenterConfig::default();
        let cap_ms = config.max_turn_frames * 36;
        assert!(
            cap_ms >= 55_000,
            "a turn is cut off after {cap_ms} ms, which a real question can reach"
        );
    }

    #[test]
    fn a_turn_that_never_pauses_still_ends() {
        // A television, a speakerphone, a microphone left open: speech that never stops is not
        // a person finishing a thought. Waiting for a pause that is not coming leaves every
        // word already said unanswered, which is the worst of both.
        let mut segmenter = Segmenter::new(config());
        // Speech with no gap at all runs to the hard stop, at twice the cap.
        let cap = segmenter.config.max_turn_frames * 2;
        let mut ended_at = None;
        for frames in 1..=cap + 10 {
            if segmenter.push_turn(frame(), 0.9).turn_ended {
                ended_at = Some(frames);
                break;
            }
        }
        let ended_at = ended_at.expect("a turn with no pause in it must still end");
        // The turn opens a frame or two after speech starts, so the cap lands just past it.
        assert!(
            (cap..=cap + 5).contains(&ended_at),
            "ended after {ended_at} frames, cap is {cap}"
        );
        // And not so early that a long but ordinary answer is cut in half.
        assert!(
            ended_at * 36 > 25_000,
            "a turn must survive at least 25 seconds"
        );
    }

    #[test]
    fn a_pause_inside_a_turn_keeps_the_endpoint_from_falling_below_it() {
        // This one the endpoint already tolerated, so it is not evidence of being too short -
        // only that relaxing past it would start cutting the speaker off.
        let mut segmenter = adaptive();
        let before = segmenter.config.endpoint.frames();
        speak(&mut segmenter, 8);
        for _ in 0..before - 2 {
            assert!(!segmenter.push_turn(frame(), 0.05).turn_ended);
        }
        speak(&mut segmenter, 8);
        silence_until_end(&mut segmenter);
        assert!(
            segmenter.config.endpoint.frames() >= before,
            "a tolerated pause must not let the endpoint relax under it"
        );
    }

    #[test]
    fn a_pause_learned_last_session_is_where_the_next_one_starts() {
        let seeded = EndpointPolicy::adaptive_from_ms(1_200);
        assert_eq!(seeded.frames(), SegmenterConfig::frames_for_ms(1_200));
        // A remembered value can never carry the policy outside its own bounds.
        assert_eq!(EndpointPolicy::adaptive_from_ms(1).frames(), seeded.floor());
        assert_eq!(
            EndpointPolicy::adaptive_from_ms(9_999).frames(),
            seeded.ceiling()
        );
    }

    #[test]
    fn one_long_pause_stops_costing_latency_within_a_few_clean_turns() {
        // Measured: the wait learned 1.47 s from someone reading aloud. Falling a frame per turn,
        // it took about eighteen clean turns to get back under a second.
        let mut policy = EndpointPolicy::adaptive_from_ms(1_470);
        let mut turns = 0;
        while policy.frames() > SegmenterConfig::frames_for_ms(1_000) {
            policy.relax();
            turns += 1;
            assert!(turns < 100, "never came back down");
        }
        assert!(
            turns <= 8,
            "took {turns} clean turns to come back under a second"
        );
        // And it still settles all the way to the floor, not somewhere above it.
        for _ in 0..200 {
            policy.relax();
        }
        assert_eq!(policy.frames(), policy.floor());
    }

    #[test]
    fn a_learned_endpoint_is_bounded_however_long_the_speaker_hesitates() {
        let mut policy = EndpointPolicy::adaptive();
        let ceiling_frames = policy.ceiling();
        for _ in 0..20 {
            policy.observe(10_000);
        }
        assert_eq!(policy.frames(), ceiling_frames);
    }

    #[test]
    fn ordinary_speech_opens_an_utterance() {
        let mut segmenter = Segmenter::new(config());
        let mut opened = false;
        for _ in 0..10 {
            segmenter.push_turn(frame(), 0.6);
            opened |= segmenter.is_open();
        }
        assert!(opened, "0.6 is ordinary speech");
    }

    // --- semantic chunking -------------------------------------------------------------------

    fn ms_to_samples(ms: usize) -> usize {
        ms * SAMPLE_RATE as usize / 1000
    }

    /// Frame probabilities for `total_ms` of speech, with pauses at the given millisecond marks.
    fn speech_with_pauses(total_ms: usize, pauses_ms: &[usize]) -> Vec<f32> {
        let frame_ms = VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize;
        let frames = total_ms / frame_ms;
        let mut probabilities = vec![0.9f32; frames];
        for pause in pauses_ms {
            let index = pause / frame_ms;
            // A pause is a few frames wide, as a real breath is.
            for offset in 0..4 {
                if let Some(slot) = probabilities.get_mut(index + offset) {
                    *slot = 0.05;
                }
            }
        }
        probabilities
    }

    #[test]
    fn a_short_utterance_is_left_whole() {
        // Splitting a five-second utterance costs a join and buys nothing.
        let samples = ms_to_samples(5_000);
        let chunks = plan_chunks(
            samples,
            &speech_with_pauses(5_000, &[2_000]),
            &ChunkConfig::default(),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, samples);
    }

    #[test]
    fn an_empty_utterance_produces_nothing() {
        assert!(plan_chunks(0, &[], &ChunkConfig::default()).is_empty());
    }

    #[test]
    fn a_long_utterance_is_split_into_several_pieces() {
        let samples = ms_to_samples(30_000);
        let chunks = plan_chunks(
            samples,
            &speech_with_pauses(30_000, &[6_000, 12_000, 18_000, 24_000]),
            &ChunkConfig::default(),
        );
        assert!(
            chunks.len() >= 3,
            "thirty seconds should split into at least three pieces, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, samples);
    }

    #[test]
    fn cuts_land_on_pauses_rather_than_on_a_timer() {
        // The whole reason to look at probabilities. An arbitrary cut lands mid-word about as
        // often as not, and then both halves transcribe badly.
        let samples = ms_to_samples(20_000);
        let chunks = plan_chunks(
            samples,
            &speech_with_pauses(20_000, &[5_800, 11_600]),
            &ChunkConfig::default(),
        );
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(
                chunk.split_on_pause,
                "chunk ending at {} did not cut on a pause",
                chunk.end
            );
        }
    }

    #[test]
    fn a_speaker_who_never_pauses_is_still_cut_at_the_limit() {
        // Someone reading aloud may not draw breath for a long time. Refusing to cut would hand
        // the transcriber an unbounded buffer and stall the turn.
        let samples = ms_to_samples(30_000);
        let unbroken = vec![0.95f32; 30_000 / 36];
        let chunks = plan_chunks(samples, &unbroken, &ChunkConfig::default());
        assert!(
            chunks.len() > 1,
            "an unbroken monologue must still be split"
        );
        assert!(
            chunks.iter().any(|chunk| !chunk.split_on_pause),
            "a forced cut must be marked so the seam is known to be untrustworthy"
        );
    }

    #[test]
    fn no_chunk_exceeds_the_hard_maximum() {
        let config = ChunkConfig::default();
        let samples = ms_to_samples(45_000);
        let chunks = plan_chunks(samples, &vec![0.95f32; 45_000 / 36], &config);
        for chunk in &chunks {
            assert!(
                chunk.duration_ms() <= config.maximum_ms + config.overlap_ms,
                "chunk of {} ms exceeds the {} ms limit",
                chunk.duration_ms(),
                config.maximum_ms
            );
        }
    }

    #[test]
    fn chunks_overlap_so_a_word_on_the_seam_survives() {
        // A transcriber uses surrounding audio for context; a word sitting exactly on a boundary
        // would be half-heard on each side and recognised on neither.
        let samples = ms_to_samples(30_000);
        let chunks = plan_chunks(
            samples,
            &speech_with_pauses(30_000, &[6_000, 12_000, 18_000, 24_000]),
            &ChunkConfig::default(),
        );
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start < pair[0].end,
                "chunks {:?} and {:?} do not overlap",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn the_pieces_cover_the_whole_utterance_with_no_gap() {
        let samples = ms_to_samples(28_000);
        let chunks = plan_chunks(
            samples,
            &speech_with_pauses(28_000, &[7_000, 14_000, 21_000]),
            &ChunkConfig::default(),
        );
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, samples);
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start <= pair[0].end,
                "a gap between {:?} and {:?} would silently drop audio",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn planning_always_terminates_even_with_no_probability_data() {
        // A caller that lost the probabilities must still get a usable plan rather than an
        // infinite loop.
        let samples = ms_to_samples(40_000);
        let chunks = plan_chunks(samples, &[], &ChunkConfig::default());
        assert!(!chunks.is_empty());
        assert_eq!(chunks.last().unwrap().end, samples);
    }

    // --- interpolation -----------------------------------------------------------------------
}
