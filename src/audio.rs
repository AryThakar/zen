// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Capture front-end: 16 kHz microphone samples to speech segments.
//!
//! The chain, and why it is in this order:
//!
//! ```text
//!   webview capture ──► silero ─────────► segmenter
//!   16 kHz mono          speech            utterances, chunk pauses
//!   320-sample frames    probability       and turn endpoints
//!                        per 576 samples
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

/// Samples per Silero inference. 576 at 16 kHz is 36 ms, and the exported graph fixes it.
pub const VAD_FRAME_SAMPLES: usize = 576;

const VAD_STATE_LEN: usize = 128;

// ============================================================================================
// Silero VAD
// ============================================================================================

/// Silero VAD, holding its own recurrent state.
///
/// The state is per-conversation, not per-frame: carrying it across utterances is what lets the
/// model use context, but stale state from a previous speaker causes false starts on the next
/// one, so [`SileroVad::reset`] exists and the segmenter calls it at every segment boundary.
///
/// Silero's published export carries both LSTM halves in one `2 x 1 x 128` tensor and returns
/// the next one as `stateN`; older exports split it into separate `h` and `c` tensors. This
/// drives the published one.
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
}

impl SileroVad {
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
        })
    }

    /// Clears recurrent state.
    ///
    /// Without this between utterances the GRU carries dirty state forward and turn two starts
    /// with the model already half-convinced someone is speaking.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Speech probability for exactly one frame.
    pub fn probability(&mut self, frame: &[f32]) -> Result<f32, AudioError> {
        if frame.len() != VAD_FRAME_SAMPLES {
            return Err(AudioError::FrameSize {
                expected: VAD_FRAME_SAMPLES,
                actual: frame.len(),
            });
        }
        let input = Value::from_array(([1usize, VAD_FRAME_SAMPLES], frame.to_vec()))
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
/// [`EndpointPolicy::Adaptive`] measures the speaker instead, from two kinds of evidence:
///
/// - **A pause inside a turn** - silence, then more words before the endpoint fired. The
///   endpoint already tolerated it, so this only says it must not fall below that.
/// - **A turn that ended and was resumed straight away.** This is the important one, and the
///   only signal that says the endpoint was *too short*: a pause longer than the endpoint ends
///   the turn by definition, so it can never be observed from inside one. What can be observed
///   is the speaker carrying on a moment later - which means the silence was a breath, not an
///   ending, and their real pause was the endpoint plus however long they took to resume.
///
/// It relaxes by a frame per uneventful turn so one long hesitation does not slow the rest of
/// the conversation for good, and is clamped at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointPolicy {
    /// Exactly this many frames of silence, whatever the speaker does.
    Fixed(usize),
    /// Learned from the speaker's own within-turn pauses.
    Adaptive {
        /// Never wait less than this, however brisk the speaker.
        floor_frames: usize,
        /// Never wait more than this, however hesitant.
        ceiling_frames: usize,
        /// Held above the longest within-turn pause seen, so a pause of the same length again
        /// does not end the turn.
        margin_frames: usize,
        /// Current estimate. Public so a caller can seed it and read back what was learned.
        current_frames: usize,
    },
}

impl EndpointPolicy {
    /// A neutral starting point: long enough for an ordinary mid-sentence beat, short enough
    /// that a decisive speaker is not left waiting.
    pub fn adaptive() -> Self {
        Self::Adaptive {
            floor_frames: SegmenterConfig::frames_for_ms(600),
            ceiling_frames: SegmenterConfig::frames_for_ms(2_000),
            margin_frames: SegmenterConfig::frames_for_ms(150),
            current_frames: SegmenterConfig::frames_for_ms(900),
        }
    }

    /// Frames of silence that end a turn right now.
    pub fn frames(self) -> usize {
        match self {
            Self::Fixed(frames) => frames,
            Self::Adaptive { current_frames, .. } => current_frames,
        }
    }

    /// Raise the endpoint clear of a pause the speaker has demonstrated they take.
    ///
    /// Only ever raises. Being cut off mid-sentence costs the speaker the whole utterance and
    /// their place in the thought; waiting a fraction of a second longer costs a fraction of a
    /// second, so the two errors are not worth trading symmetrically.
    fn observe(&mut self, pause: usize) {
        let Self::Adaptive {
            floor_frames,
            ceiling_frames,
            margin_frames,
            current_frames,
        } = self
        else {
            return;
        };
        let wanted = (pause + *margin_frames).clamp(*floor_frames, *ceiling_frames);
        *current_frames = (*current_frames).max(wanted);
    }

    /// One turn that ended and stayed ended: evidence the endpoint is not too short.
    ///
    /// Falls by a single frame, so a stretch of decisive answers slowly recovers the latency a
    /// hesitant one cost, without one brisk reply undoing what was learned.
    fn relax(&mut self) {
        if let Self::Adaptive {
            floor_frames,
            current_frames,
            ..
        } = self
        {
            *current_frames = current_frames.saturating_sub(1).max(*floor_frames);
        }
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
    /// speaker gathering their thoughts is not cut off, and - deliberately - long enough to hide
    /// the final chunk's transcription inside it. Recognising the last phrase costs roughly
    /// 450 ms plus 240 ms per second of audio; while that runs, this timer is running too, so
    /// the work is finished before the endpoint fires and the model starts with nothing to wait
    /// for. [`EndpointPolicy::Adaptive`] keeps it there without anyone having to choose a number.
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
    /// Hard cut, so a monologue is handed to the transcriber in pieces rather than all at once.
    pub maximum_frames: usize,
}

impl SegmenterConfig {
    pub const fn frames_for_ms(ms: usize) -> usize {
        // 576 samples at 16 kHz is 36 ms.
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
        Self {
            enter_threshold: 0.5,
            exit_threshold: 0.35,
            onset_frames: 2,                              // ~72 ms
            chunk_pause_frames: Self::frames_for_ms(400), // ~400 ms
            endpoint: EndpointPolicy::adaptive(),
            preroll_frames: Self::frames_for_ms(300), // ~300 ms
            postroll_frames: Self::frames_for_ms(200), // ~200 ms
            resume_window_frames: Self::frames_for_ms(1_000), // ~1 s
            minimum_frames: Self::frames_for_ms(300), // ~300 ms
            maximum_frames: Self::frames_for_ms(20_000), // 20 s
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
    pending_probabilities: VecDeque<f32>,
    speech_run: usize,
    silence_run: usize,
    /// Longest silence in the current turn that was followed by more speech. The endpoint
    /// already tolerated it, so it only says the endpoint must not fall below it.
    longest_pause: usize,
    /// Frames of silence since the last turn ended, while none is open. A speaker who resumes
    /// after only a few of these was not finished, and the endpoint that cut them was short.
    since_turn_end: Option<usize>,
    open: bool,
}

impl Segmenter {
    pub fn new(config: SegmenterConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            active: Vec::new(),
            active_probabilities: Vec::new(),
            pending_probabilities: VecDeque::new(),
            speech_run: 0,
            silence_run: 0,
            longest_pause: 0,
            since_turn_end: None,
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
            self.open = true;
            self.silence_run = 0;
        }
        None
    }

    fn push_open(&mut self, frame: Vec<f32>, probability: f32) -> SegmentOutcome {
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

        // The turn is over. Everything still buffered goes out with it.
        if self.silence_run >= self.config.endpoint.frames() {
            let chunk = self.close(SegmentEnd::Silence);
            return SegmentOutcome {
                chunk,
                turn_ended: true,
            };
        }

        // Too long without a break to keep holding. The speaker has not finished, so the turn
        // stays open and this is emitted as a piece rather than an endpoint.
        if self.active.len() >= self.config.maximum_frames {
            let chunk = self.cut(SegmentEnd::MaximumLength);
            return SegmentOutcome {
                chunk,
                turn_ended: false,
            };
        }

        // A pause between phrases. Emit exactly once, on the frame the threshold is crossed -
        // `==` not `>=`, or every further silent frame would emit another empty piece.
        if self.silence_run == self.config.chunk_pause_frames
            && self.active.len() > self.config.minimum_frames
        {
            let chunk = self.cut(SegmentEnd::Pause);
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

    /// Takes the audio buffered so far while leaving the turn open.
    fn cut(&mut self, end: SegmentEnd) -> Option<SpeechSegment> {
        let mut frames = std::mem::take(&mut self.active);
        let mut probabilities = std::mem::take(&mut self.active_probabilities);
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
        if frames.len() < self.config.minimum_frames {
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
        self.open = false;
        self.speech_run = 0;
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

        if frames.len() < self.config.minimum_frames {
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
        })
    }
}

// ============================================================================================
// Semantic chunking
// ============================================================================================

/// How to cut a long utterance into pieces that can be transcribed at the same time.
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
    /// True when the cut landed on a real pause rather than the hard limit.
    ///
    /// Worth carrying: a chunk cut mid-word is far likelier to produce a broken transcript at its
    /// edge, so a consumer can join those differently from clean ones.
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
/// Long utterances are the ones that hurt: a transcriber is roughly linear in audio length, so a
/// thirty-second monologue is a thirty-second wait before the assistant can even start thinking.
/// Cutting it into pieces lets them be transcribed at the same time.
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

/// Makeup gain and a peak limiter for synthesised speech.
///
/// The limiter is an envelope follower, not a waveshaper, and the difference is the whole
/// point. Bending each sample independently toward a ceiling is a distortion effect: it
/// changes the shape of the waveform, so the harmonics it adds are heard as roughness on
/// exactly the loudest, most sustained sounds - which in speech are the vowels. Following the
/// peak instead and applying one smooth gain to the passage leaves the waveform's shape
/// alone; only its level moves, which is what a listener does not notice.
///
/// Attack is immediate because there is no look-ahead here and a sample over the ceiling
/// cannot be let through. Release is slow, so the gain does not lurch back up between
/// syllables and pump.
pub struct Loudness {
    gain: f32,
    /// Gain reduction currently applied, where 1.0 is none.
    envelope: f32,
}

impl Loudness {
    /// Just below full scale, so the conversion to 16-bit is never the thing that clips.
    const CEILING: f32 = 0.98;
    /// Gain that brings measured synthesiser output to roughly -15 dBFS RMS.
    pub const DEFAULT_GAIN: f32 = 1.8;
    /// Range a caller may ask for. Below 0.5 the assistant is inaudible; above 4.0 the limiter
    /// is holding the level down through every syllable and the result is loud and flat.
    pub const MIN_GAIN: f32 = 0.5;
    pub const MAX_GAIN: f32 = 4.0;
    /// Per-sample recovery, about 150 ms at 24 kHz. Slow enough not to pump between words,
    /// fast enough that a single loud syllable does not hold the whole reply down.
    const RELEASE: f32 = 1.0 / (0.150 * TTS_SAMPLE_RATE as f32);

    pub fn new(gain: f32) -> Self {
        Self {
            gain: gain.clamp(Self::MIN_GAIN, Self::MAX_GAIN),
            envelope: 1.0,
        }
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Applies gain in place, holding peaks under the ceiling by moving the level rather than
    /// by reshaping the wave.
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let amplified = *sample * self.gain;
            let magnitude = amplified.abs();
            let needed = if magnitude > Self::CEILING {
                Self::CEILING / magnitude
            } else {
                1.0
            };
            self.envelope = if needed < self.envelope {
                // A sample above the ceiling cannot wait for a smoothed attack.
                needed
            } else {
                (self.envelope + (needed - self.envelope) * Self::RELEASE).min(1.0)
            };
            // The envelope already holds this under the ceiling; the clamp is only a guard
            // against a non-finite sample reaching the conversion to 16-bit.
            *sample = (amplified * self.envelope).clamp(-1.0, 1.0);
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

/// The whole capture chain: 16 kHz microphone samples in, speech segments out.
///
/// Echo cancellation, noise suppression and gain control belong to the webview, which owns the
/// device and knows what it just played; this side owns only the decision of when someone is
/// speaking. Running the WebRTC chain again here would process the same audio twice, which
/// pumps noise and strips quiet consonants the recogniser needs.
pub enum CaptureEvent {
    Started,
    Segment(SpeechSegment),
    Ended,
}

pub struct CapturePipeline {
    vad: SileroVad,
    segmenter: Segmenter,
    /// Samples not yet forming a whole VAD frame.
    pending: Vec<f32>,
}

impl CapturePipeline {
    pub fn new(
        vad_model: impl AsRef<Path>,
        segmenter_config: SegmenterConfig,
    ) -> Result<Self, AudioError> {
        Ok(Self {
            vad: SileroVad::load(vad_model)?,
            segmenter: Segmenter::new(segmenter_config),
            pending: Vec::new(),
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
            let was_open = self.segmenter.is_open();
            let outcome = self.segmenter.push_turn(frame, probability);
            if !was_open && self.segmenter.is_open() {
                events.push(CaptureEvent::Started);
            }
            if let Some(segment) = outcome.chunk {
                events.push(CaptureEvent::Segment(segment));
            }
            if outcome.turn_ended {
                // Recurrent state carried into the next utterance makes the model start it
                // already half-convinced someone is speaking.
                self.vad.reset();
                events.push(CaptureEvent::Ended);
            }
        }
        self.pending.drain(..offset);
        Ok(events)
    }

    pub fn flush(&mut self) -> Option<SpeechSegment> {
        self.segmenter.flush()
    }

    /// Choose how the end of a turn is decided, without rebuilding the pipeline.
    ///
    /// Changing this must not cost a session restart, which would unload five gigabytes of
    /// model to change one number.
    pub fn set_endpoint(&mut self, endpoint: EndpointPolicy) {
        self.segmenter.config.endpoint = endpoint;
        self.reset_capture();
    }

    /// What the endpoint currently is, in milliseconds. Adaptive mode moves this as it learns,
    /// and the interface shows it so the behaviour is legible rather than mysterious.
    pub fn endpoint_ms(&self) -> usize {
        self.segmenter.config.endpoint.frames() * VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE as usize
    }

    pub fn reset_capture(&mut self) {
        // `config` carries the learned endpoint, so rebuilding from it keeps what the
        // speaker has already demonstrated about their own pauses.
        self.segmenter = Segmenter::new(self.segmenter.config);
        self.vad.reset();
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn quiet_speech_is_amplified_linearly() {
        // Anything under the ceiling is scaled exactly and nothing else is done to it, which
        // is what keeps ordinary speech uncoloured.
        let mut samples = [0.05, -0.1, 0.2, -0.3];
        Loudness::new(2.0).process(&mut samples);
        for (got, want) in samples.iter().zip([0.1, -0.2, 0.4, -0.6]) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn nothing_can_be_pushed_past_the_ceiling() {
        // The reason the limiter exists. Without it, enough gain to fix the average drives peaks
        // into the device clamp, and the loudest part of every sentence becomes distortion.
        for gain in [1.0, 1.8, 2.5, 4.0] {
            let mut samples: Vec<f32> = (-20..=20).map(|i| i as f32 / 10.0).collect();
            Loudness::new(gain).process(&mut samples);
            for sample in samples {
                assert!(
                    sample.abs() <= Loudness::CEILING,
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
        let mut limited = tone.clone();
        Loudness::default().process(&mut limited);
        assert!(
            limited.iter().all(|s| s.abs() <= Loudness::CEILING),
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
            let mut samples = sine(input, 220.0);
            Loudness::default().process(&mut samples);
            let energy: f64 = samples.iter().map(|s| *s as f64 * *s as f64).sum();
            (energy / samples.len() as f64).sqrt()
        };
        let (quiet, mid, loud) = (level(0.1), level(0.4), level(0.9));
        assert!(quiet < mid && mid < loud, "{quiet} {mid} {loud}");
    }

    #[test]
    fn the_level_recovers_after_a_peak_rather_than_staying_ducked() {
        // A limiter that never releases leaves everything after one loud syllable quiet.
        let mut samples = sine(0.95, 220.0);
        samples.extend(sine(0.2, 220.0));
        let quiet_alone = sine(0.2, 220.0);
        Loudness::default().process(&mut samples);
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
    fn the_waveform_keeps_its_shape_around_zero() {
        let mut samples = [0.0, -0.0, 0.25, -0.25];
        Loudness::new(1.8).process(&mut samples);
        assert_eq!(samples[0], 0.0);
        assert!((samples[2] + samples[3]).abs() < 1e-6, "asymmetric");
    }

    #[test]
    fn the_measured_synthesiser_level_lands_in_the_broadcast_range() {
        // Measured output: peak about 0.53, RMS about 0.09. The default gain brings that
        // average to roughly -15 dBFS with the peak still short of the ceiling, so the limiter
        // has nothing to do on ordinary speech and only catches genuine peaks.
        let mut peak = [0.53f32];
        Loudness::default().process(&mut peak);
        assert!(peak[0] < Loudness::CEILING, "peak clipped at {}", peak[0]);
        assert!(
            peak[0] > 0.8,
            "real speech peaks should use the range: {}",
            peak[0]
        );
        let mut rms = [0.09f32];
        Loudness::default().process(&mut rms);
        let db = 20.0 * rms[0].log10();
        assert!((-17.0..=-13.0).contains(&db), "average landed at {db} dBFS");
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
            endpoint: EndpointPolicy::Fixed(6),
            resume_window_frames: 4,
            // Deliberately larger than `onset_frames`. If the two are equal, the onset frames
            // fill the pre-roll buffer entirely and no genuine pre-speech padding survives -
            // the setting looks enabled while doing nothing.
            preroll_frames: 4,
            postroll_frames: 1,
            minimum_frames: 3,
            maximum_frames: 50,
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
        let EndpointPolicy::Adaptive { ceiling_frames, .. } = SegmenterConfig::default().endpoint
        else {
            panic!("the shipped default must be adaptive")
        };
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
        let EndpointPolicy::Adaptive { floor_frames, .. } = config.endpoint else {
            panic!("the shipped default must be adaptive")
        };
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
        };
        assert_eq!(segment.duration_ms(), 1_000);
    }

    #[test]
    fn frames_for_ms_never_rounds_down_to_nothing() {
        // A caller asking for 10 ms of hangover must not silently get zero frames, which would
        // close every utterance on its first quiet frame.
        assert_eq!(SegmenterConfig::frames_for_ms(0), 1);
        assert_eq!(SegmenterConfig::frames_for_ms(10), 1);
        assert_eq!(SegmenterConfig::frames_for_ms(360), 10);
    }

    #[test]
    fn a_wrong_sized_frame_is_rejected_rather_than_reshaped() {
        // The exported Silero graph fixes the frame at 576; quietly padding or truncating would
        // produce a plausible-looking probability from the wrong audio.
        let model = r"C:\zen-ai\model\VAD\silero_vad.onnx";
        if !Path::new(model).is_file() {
            return;
        }
        let mut vad = SileroVad::load(model).expect("load silero");
        let error = vad.probability(&[0.0; 100]).unwrap_err();
        assert!(matches!(error, AudioError::FrameSize { expected: 576, .. }));
    }

    #[test]
    fn silero_separates_speech_from_silence() {
        // An end-to-end sanity check on the real model: digital silence must score low. Skipped
        // when the model is absent so the suite still runs on a machine without it.
        let model = r"C:\zen-ai\model\VAD\silero_vad.onnx";
        if !Path::new(model).is_file() {
            return;
        }
        let mut vad = SileroVad::load(model).expect("load silero");
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
            endpoint: EndpointPolicy::Fixed(19),
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
        let EndpointPolicy::Adaptive { floor_frames, .. } = EndpointPolicy::adaptive() else {
            unreachable!()
        };
        assert_eq!(segmenter.config.endpoint.frames(), floor_frames);
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
    fn a_learned_endpoint_is_bounded_however_long_the_speaker_hesitates() {
        let mut policy = EndpointPolicy::adaptive();
        let EndpointPolicy::Adaptive { ceiling_frames, .. } = policy else {
            unreachable!()
        };
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
