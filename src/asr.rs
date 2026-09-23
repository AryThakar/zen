// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Speech to text: Qwen3-ASR through `crispasr.dll`, off the capture thread.
//!
//! The model is roughly 1.4 GB and runs on the CPU, which shapes the whole design:
//!
//! - **One session, not a pool.** A second session would cost another 1.4 GB of resident memory,
//!   and since transcription is CPU-bound on the same cores, two of them competing would finish
//!   two utterances no faster than one session finishes them in turn.
//! - **Its own thread.** Transcription takes hundreds of milliseconds; running it on the capture
//!   path would drop microphone frames for the whole duration.
//! - **A bounded queue.** If transcription falls behind, the backlog is visible and capped rather
//!   than growing until the machine swaps.
//!
//! Long utterances are still split (see `plan_chunks`), but the win is *streaming*, not
//! parallelism: the first piece of a thirty-second monologue comes back in about a second instead
//! of the caller waiting thirty for the whole thing.

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_float, c_int, c_void},
    path::{Path, PathBuf},
    sync::{
        atomic::Ordering,
        mpsc::{sync_channel, Receiver, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use crate::audio::{
    plan_chunks, ChunkConfig, SpeechSegment, SAMPLE_RATE, SPEECH_PROBABILITY, VAD_FRAME_SAMPLES,
};

#[cfg(target_os = "windows")]
extern "system" {
    fn SetDllDirectoryW(path: *const u16) -> i32;
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
}

pub enum CrispSession {}
pub enum CrispResult {}

type FnOpen = unsafe extern "C" fn(*const c_char, c_int) -> *mut CrispSession;
type FnTranscribe =
    unsafe extern "C" fn(*mut CrispSession, *const c_float, c_int) -> *mut CrispResult;
type FnSegments = unsafe extern "C" fn(*mut CrispResult) -> c_int;
type FnSegmentText = unsafe extern "C" fn(*mut CrispResult, c_int) -> *const c_char;
type FnResultFree = unsafe extern "C" fn(*mut CrispResult);
type FnClose = unsafe extern "C" fn(*mut CrispSession);
type FnHotwords = unsafe extern "C" fn(*mut CrispSession, *const c_char, c_float) -> c_int;

/// What the assistant is called, given to the recogniser as a word to expect: see
/// [`AsrEngine::load`].
pub const NAME: &str = "Zen";

/// The sentence the library primes the recogniser with ahead of the hotwords, word for word as
/// it appears in crispasr.dll.
const HOTWORD_PROMPT: &str = "The following words may appear in the audio:";

/// The recogniser's own hotword prompt, taken back out of what it heard. Returns what is left,
/// and whether the prompt was there.
///
/// Given sound with no words in it, the recogniser sometimes writes out the sentence it was
/// primed with rather than nothing: "The following words may appear in the audio: Zen." Measured
/// on 32 clips holding no words (hums, noise bursts, clicks, speech-shaped noise, reversed
/// speech; 2026-09-23), it did so on 6 with the hotword set and on none without, hums at 150 and
/// 200 Hz among them - which the detector passes, as it would someone humming. The filter took
/// the sentence for a question and passed it on word for word, and Zen answered it.
pub(crate) fn without_hotword_prompt(text: &str) -> (String, bool) {
    // ASCII lowering keeps every byte offset valid in the original.
    let lower = text.to_ascii_lowercase();
    let prompt = HOTWORD_PROMPT.to_ascii_lowercase();
    let name = NAME.to_ascii_lowercase();
    let mut kept = String::with_capacity(text.len());
    let mut at = 0;
    let mut found = false;
    while let Some(offset) = lower[at..].find(&prompt) {
        found = true;
        kept.push_str(&text[at..at + offset]);
        let mut end = at + offset + prompt.len();
        let skip_space = |end: usize| end + lower[end..].len() - lower[end..].trim_start().len();
        end = skip_space(end);
        // The hotword it was primed with, and the full stop the model puts after it.
        if lower[end..].starts_with(&name)
            && !lower[end + name.len()..].starts_with(|c: char| c.is_alphanumeric())
        {
            end += name.len();
            if lower[end..].starts_with(['.', ',', '!', '?', ';', ':']) {
                end += 1;
            }
        }
        kept.push(' ');
        at = skip_space(end);
    }
    kept.push_str(&text[at..]);
    (kept.split_whitespace().collect::<Vec<_>>().join(" "), found)
}

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("asr asset is missing: {0}")]
    Missing(String),
    #[error("failed to load crispasr.dll: {0}")]
    Load(String),
    #[error("crispasr symbol {0} is missing")]
    Symbol(String),
    #[error("crispasr session could not be opened")]
    SessionOpen,
    #[error("crispasr returned no result")]
    NoResult,
    #[error("asr worker has stopped")]
    WorkerGone,
    #[error("asr worker could not start: {0}")]
    WorkerStart(#[source] std::io::Error),
    #[error("asr queue is full; transcription is falling behind")]
    Backlogged,
}

/// One transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub latency_ms: f32,
}

/// Frees the C-side result even if a later step panics or returns early.
struct ResultGuard {
    pointer: *mut CrispResult,
    free: FnResultFree,
}

impl Drop for ResultGuard {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.free)(self.pointer) };
        }
    }
}

/// A loaded Qwen3-ASR session.
pub struct AsrEngine {
    _library: libloading::Library,
    session: Mutex<*mut CrispSession>,
    transcribe: FnTranscribe,
    segments: FnSegments,
    segment_text: FnSegmentText,
    result_free: FnResultFree,
    close: FnClose,
}

// The raw handle is only ever touched behind the mutex, and the library outlives it.
unsafe impl Send for AsrEngine {}
unsafe impl Sync for AsrEngine {}

impl AsrEngine {
    /// Loads the session from a directory holding `crispasr.dll`, its ggml dependencies, and the
    /// model.
    pub fn load(directory: impl AsRef<Path>, threads: usize) -> Result<Self, AsrError> {
        let directory = directory.as_ref();
        let library_path = directory.join("crispasr.dll");
        let model_path = directory.join("qwen3-asr-1.7b-q4_k.gguf");
        for required in [&library_path, &model_path] {
            if !required.is_file() {
                return Err(AsrError::Missing(required.display().to_string()));
            }
        }

        #[cfg(target_os = "windows")]
        unsafe {
            use std::os::windows::ffi::OsStrExt;
            // crispasr.dll resolves its ggml dependencies by name. Without pointing the loader at
            // this directory first, Windows may bind them to the CUDA-enabled copies in bin/,
            // which belong to llama-server and expect the GPU this model is deliberately off.
            let wide: Vec<u16> = directory
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            SetDllDirectoryW(wide.as_ptr());
            for dependency in ["ggml-base.dll", "ggml-cpu.dll", "ggml.dll"] {
                let path = directory.join(dependency);
                let wide: Vec<u16> = path
                    .as_os_str()
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();
                LoadLibraryW(wide.as_ptr());
            }
        }

        let library = unsafe { libloading::Library::new(&library_path) }
            .map_err(|error| AsrError::Load(error.to_string()))?;

        unsafe {
            let symbol = |name: &str| -> Result<*const (), AsrError> {
                library
                    .get::<*const ()>(name.as_bytes())
                    .map(|found| *found)
                    .map_err(|_| AsrError::Symbol(name.to_string()))
            };
            let open: FnOpen = std::mem::transmute(symbol("crispasr_session_open")?);
            let transcribe: FnTranscribe =
                std::mem::transmute(symbol("crispasr_session_transcribe")?);
            let segments: FnSegments =
                std::mem::transmute(symbol("crispasr_session_result_n_segments")?);
            let segment_text: FnSegmentText =
                std::mem::transmute(symbol("crispasr_session_result_segment_text")?);
            let result_free: FnResultFree =
                std::mem::transmute(symbol("crispasr_session_result_free")?);
            let close: FnClose = std::mem::transmute(symbol("crispasr_session_close")?);
            // Older builds of the library do not have it, and Zen works without it.
            let hotwords: Option<FnHotwords> = symbol("crispasr_session_set_hotwords")
                .ok()
                .map(|found| std::mem::transmute(found));

            let model = CString::new(model_path.to_string_lossy().as_ref())
                .map_err(|error| AsrError::Load(error.to_string()))?;
            let session = open(model.as_ptr(), threads as c_int);
            if session.is_null() {
                return Err(AsrError::SessionOpen);
            }
            // Zen's name is not a common word, and the recogniser wrote it as Zan, Zhen, Ben,
            // Dan or "then" often enough that "Hey Zen" came out as something else. Naming it
            // as a word to expect is the fix at the source: measured over 125 recordings of six
            // wake phrases (clean, noisy at 5 and 0 dB, and pitched down), the name came out
            // right 57 times without it and 93 with it, and on clean speech 25 times out of 25.
            //
            // It costs a little: in noise the recogniser returns nothing slightly more often
            // (6 of 54 ordinary utterances against 1 without), which the blank retry below
            // recovers every time - measured 19 of 19 - and accuracy over the same set was no
            // worse. Sound-alike sentences that are not addressed to Zen wake it no more often
            // than before (3 of 30 either way). Rechecked 2026-09-23 on ten sentences with the
            // name and its sound-alikes: "Zen" right in all of them with it, "Then" and "Zan"
            // without; "when", "send" and "then" untouched either way.
            //
            // Its one real cost is in sound with no words: the recogniser can write the prompt
            // it was primed with back out, which `without_hotword_prompt` takes back.
            if let Some(set_hotwords) = hotwords {
                if let Ok(name) = CString::new(NAME) {
                    // The boost applies to the word-lattice backends, not to this one, so it is
                    // the library's own default.
                    set_hotwords(session, name.as_ptr(), 1.5);
                }
            }

            let engine = Self {
                _library: library,
                session: Mutex::new(session),
                transcribe,
                segments,
                segment_text,
                result_free,
                close,
            };
            // Measured: the first inference costs about 2.5x a warm one (2469 ms against
            // ~980 ms for the same audio). Paying that here, during startup, keeps it off the
            // user's first utterance - which is the one that decides whether this feels fast.
            engine.warm_up();
            Ok(engine)
        }
    }

    /// Runs one throwaway inference so the first real utterance meets a warm session.
    pub fn warm_up(&self) {
        let silence = vec![0.0f32; SAMPLE_RATE as usize / 2];
        let _ = self.transcribe(&silence);
    }

    /// A sensible worker thread count for this machine.
    ///
    /// Not every core: transcription shares the CPU with speech synthesis and with the model
    /// runtime's host-side work, and taking all of them makes those stall instead.
    pub fn default_threads() -> usize {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4);
        available.saturating_sub(4).clamp(2, 8)
    }

    /// Transcribes 16 kHz mono float samples.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
        if samples.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                latency_ms: 0.0,
            });
        }

        let started = Instant::now();
        let guard = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = *guard;
        if session.is_null() {
            return Err(AsrError::SessionOpen);
        }

        let raw = unsafe { (self.transcribe)(session, samples.as_ptr(), samples.len() as c_int) };
        if raw.is_null() {
            return Err(AsrError::NoResult);
        }
        let result = ResultGuard {
            pointer: raw,
            free: self.result_free,
        };

        let count = unsafe { (self.segments)(result.pointer) };
        let mut text = String::new();
        for index in 0..count {
            let pointer = unsafe { (self.segment_text)(result.pointer, index) };
            if pointer.is_null() {
                continue;
            }
            let piece = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(piece.trim());
        }

        Ok(Transcript {
            text: text.trim().to_string(),
            latency_ms: started.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

impl Drop for AsrEngine {
    fn drop(&mut self) {
        // A panic mid-transcription poisons the lock but leaves the session as valid as it was;
        // skipping the close then would leak the whole model.
        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !guard.is_null() {
            unsafe { (self.close)(*guard) };
            *guard = std::ptr::null_mut();
        }
    }
}

// ============================================================================================
// Worker
// ============================================================================================

/// What came back from the worker.
#[derive(Debug, Clone, PartialEq)]
pub enum AsrOutcome {
    /// One piece of a long utterance. Arrives while later pieces are still running.
    Partial {
        sequence: u64,
        chunk: usize,
        text: String,
    },
    /// The whole utterance, with overlaps joined.
    Complete {
        sequence: u64,
        transcript: Transcript,
        /// Speech the detector was sure of that still came back with no words after a retry,
        /// in milliseconds. Anything above zero means the transcript has a hole in it.
        unheard_ms: usize,
        /// How long the piece waited for the recogniser before work on it began.
        queued_ms: u64,
    },
    Failed {
        sequence: u64,
        reason: String,
    },
}

struct Job {
    sequence: u64,
    samples: Vec<f32>,
    probabilities: Vec<f32>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    submitted: Instant,
}

/// Runs transcription on its own thread with a bounded queue.
pub trait Recognizer: Send + Sync {
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError>;
    fn transcribe_cancellable(
        &self,
        samples: &[f32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<Transcript, AsrError> {
        if cancelled.load(Ordering::Acquire) {
            return Err(AsrError::WorkerGone);
        }
        self.transcribe(samples)
    }
}
impl Recognizer for AsrEngine {
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
        AsrEngine::transcribe(self, samples)
    }
}

pub struct AsrWorker {
    jobs: Option<SyncSender<Job>>,
    results: Receiver<AsrOutcome>,
    next_sequence: u64,
    handle: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl AsrWorker {
    /// `depth` bounds how many utterances may wait. Small on purpose: a deep queue just means
    /// answering a question the user asked a minute ago.
    pub fn spawn(
        engine: Arc<dyn Recognizer>,
        depth: usize,
        chunking: ChunkConfig,
    ) -> Result<Self, AsrError> {
        let (jobs, job_rx) = sync_channel::<Job>(depth.max(1));
        let (result_tx, results) = sync_channel::<AsrOutcome>(depth.max(1) * 4);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = shutdown.clone();

        let handle = thread::Builder::new()
            .name("zen-asr".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if job.cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        transcribe_job(engine.as_ref(), &job, &chunking, &result_tx)
                    }))
                    .unwrap_or_else(|_| AsrOutcome::Failed {
                        sequence: job.sequence,
                        reason: "recognition worker panicked".into(),
                    });
                    if job.cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    let mut outcome = outcome;
                    loop {
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        if job.cancelled.load(Ordering::Acquire) {
                            break;
                        }
                        match result_tx.try_send(outcome) {
                            Ok(()) => break,
                            Err(TrySendError::Full(value)) => outcome = value,
                            Err(TrySendError::Disconnected(_)) => return,
                        }
                        thread::sleep(std::time::Duration::from_millis(2));
                    }
                }
            })
            .map_err(AsrError::WorkerStart)?;

        Ok(Self {
            jobs: Some(jobs),
            results,
            next_sequence: 0,
            handle: Some(handle),
            shutdown,
        })
    }

    /// Queues an utterance, returning its sequence number. `cancelled` abandons it, and every
    /// other piece of the same turn, once set.
    ///
    /// Never blocks: the capture path must keep running even when transcription is behind, so a
    /// full queue is reported rather than waited on.
    pub fn submit_cancellable(
        &mut self,
        segment: SpeechSegment,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<u64, AsrError> {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        let job = Job {
            sequence,
            samples: segment.samples,
            probabilities: segment.probabilities,
            cancelled,
            submitted: Instant::now(),
        };
        match self
            .jobs
            .as_ref()
            .ok_or(AsrError::WorkerGone)?
            .try_send(job)
        {
            Ok(()) => Ok(sequence),
            Err(TrySendError::Full(_)) => Err(AsrError::Backlogged),
            Err(TrySendError::Disconnected(_)) => Err(AsrError::WorkerGone),
        }
    }

    /// Collects whatever has finished, without blocking.
    pub fn poll(&self) -> Vec<AsrOutcome> {
        self.results.try_iter().collect()
    }

    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }

    pub fn shutdown(&mut self, timeout: std::time::Duration) -> bool {
        self.shutdown.store(true, Ordering::Release);
        self.jobs.take();
        let deadline = Instant::now() + timeout;
        while !self.is_finished() && Instant::now() < deadline {
            self.poll();
            thread::sleep(std::time::Duration::from_millis(5));
        }
        if self.is_finished() {
            if let Some(handle) = self.handle.take() {
                return handle.join().is_ok();
            }
            return true;
        }
        false
    }
}

impl Drop for AsrWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.jobs.take();
    }
}

fn transcribe_job(
    engine: &dyn Recognizer,
    job: &Job,
    chunking: &ChunkConfig,
    partials: &SyncSender<AsrOutcome>,
) -> AsrOutcome {
    let started = Instant::now();
    let queued_ms = started.saturating_duration_since(job.submitted).as_millis() as u64;
    let chunks = plan_chunks(job.samples.len(), &job.probabilities, chunking);

    let mut pieces = Vec::with_capacity(chunks.len());
    let mut unheard_ms = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        if job.cancelled.load(Ordering::Acquire) {
            return AsrOutcome::Failed {
                sequence: job.sequence,
                reason: "recognition cancelled".into(),
            };
        }
        let slice =
            &job.samples[chunk.start.min(job.samples.len())..chunk.end.min(job.samples.len())];
        let mut result = engine.transcribe_cancellable(slice, &job.cancelled);
        // The recogniser reading back its own hotword prompt is it saying there were no words,
        // so a chunk that held nothing else is taken as honest silence: no retry, and nothing
        // reported unheard.
        let mut prompt_only = strip_hotword_prompt(&mut result);
        // Measured in a live session: three seconds of audio the detector rated speech on four
        // frames out of five came back with no words at all, and the decoder had stopped early
        // rather than timed out. The recogniser decodes the same way every time, so asking again
        // with identical audio would only repeat the answer. A little silence either side is
        // what changes: a model given speech that starts or stops on its very first or last
        // sample is the case most likely to end its output before it has begun.
        let blank = |result: &Result<Transcript, AsrError>| matches!(result, Ok(transcript) if transcript.text.trim().is_empty());
        if blank(&result)
            && !prompt_only
            && voiced_share(&job.probabilities, chunk.start, chunk.end) >= MIN_VOICED_SHARE
        {
            result = engine.transcribe_cancellable(&padded(slice), &job.cancelled);
            prompt_only = strip_hotword_prompt(&mut result);
            if blank(&result) && !prompt_only {
                unheard_ms += slice.len() * 1000 / SAMPLE_RATE as usize;
                keep_unheard(slice, job.sequence, index);
            }
        }
        match result {
            Ok(transcript) => {
                // Emitted as it finishes rather than held back, so a long utterance shows
                // progress instead of a silent wait.
                if chunks.len() > 1 {
                    // Partial display is optional; never block inference/cancellation on it.
                    let _ = partials.try_send(AsrOutcome::Partial {
                        sequence: job.sequence,
                        chunk: index,
                        text: transcript.text.clone(),
                    });
                }
                pieces.push(transcript.text);
            }
            Err(error) => {
                return AsrOutcome::Failed {
                    sequence: job.sequence,
                    reason: error.to_string(),
                }
            }
        }
    }

    AsrOutcome::Complete {
        sequence: job.sequence,
        transcript: Transcript {
            text: join_overlapping(&pieces),
            latency_ms: started.elapsed().as_secs_f32() * 1000.0,
        },
        unheard_ms,
        queued_ms,
    }
}

/// Takes the hotword prompt out of a result in place (see [`without_hotword_prompt`]). True when
/// the prompt was all there was.
fn strip_hotword_prompt(result: &mut Result<Transcript, AsrError>) -> bool {
    let Ok(transcript) = result else {
        return false;
    };
    let (text, found) = without_hotword_prompt(&transcript.text);
    transcript.text = text;
    found && transcript.text.is_empty()
}

/// Share of a chunk the detector was sure was speech, above which an empty transcript is a
/// failure rather than an answer. A majority: below it the chunk may honestly hold no words.
const MIN_VOICED_SHARE: f32 = 0.5;

/// Silence put either side of a chunk that came back blank, before asking once more.
const RETRY_PADDING_MS: usize = 250;

/// Share of the frames in `start..end` rated as speech.
fn voiced_share(probabilities: &[f32], start: usize, end: usize) -> f32 {
    let first = start / VAD_FRAME_SAMPLES;
    let last = end.div_ceil(VAD_FRAME_SAMPLES).min(probabilities.len());
    if first >= last {
        return 0.0;
    }
    let frames = &probabilities[first..last];
    frames.iter().filter(|p| **p >= SPEECH_PROBABILITY).count() as f32 / frames.len() as f32
}

fn padded(samples: &[f32]) -> Vec<f32> {
    let pad = SAMPLE_RATE as usize * RETRY_PADDING_MS / 1000;
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    out.resize(pad, 0.0);
    out.extend_from_slice(samples);
    out.resize(out.len() + pad, 0.0);
    out
}

/// Keeps a chunk that could not be recognised, only when asked to.
///
/// Nothing anyone says is written anywhere by default. Setting `ZEN_UNHEARD_AUDIO_DIR` to a
/// directory saves each chunk that still came back blank as a 16 kHz WAV there, because without
/// the audio nobody can say whether the recogniser, the microphone or the detector was at fault.
fn keep_unheard(samples: &[f32], sequence: u64, chunk: usize) {
    let Some(dir) = std::env::var_os("ZEN_UNHEARD_AUDIO_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis());
    let _ = write_wav(
        &dir.join(format!("unheard-{stamp}-{sequence}-{chunk}.wav")),
        samples,
    );
}

fn write_wav(path: &Path, samples: &[f32]) -> std::io::Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

/// Joins chunk transcripts, removing the words duplicated by their overlap.
///
/// Chunks deliberately overlap so a word on a seam is heard whole at least once - which means it
/// is also transcribed twice. Plain concatenation would stutter ("book a table a table for two"),
/// so the repeated run is detected and dropped.
pub fn join_overlapping(pieces: &[String]) -> String {
    let mut joined = String::new();
    for piece in pieces {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        if joined.is_empty() {
            joined.push_str(piece);
            continue;
        }
        let left: Vec<&str> = joined.split_whitespace().collect();
        let right: Vec<&str> = piece.split_whitespace().collect();
        // Longest run of words that ends `joined` and starts `piece`. Capped because a genuine
        // repetition ("very very good") is shorter than an overlap artefact, and because an
        // unbounded search would happily merge two unrelated sentences.
        let limit = left.len().min(right.len()).min(12);
        let mut overlap = 0;
        for length in (1..=limit).rev() {
            let tail = &left[left.len() - length..];
            let head = &right[..length];
            if tail.iter().zip(head).all(|(a, b)| same_word(a, b)) {
                overlap = length;
                break;
            }
        }
        let remainder = right[overlap..].join(" ");
        if !remainder.is_empty() {
            joined.push(' ');
            joined.push_str(&remainder);
        }
    }
    joined
}

/// Whether two words on either side of a seam are the same word. Each piece is punctuated as if
/// it were a whole utterance, so the first may end a sentence the second begins: "table." and
/// "Table" are one word heard twice.
fn same_word(a: &str, b: &str) -> bool {
    let bare = |word: &str| {
        word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let (a, b) = (bare(a), bare(b));
    !a.is_empty() && a == b
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// A recogniser that answers from a rule, and counts how often it was asked.
    struct Scripted {
        answer: fn(&[f32]) -> &'static str,
        calls: AtomicUsize,
    }
    impl Recognizer for Scripted {
        fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Transcript {
                text: (self.answer)(samples).to_string(),
                latency_ms: 0.0,
            })
        }
    }

    /// One second of loud audio, with the detector's view of it at `probability` throughout.
    fn job(probability: f32) -> Job {
        let samples = vec![0.3f32; SAMPLE_RATE as usize];
        Job {
            sequence: 7,
            probabilities: vec![probability; samples.len() / VAD_FRAME_SAMPLES],
            samples,
            cancelled: Arc::new(AtomicBool::new(false)),
            submitted: Instant::now(),
        }
    }

    /// A recogniser that takes a fixed time over every piece.
    struct Slow;
    impl Recognizer for Slow {
        fn transcribe(&self, _: &[f32]) -> Result<Transcript, AsrError> {
            thread::sleep(std::time::Duration::from_millis(150));
            Ok(Transcript {
                text: "words".into(),
                latency_ms: 150.0,
            })
        }
    }

    #[test]
    fn time_spent_waiting_behind_another_piece_is_measured() {
        // Recognition runs one piece at a time, so a question's last piece can sit behind the
        // one before it. That wait used to be inferred from timestamps after the fact.
        let mut worker = AsrWorker::spawn(Arc::new(Slow), 4, ChunkConfig::default()).unwrap();
        let segment = || SpeechSegment {
            samples: vec![0.3; SAMPLE_RATE as usize / 2],
            probabilities: vec![0.9; SAMPLE_RATE as usize / 2 / VAD_FRAME_SAMPLES],
            end: crate::audio::SegmentEnd::Pause,
            overlaps_previous: false,
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let first = worker
            .submit_cancellable(segment(), cancelled.clone())
            .unwrap();
        let second = worker.submit_cancellable(segment(), cancelled).unwrap();
        let mut waits = std::collections::HashMap::new();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while waits.len() < 2 && Instant::now() < deadline {
            for outcome in worker.poll() {
                if let AsrOutcome::Complete {
                    sequence,
                    queued_ms,
                    ..
                } = outcome
                {
                    waits.insert(sequence, queued_ms);
                }
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            waits[&first] < 100,
            "the first piece waited {} ms",
            waits[&first]
        );
        assert!(
            waits[&second] >= 140,
            "the second piece waited {} ms",
            waits[&second]
        );
        assert!(worker.shutdown(std::time::Duration::from_secs(5)));
    }

    fn run(engine: &Scripted, job: &Job) -> (String, usize) {
        let (partials, _keep) = sync_channel(16);
        match transcribe_job(engine, job, &ChunkConfig::default(), &partials) {
            AsrOutcome::Complete {
                transcript,
                unheard_ms,
                ..
            } => (transcript.text, unheard_ms),
            _ => panic!("expected a completed transcript"),
        }
    }

    #[test]
    fn clear_speech_that_comes_back_blank_is_asked_about_again_with_room_around_it() {
        // Blank unless the audio starts in silence: the failure the retry exists for.
        let engine = Scripted {
            answer: |samples| if samples[0] == 0.0 { "hello there" } else { "" },
            calls: AtomicUsize::new(0),
        };
        let (text, unheard) = run(&engine, &job(0.9));
        assert_eq!(text, "hello there");
        assert_eq!(unheard, 0);
        assert_eq!(engine.calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn speech_that_stays_blank_is_reported_rather_than_dropped() {
        // Before this, three seconds of clear speech vanished from the middle of a question and
        // nothing anywhere said so.
        let engine = Scripted {
            answer: |_| "",
            calls: AtomicUsize::new(0),
        };
        let (text, unheard) = run(&engine, &job(0.9));
        assert_eq!(text, "");
        assert_eq!(unheard, 1_000);
        assert_eq!(
            engine.calls.load(Ordering::Relaxed),
            2,
            "one retry, not more"
        );
    }

    #[test]
    fn a_blank_answer_for_audio_that_barely_sounded_like_speech_is_believed() {
        // A cough or a door the detector only half believed in may honestly hold no words.
        let engine = Scripted {
            answer: |_| "",
            calls: AtomicUsize::new(0),
        };
        let (text, unheard) = run(&engine, &job(0.2));
        assert_eq!(text, "");
        assert_eq!(unheard, 0);
        assert_eq!(
            engine.calls.load(Ordering::Relaxed),
            1,
            "no retry for noise"
        );
    }

    #[test]
    fn the_hotword_prompt_read_back_from_a_hum_is_silence_not_a_question() {
        // What the recogniser wrote for a 150 Hz hum the detector had passed as speech.
        let engine = Scripted {
            answer: |_| "The following words may appear in the audio: Zen.",
            calls: AtomicUsize::new(0),
        };
        let (text, unheard) = run(&engine, &job(0.9));
        assert_eq!(text, "");
        assert_eq!(
            unheard, 0,
            "the recogniser said there were no words; believe it"
        );
        assert_eq!(
            engine.calls.load(Ordering::Relaxed),
            1,
            "and do not ask again"
        );
    }

    #[test]
    fn words_heard_around_a_read_back_prompt_are_kept() {
        let engine = Scripted {
            answer: |_| "The following words may appear in the audio: Zen. What time is it?",
            calls: AtomicUsize::new(0),
        };
        assert_eq!(run(&engine, &job(0.9)).0, "What time is it?");
    }

    #[test]
    fn only_the_prompt_itself_is_taken_out() {
        let cases = [
            (
                "the following words may appear in the audio: zen",
                ("", true),
            ),
            ("The following words may appear in the audio:", ("", true)),
            (
                "Hey Zen. The following words may appear in the audio: Zen.",
                ("Hey Zen.", true),
            ),
            ("Zen.", ("Zen.", false)),
            (
                "The following words are the ones I need.",
                ("The following words are the ones I need.", false),
            ),
            (
                "The following words may appear in the audio: Zenith cameras.",
                ("Zenith cameras.", true),
            ),
        ];
        for (input, (kept, found)) in cases {
            assert_eq!(
                without_hotword_prompt(input),
                (kept.to_string(), found),
                "{input:?}"
            );
        }
    }

    #[test]
    fn a_written_wav_is_a_valid_sixteen_kilohertz_file() {
        let dir = std::env::temp_dir().join(format!("zen-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.wav");
        write_wav(&path, &[0.0, 0.5, -0.5, 1.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        assert_eq!(bytes.len(), 44 + 4 * 2);
    }

    #[test]
    fn a_single_piece_is_returned_unchanged() {
        assert_eq!(join_overlapping(&["book a table".into()]), "book a table");
    }

    #[test]
    fn the_overlap_between_chunks_is_not_repeated() {
        // Chunks overlap by design so a word on the seam survives; without dedup the join
        // stutters and the model is handed text no human said.
        let joined = join_overlapping(&["book a table".into(), "a table for two please".into()]);
        assert_eq!(joined, "book a table for two please");
    }

    #[test]
    fn a_one_word_overlap_is_removed() {
        assert_eq!(
            join_overlapping(&["turn on the".into(), "the kitchen light".into()]),
            "turn on the kitchen light"
        );
    }

    #[test]
    fn overlap_matching_ignores_the_punctuation_each_piece_adds() {
        assert_eq!(
            join_overlapping(&["Please book a table.".into(), "Table for two.".into()]),
            "Please book a table. for two."
        );
        // Punctuation on its own is not a word both sides share.
        assert_eq!(
            join_overlapping(&["Wait -".into(), "- what?".into()]),
            "Wait - - what?"
        );
    }

    #[test]
    fn overlap_matching_ignores_capitalisation() {
        // Transcribers capitalise the first word of a segment, so the same word can come back
        // differently cased on each side of a seam.
        assert_eq!(
            join_overlapping(&["what is the".into(), "The weather today".into()]),
            "what is the weather today"
        );
    }

    #[test]
    fn unrelated_pieces_are_simply_concatenated() {
        assert_eq!(
            join_overlapping(&["hello there".into(), "goodbye now".into()]),
            "hello there goodbye now"
        );
    }

    #[test]
    fn empty_pieces_do_not_introduce_stray_spaces() {
        let joined = join_overlapping(&["hello".into(), String::new(), "world".into()]);
        assert_eq!(joined, "hello world");
        assert!(!joined.contains("  "));
    }

    #[test]
    fn nothing_at_all_joins_to_nothing() {
        assert_eq!(join_overlapping(&[]), "");
        assert_eq!(join_overlapping(&[String::new()]), "");
    }

    #[test]
    fn a_deliberate_repetition_is_not_swallowed_whole() {
        // A word repeated across a seam cannot be told from the overlap, so one "very" of "very
        // very good" is lost - but only that word: the dedup must never take the rest of the
        // sentence with it.
        let joined = join_overlapping(&["it was very".into(), "very good indeed".into()]);
        assert_eq!(joined, "it was very good indeed");
    }

    #[test]
    fn a_missing_model_directory_is_reported_clearly() {
        let outcome = AsrEngine::load(std::env::temp_dir().join("zen-no-such-model"), 6);
        assert!(matches!(outcome, Err(AsrError::Missing(_))));
    }
}
