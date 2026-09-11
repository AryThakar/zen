// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Speech to text: Qwen3-ASR through `crispasr.dll`, off the capture thread.
//!
//! The model is roughly 1.4 GB and runs on the CPU, which shapes the whole design:
//!
//! - **One session, not a pool.** A second session would cost another 1.4 GB of resident memory,
//!   and since transcription is CPU-bound on the same twelve threads, two of them competing would
//!   finish two utterances no faster than one session finishes them in turn.
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
        atomic::{AtomicUsize, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use crate::audio::{plan_chunks, ChunkConfig, SpeechSegment, SAMPLE_RATE};

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
    pub audio_ms: usize,
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
    pub model_path: PathBuf,
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

            let model = CString::new(model_path.to_string_lossy().as_ref())
                .map_err(|error| AsrError::Load(error.to_string()))?;
            let session = open(model.as_ptr(), threads as c_int);
            if session.is_null() {
                return Err(AsrError::SessionOpen);
            }

            let engine = Self {
                _library: library,
                session: Mutex::new(session),
                transcribe,
                segments,
                segment_text,
                result_free,
                close,
                model_path,
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
        let audio_ms = samples.len() * 1000 / SAMPLE_RATE as usize;
        if samples.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                audio_ms: 0,
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
            audio_ms,
            latency_ms: started.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

impl Drop for AsrEngine {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.session.lock() {
            if !guard.is_null() {
                unsafe { (self.close)(*guard) };
                *guard = std::ptr::null_mut();
            }
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
        of: usize,
        text: String,
    },
    /// The whole utterance, with overlaps joined.
    Complete {
        sequence: u64,
        transcript: Transcript,
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
    dropped: Arc<AtomicUsize>,
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
        let dropped = Arc::new(AtomicUsize::new(0));
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
            dropped,
            handle: Some(handle),
            shutdown,
        })
    }

    /// Queues an utterance, returning its sequence number.
    ///
    /// Never blocks: the capture path must keep running even when transcription is behind, so a
    /// full queue is reported rather than waited on.
    pub fn submit(&mut self, segment: SpeechSegment) -> Result<u64, AsrError> {
        self.submit_cancellable(segment, Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }

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
        };
        match self
            .jobs
            .as_ref()
            .ok_or(AsrError::WorkerGone)?
            .try_send(job)
        {
            Ok(()) => Ok(sequence),
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Err(AsrError::Backlogged)
            }
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
    let audio_ms = job.samples.len() * 1000 / SAMPLE_RATE as usize;
    let chunks = plan_chunks(job.samples.len(), &job.probabilities, chunking);

    let mut pieces = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        if job.cancelled.load(Ordering::Acquire) {
            return AsrOutcome::Failed {
                sequence: job.sequence,
                reason: "recognition cancelled".into(),
            };
        }
        let slice =
            &job.samples[chunk.start.min(job.samples.len())..chunk.end.min(job.samples.len())];
        match engine.transcribe_cancellable(slice, &job.cancelled) {
            Ok(transcript) => {
                // Emitted as it finishes rather than held back, so a long utterance shows
                // progress instead of a silent wait.
                if chunks.len() > 1 {
                    // Partial display is optional; never block inference/cancellation on it.
                    let _ = partials.try_send(AsrOutcome::Partial {
                        sequence: job.sequence,
                        chunk: index,
                        of: chunks.len(),
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
            audio_ms,
            latency_ms: started.elapsed().as_secs_f32() * 1000.0,
        },
    }
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
            if tail
                .iter()
                .zip(head)
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
            {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // "very very good" is something a person actually says. The dedup must not treat the
        // second "very" as an overlap artefact and delete the rest of the sentence.
        let joined = join_overlapping(&["it was very".into(), "very good indeed".into()]);
        assert_eq!(joined, "it was very good indeed");
    }

    #[test]
    fn a_missing_model_directory_is_reported_clearly() {
        let outcome = AsrEngine::load(r"C:\zen-ai\model\does-not-exist", 6);
        assert!(matches!(outcome, Err(AsrError::Missing(_))));
    }
}
