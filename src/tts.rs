// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Speech synthesis: Qwen-TTS through `qwen.dll`.
//!
//! Loaded at runtime rather than linked, because no import library ships with the DLL - the same
//! reason the recogniser goes through `libloading`.
//!
//! The engine streams. Synthesis calls back with each codec chunk as it is produced, roughly
//! every 250 ms of audio, so playback starts long before the sentence is finished. That matters
//! more than raw speed: a listener judges responsiveness by when sound *starts*, and a synthesiser
//! that returns a whole utterance 900 ms later feels slower than one that starts at 250 ms and
//! takes longer overall.
//!
//! Cancellation runs through a callback the C side polls between chunks. It is the mechanism
//! behind interruption: when the user talks over a reply, the flag flips and synthesis abandons
//! the rest of the sentence instead of finishing into a buffer nobody will hear.

use std::{
    ffi::{c_char, c_void, CStr, CString},
    mem::MaybeUninit,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

use crate::resample::StreamResampler;

/// Rate the codec produces. Distinct from the 16 kHz capture side.
pub const TTS_SAMPLE_RATE: u32 = 24_000;

/// Audio per character at the slowest pace this voice has been measured speaking: 47.4 to
/// 56.0 ms across eight reply-shaped passages and eight lengths of one passage
/// (`examples/phrasepace.rs`). Speech length follows characters far more closely than words,
/// which vary from "a" to "internationalisation".
pub const SLOWEST_MS_PER_CHAR: usize = 56;

/// The codec emits one frame per 80 ms of audio - it is the "12 Hz" tokenizer, at 12.5 Hz.
const CODEC_FRAME_MS: usize = 80;

/// Frames synthesis may produce for `text`.
///
/// Budgeted from the text rather than left at a fixed ceiling: an unbounded budget on a short
/// phrase lets the model ramble past the end of the sentence. Twice the slowest measured pace
/// leaves room for a slow, emphatic delivery. It was five frames a word, which gave a phrase of
/// long words less time than it takes to say them, and cut it off before its end.
fn frame_budget(text: &str) -> i32 {
    let frames = text.trim().chars().count() * 2 * SLOWEST_MS_PER_CHAR / CODEC_FRAME_MS;
    frames.clamp(100, 1_500) as i32
}

#[repr(C)]
pub struct QtContext {
    _private: [u8; 0],
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtStatus {
    Ok = 0,
    Error = 1,
    Cancelled = 2,
}

#[repr(C)]
pub struct QtAudio {
    pub samples: *mut f32,
    pub n_samples: i32,
    pub sample_rate: i32,
    pub channels: i32,
}

#[repr(C)]
pub struct QtVoiceRef {
    pub ref_spk_emb: *mut f32,
    pub ref_spk_dim: i32,
    pub ref_codes: *mut i32,
    pub ref_t: i32,
    pub num_codebooks: i32,
}

#[repr(C)]
pub struct QtInitParams {
    pub abi_version: i32,
    pub talker_path: *const c_char,
    pub codec_path: *const c_char,
    pub use_fa: bool,
    pub clamp_fp16: bool,
    pub max_batch: i32,
}

pub type QtCancelCb = Option<unsafe extern "C" fn(*mut c_void) -> bool>;
pub type QtAudioChunkCb = Option<unsafe extern "C" fn(*const f32, i32, *mut c_void) -> bool>;

#[repr(C)]
pub struct QtTtsParams {
    pub abi_version: i32,
    pub text: *const c_char,
    pub lang: *const c_char,
    pub instruct: *const c_char,
    pub speaker: *const c_char,
    pub ref_audio_24k: *const f32,
    pub ref_n_samples: i32,
    pub ref_text: *const c_char,
    pub seed: i64,
    pub max_new_tokens: i32,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: i32,
    pub subtalker_top_p: f32,
    pub dump_dir: *const c_char,
    pub cancel: QtCancelCb,
    pub cancel_user_data: *mut c_void,
    pub on_chunk: QtAudioChunkCb,
    pub on_chunk_user_data: *mut c_void,
    pub codec_chunk_sec: f32,
    pub codec_left_context_sec: f32,
    pub ref_spk_emb: *const f32,
    pub ref_spk_dim: i32,
    pub ref_codes: *const i32,
    pub ref_t: i32,
}

type FnInitDefaults = unsafe extern "C" fn(*mut QtInitParams);
type FnTtsDefaults = unsafe extern "C" fn(*mut QtTtsParams);
type FnInit = unsafe extern "C" fn(*const QtInitParams) -> *mut QtContext;
type FnFree = unsafe extern "C" fn(*mut QtContext);
type FnSynthesize = unsafe extern "C" fn(*mut QtContext, *const QtTtsParams, *mut QtAudio) -> i32;
type FnAudioFree = unsafe extern "C" fn(*mut QtAudio);
type FnExtractRef = unsafe extern "C" fn(*mut QtContext, *const f32, i32, *mut QtVoiceRef) -> i32;
type FnVoiceRefFree = unsafe extern "C" fn(*mut QtVoiceRef);
type FnLastError = unsafe extern "C" fn() -> *const c_char;

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("tts asset is missing: {0}")]
    Missing(String),
    #[error("failed to load qwen.dll: {0}")]
    Load(String),
    #[error("qwen symbol {0} is missing")]
    Symbol(String),
    #[error("qwen context could not be created: {0}")]
    Init(String),
    #[error("reference voice could not be read: {0}")]
    Reference(String),
    #[error("synthesis failed: {0}")]
    Synthesis(String),
    #[error("synthesis was cancelled")]
    Cancelled,
}

/// Where the chunk callback delivers audio.
type AudioCallback<'a> = &'a mut dyn FnMut(&[f32]) -> bool;
struct ChunkSink<'a> {
    /// Streaming destination. When absent, chunks accumulate instead.
    callback: Option<AudioCallback<'a>>,
    collected: Vec<f32>,
    cancelled: &'a AtomicBool,
}

/// Receives one codec chunk from the C side.
///
/// Wrapped in `catch_unwind` because a panic crossing an FFI boundary is undefined behaviour;
/// returning `false` instead tells the synthesiser to stop, which is the correct response to a
/// consumer that has gone away.
unsafe extern "C" fn on_chunk(samples: *const f32, count: i32, user_data: *mut c_void) -> bool {
    if user_data.is_null() || samples.is_null() || count <= 0 || count > TTS_SAMPLE_RATE as i32 * 30
    {
        return false;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sink = &mut *(user_data as *mut ChunkSink<'_>);
        if sink.cancelled.load(Ordering::Acquire) {
            return false;
        }
        let slice = std::slice::from_raw_parts(samples, count as usize);
        if slice.iter().any(|sample| !sample.is_finite()) {
            return false;
        }
        // A consumer that returns false has stopped listening - the listener interrupted, or
        // shutdown began - and the rest of the sentence is abandoned rather than synthesised
        // into a buffer nobody will hear.
        if let Some(callback) = sink.callback.as_mut() {
            return callback(slice);
        }
        if sink.collected.len().saturating_add(slice.len()) > TTS_SAMPLE_RATE as usize * 180 {
            return false;
        }
        sink.collected.extend_from_slice(slice);
        true
    }))
    .unwrap_or(false)
}

/// Polled between chunks so an interruption stops synthesis promptly.
unsafe extern "C" fn on_cancel(user_data: *mut c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (*(user_data as *const AtomicBool)).load(Ordering::Relaxed)
    }))
    // A panic here means the flag cannot be read. Cancelling is the safe reading: worst case a
    // reply is cut short, where the alternative is synthesis that cannot be stopped.
    .unwrap_or(true)
}

struct Symbols {
    synthesize: FnSynthesize,
    audio_free: FnAudioFree,
    voice_ref_free: FnVoiceRefFree,
    free: FnFree,
    last_error: FnLastError,
    tts_defaults: FnTtsDefaults,
}

struct Inner {
    context: *mut QtContext,
    voice: QtVoiceRef,
}

// The context is only ever used behind the mutex below.
unsafe impl Send for Inner {}

pub struct TtsEngine {
    _library: libloading::Library,
    symbols: Symbols,
    inner: Mutex<Inner>,
    reference_text: Option<CString>,
}

pub trait Synthesizer: Send + Sync {
    fn synthesize_cancellable(
        &self,
        text: &str,
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), TtsError>;
}
impl Synthesizer for TtsEngine {
    fn synthesize_cancellable(
        &self,
        text: &str,
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), TtsError> {
        TtsEngine::synthesize_cancellable(self, text, cancelled, callback)
    }
}

unsafe impl Sync for TtsEngine {}

/// The talker weights to load, best first.
///
/// Q8 sounds better and costs about 320 MB more graphics memory than Q4, which is most of
/// what is spare on a 4 GB card. Whichever is installed wins, Q8 first, so trying the larger
/// one is a matter of putting the file in place - and backing it out is deleting it again.
/// The Q4 name is the fallback so a missing-file error names the file people actually have.
fn talker_file(model_dir: &Path) -> PathBuf {
    const BY_QUALITY: [&str; 2] = [
        "qwen-talker-0.6b-base-Q8_0.gguf",
        "qwen-talker-0.6b-base-Q4_K_M.gguf",
    ];
    best_of(model_dir, &BY_QUALITY)
}

/// The codec, which turns the talker's tokens back into a waveform.
///
/// Chosen the same way as the talker rather than named outright. It was hardcoded to the Q4
/// file, so dropping a higher-precision codec into the model directory changed nothing and
/// there was no way to tell from the outside that it had been ignored. The two models are
/// independent: the talker decides what is said, this decides how it sounds, and either can be
/// upgraded on its own if there is room for it.
fn codec_file(model_dir: &Path) -> PathBuf {
    const BY_QUALITY: [&str; 2] = [
        "qwen-tokenizer-12hz-Q8_0.gguf",
        "qwen-tokenizer-12hz-Q4_K_M.gguf",
    ];
    best_of(model_dir, &BY_QUALITY)
}

/// The first of `names` present, or the last as the name to complain about when none is.
fn best_of(model_dir: &Path, names: &[&str]) -> PathBuf {
    names
        .iter()
        .map(|name| model_dir.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| model_dir.join(names[names.len() - 1]))
}

impl TtsEngine {
    /// Loads the engine from a directory holding `qwen.dll`'s models and the reference voice.
    ///
    /// `library` is the path to `qwen.dll`, which lives in `lib/` rather than beside the models.
    pub fn load(library: impl AsRef<Path>, model_dir: impl AsRef<Path>) -> Result<Self, TtsError> {
        let library_path = library.as_ref().to_path_buf();
        let model_dir = model_dir.as_ref();
        let talker_path = talker_file(model_dir);
        let codec_path = codec_file(model_dir);
        let reference_wav = model_dir.join("user_ref_voice.wav");
        let reference_txt = model_dir.join("user_ref_text.txt");

        for required in [
            &library_path,
            &talker_path,
            &codec_path,
            &reference_wav,
            &reference_txt,
        ] {
            if !required.is_file() {
                return Err(TtsError::Missing(required.display().to_string()));
            }
        }

        // qwen.dll resolves ggml and CUDA by name, and they are split across two directories:
        // the ggml CPU pieces sit beside it in lib/, while the CUDA halves live in bin/ with
        // llama-server. Windows searches one directory at a time, so each dependency is loaded
        // by full path first - once resolved, later by-name lookups find it already in the
        // process. Without this, loading qwen.dll fails with an unhelpful LoadLibraryExW error
        // that names nothing.
        #[cfg(target_os = "windows")]
        preload_dependencies(&library_path);

        let library = unsafe { libloading::Library::new(&library_path) }.map_err(|error| {
            TtsError::Load(format!("{error} (loading {})", library_path.display()))
        })?;

        unsafe {
            let resolve = |name: &str| -> Result<*const (), TtsError> {
                library
                    .get::<*const ()>(name.as_bytes())
                    .map(|found| *found)
                    .map_err(|_| TtsError::Symbol(name.to_string()))
            };
            let init_defaults: FnInitDefaults =
                std::mem::transmute(resolve("qt_init_default_params")?);
            let tts_defaults: FnTtsDefaults =
                std::mem::transmute(resolve("qt_tts_default_params")?);
            let init: FnInit = std::mem::transmute(resolve("qt_init")?);
            let free: FnFree = std::mem::transmute(resolve("qt_free")?);
            let synthesize: FnSynthesize = std::mem::transmute(resolve("qt_synthesize")?);
            let audio_free: FnAudioFree = std::mem::transmute(resolve("qt_audio_free")?);
            let extract_ref: FnExtractRef = std::mem::transmute(resolve("qt_extract_voice_ref")?);
            let voice_ref_free: FnVoiceRefFree = std::mem::transmute(resolve("qt_voice_ref_free")?);
            let last_error: FnLastError = std::mem::transmute(resolve("qt_last_error")?);

            // Start from the library's own defaults rather than zeroing the struct: the ABI
            // version and sampling settings live there, and a zeroed struct silently asks for
            // greedy decoding at abi_version 0.
            let mut params = MaybeUninit::<QtInitParams>::zeroed();
            init_defaults(params.as_mut_ptr());
            let mut params = params.assume_init();

            let talker_c = to_c(&talker_path)?;
            let codec_c = to_c(&codec_path)?;
            params.talker_path = talker_c.as_ptr();
            params.codec_path = codec_c.as_ptr();
            params.use_fa = true;
            params.clamp_fp16 = false;
            params.max_batch = 1;

            let reference = read_reference_wav(&reference_wav)?;
            // Required, not optional. The transcript tells the model what the reference audio is
            // saying, and cloning is measurably worse without it - but nothing errors, so a
            // silently skipped file shows up only as a voice that sounds slightly wrong, with
            // nothing pointing at the cause.
            let transcript = std::fs::read_to_string(&reference_txt).map_err(|error| {
                TtsError::Reference(format!("{}: {error}", reference_txt.display()))
            })?;
            if transcript.trim().is_empty() {
                return Err(TtsError::Reference(format!(
                    "{} is empty; it must contain the words spoken in {}",
                    reference_txt.display(),
                    reference_wav.display()
                )));
            }
            let reference_text = Some(
                CString::new(transcript.trim())
                    .map_err(|error| TtsError::Reference(error.to_string()))?,
            );

            let context = init(&params);
            if context.is_null() {
                return Err(TtsError::Init(read_error(last_error)));
            }

            let mut voice = MaybeUninit::<QtVoiceRef>::zeroed().assume_init();
            let status = extract_ref(
                context,
                reference.as_ptr(),
                reference.len() as i32,
                &mut voice,
            );
            if status != QtStatus::Ok as i32 {
                free(context);
                return Err(TtsError::Reference(read_error(last_error)));
            }

            Ok(Self {
                _library: library,
                symbols: Symbols {
                    synthesize,
                    audio_free,
                    voice_ref_free,
                    free,
                    last_error,
                    tts_defaults,
                },
                inner: Mutex::new(Inner { context, voice }),
                reference_text,
            })
        }
    }

    /// Synthesises to a buffer.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, TtsError> {
        self.run(text, None, &AtomicBool::new(false))
    }

    /// A turn owns this cancellation flag. It is never reset by a later synthesis job.
    pub fn synthesize_cancellable(
        &self,
        text: &str,
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), TtsError> {
        self.run(text, Some(callback), cancelled).map(|_| ())
    }

    fn run<'a>(
        &self,
        text: &str,
        callback: Option<AudioCallback<'a>>,
        cancelled: &'a AtomicBool,
    ) -> Result<Vec<f32>, TtsError> {
        if cancelled.load(Ordering::Acquire) {
            return Err(TtsError::Cancelled);
        }
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let streaming = callback.is_some();
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let text_c = CString::new(text.replace('\0', ""))
            .map_err(|error| TtsError::Synthesis(error.to_string()))?;
        let mut sink = ChunkSink {
            callback,
            collected: Vec::new(),
            cancelled,
        };

        unsafe {
            let mut params = MaybeUninit::<QtTtsParams>::zeroed();
            (self.symbols.tts_defaults)(params.as_mut_ptr());
            let mut params = params.assume_init();

            params.text = text_c.as_ptr();
            params.ref_spk_emb = inner.voice.ref_spk_emb;
            params.ref_spk_dim = inner.voice.ref_spk_dim;
            // The residual-quantiser codes travel with the embedding. Passing reference text
            // without them is rejected: the text describes what the reference audio says, and
            // without the codes there is nothing for it to describe.
            params.ref_codes = inner.voice.ref_codes;
            params.ref_t = inner.voice.ref_t;
            if let Some(reference) = &self.reference_text {
                params.ref_text = reference.as_ptr();
            }
            params.max_new_tokens = frame_budget(text);
            // Keep codec sampling conservative. The library defaults are tuned for open-ended
            // generation; on short streamed phrases they occasionally produce unstable phonemes
            // and a smeared syllable. Lower temperature and nucleus sampling retain prosody while
            // removing those rare, audible outliers.
            params.do_sample = true;
            params.temperature = 0.7;
            params.top_k = 50;
            params.top_p = 0.9;
            params.repetition_penalty = 1.05;
            params.subtalker_do_sample = true;
            params.subtalker_temperature = 0.7;
            params.subtalker_top_k = 50;
            params.subtalker_top_p = 0.9;
            params.on_chunk = Some(on_chunk);
            params.on_chunk_user_data = &mut sink as *mut ChunkSink as *mut c_void;
            params.codec_chunk_sec = 0.25;
            params.codec_left_context_sec = 0.08;
            params.cancel = Some(on_cancel);
            params.cancel_user_data = cancelled as *const AtomicBool as *mut c_void;

            let mut audio = MaybeUninit::<QtAudio>::zeroed().assume_init();
            let status = (self.symbols.synthesize)(inner.context, &params, &mut audio);

            // The whole-buffer return is freed on every path. It is populated even when chunks
            // were streamed, and leaking it once per phrase adds up quickly.
            let collected = std::mem::take(&mut sink.collected);
            let whole = if !streaming
                && status == QtStatus::Ok as i32
                && !audio.samples.is_null()
                && audio.n_samples > 0
            {
                std::slice::from_raw_parts(audio.samples, audio.n_samples as usize).to_vec()
            } else {
                Vec::new()
            };
            (self.symbols.audio_free)(&mut audio);

            if cancelled.load(Ordering::Acquire) {
                return Err(TtsError::Cancelled);
            }
            if streaming {
                if status == QtStatus::Ok as i32 {
                    return Ok(Vec::new());
                }
                return Err(TtsError::Synthesis(read_error(self.symbols.last_error)));
            }
            if !collected.is_empty() {
                return Ok(collected);
            }
            if !whole.is_empty() {
                return Ok(whole);
            }
            Err(TtsError::Synthesis(read_error(self.symbols.last_error)))
        }
    }
}

impl Drop for TtsEngine {
    fn drop(&mut self) {
        // A panic mid-synthesis poisons the lock but leaves the context as valid as it was;
        // skipping the free then would leak the model's graphics memory.
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            (self.symbols.voice_ref_free)(&mut inner.voice);
            if !inner.context.is_null() {
                (self.symbols.free)(inner.context);
                inner.context = std::ptr::null_mut();
            }
        }
    }
}

/// Loads qwen.dll's dependencies by absolute path, in dependency order.
///
/// Failures are ignored deliberately: some builds need CUDA and some do not, and a DLL that is
/// genuinely required will surface as a clear error when qwen.dll itself fails to load.
#[cfg(target_os = "windows")]
fn preload_dependencies(library_path: &Path) {
    use std::os::windows::ffi::OsStrExt;

    let library_dir = library_path.parent().map(Path::to_path_buf);
    // bin/ is where the CUDA-enabled ggml lives, alongside llama-server.
    let sibling_bin = library_path
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("bin"));

    // The CUDA runtime is versioned, not built against anything here, so both the server and
    // synthesis can share one copy. It is half a gigabyte - cublasLt alone is 463 MB - and a
    // second copy beside qwen.dll would be paid for on every download for nothing. Loaded
    // first and by absolute path, so whatever imports it later binds to the one already in
    // the process, wherever that came from.
    const CUDA_RUNTIME: [&str; 3] = ["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"];
    // These must match each other and match qwen.dll, so they come from one directory.
    const GGML: [&str; 4] = ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"];

    // qwen.dll's own directory wins when it carries the CUDA backend as well as ggml itself.
    // That is a set shipped together and built against each other, and it must not be mixed
    // with a different llama.cpp build's ggml: the official Windows releases link no backend
    // into ggml.dll at all - they load them at runtime, which llama-server does and qwen.dll
    // does not - so borrowing their ggml leaves synthesis with no backend registered.
    //
    // Otherwise bin/, which is where a hand-built layout keeps the one CUDA ggml that the
    // server and synthesis were both compiled against and share.
    //
    // The whole set, not merely some of it. A directory holding one build's ggml.dll beside
    // another's ggml-cuda.dll registers no backend at all and synthesis silently drops to the
    // processor: measured here as 3.2 s of speech taking 16.7 s instead of 1.5 s. Requiring
    // every piece means whoever assembled the directory put a matching set there on purpose.
    let matched_set = library_dir.as_ref().filter(|p| {
        ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"]
            .iter()
            .all(|name| p.join(name).is_file())
    });

    let ggml_dir = matched_set
        .or(sibling_bin
            .as_ref()
            .filter(|p| p.join("ggml.dll").is_file()))
        .or(library_dir.as_ref());

    // Wherever the runtime actually is: beside qwen.dll if it was shipped that way, otherwise
    // bin/, which is where the server's copy lives.
    let cuda_dir = [library_dir.as_ref(), sibling_bin.as_ref()]
        .into_iter()
        .flatten()
        .find(|p| p.join(CUDA_RUNTIME[0]).is_file());

    for (directory, names) in [(cuda_dir, &CUDA_RUNTIME[..]), (ggml_dir, &GGML[..])] {
        let Some(directory) = directory else {
            continue;
        };
        let wide: Vec<u16> = directory
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        unsafe { SetDllDirectoryW(wide.as_ptr()) };
        for name in names {
            let candidate = directory.join(name);
            if !candidate.is_file() {
                continue;
            }
            let wide: Vec<u16> = candidate
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            unsafe { LoadLibraryW(wide.as_ptr()) };
        }
    }
}

#[cfg(target_os = "windows")]
extern "system" {
    fn SetDllDirectoryW(path: *const u16) -> i32;
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
}

fn to_c(path: &Path) -> Result<CString, TtsError> {
    CString::new(path.to_string_lossy().as_ref()).map_err(|error| TtsError::Load(error.to_string()))
}

unsafe fn read_error(last_error: FnLastError) -> String {
    let pointer = last_error();
    if pointer.is_null() {
        "no error reported".into()
    } else {
        CStr::from_ptr(pointer).to_string_lossy().into_owned()
    }
}

/// Loads the reference voice as 24 kHz mono float.
///
/// The speaker embedding and the codec's reference codes are both taken from this once at
/// startup, so its quality sets the character of every reply the assistant ever speaks - which
/// is why a clip at another rate goes through the same band-limited converter as everything
/// else, not a cheaper one.
fn read_reference_wav(path: &Path) -> Result<Vec<f32>, TtsError> {
    let mut reader =
        hound::WavReader::open(path).map_err(|error| TtsError::Reference(error.to_string()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|error| TtsError::Reference(error.to_string()))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|error| TtsError::Reference(error.to_string()))?
        }
    };

    let mono = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        samples
    };

    if spec.sample_rate == TTS_SAMPLE_RATE {
        return Ok(mono);
    }
    let converted = || -> Result<Vec<f32>, crate::audio::AudioError> {
        let mut converter = StreamResampler::new(spec.sample_rate, TTS_SAMPLE_RATE)?;
        let mut out = Vec::new();
        converter.push(&mono, &mut out)?;
        converter.finish(&mut out)?;
        Ok(out)
    };
    converted().map_err(|error| TtsError::Reference(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_reported_rather_than_panicking() {
        let outcome = TtsEngine::load(r"C:\zen-ai\lib\not-here.dll", r"C:\zen-ai\model\Qwen TTS");
        assert!(matches!(outcome, Err(TtsError::Missing(_))));
    }

    #[test]
    fn a_missing_model_directory_is_reported() {
        let outcome = TtsEngine::load(r"C:\zen-ai\lib\qwen.dll", r"C:\zen-ai\model\nope");
        assert!(matches!(outcome, Err(TtsError::Missing(_))));
    }

    #[test]
    fn the_better_talker_wins_when_both_are_installed() {
        let dir = std::env::temp_dir().join(format!("zen-talker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let q4 = dir.join("qwen-talker-0.6b-base-Q4_K_M.gguf");
        let q8 = dir.join("qwen-talker-0.6b-base-Q8_0.gguf");
        // Neither installed: the error should name the file most people have.
        assert_eq!(talker_file(&dir), q4);
        std::fs::write(&q4, b"gguf").unwrap();
        assert_eq!(talker_file(&dir), q4);
        std::fs::write(&q8, b"gguf").unwrap();
        assert_eq!(talker_file(&dir), q8, "Q8 must win when it is installed");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_reference_at_another_rate_is_read_at_the_codec_rate() {
        let dir = std::env::temp_dir().join(format!("zen-reference-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo-44k.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        // One second of a 440 Hz tone, the same in both channels.
        for n in 0..44_100 {
            let value = (f32::sin(n as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44_100.0)
                * 16_000.0) as i16;
            writer.write_sample(value).unwrap();
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        let samples = read_reference_wav(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(samples.len(), 24_000, "the duration is kept");
        // Away from the edges the tone comes through at its own level.
        let middle = &samples[6_000..18_000];
        let rms = (middle.iter().map(|s| s * s).sum::<f32>() / middle.len() as f32).sqrt();
        let expected = 16_000.0 / 32_768.0 / std::f32::consts::SQRT_2;
        assert!(
            (rms - expected).abs() < 0.01,
            "rms {rms}, expected {expected}"
        );
    }

    #[test]
    fn a_phrase_of_long_words_gets_the_time_it_takes_to_say() {
        let short_words = "I am not sure it is so. ".repeat(10);
        let long_words = "Internationalisation considerations notwithstanding. ".repeat(4);
        // Twelve words take far longer to say than sixty short ones - and the budget knows it.
        assert!(frame_budget(&long_words) > frame_budget(&short_words) / 2);
        let slowest_frames =
            long_words.trim().chars().count() * SLOWEST_MS_PER_CHAR / CODEC_FRAME_MS;
        assert!(frame_budget(&long_words) as usize >= slowest_frames);
        // A bare word still gets the floor, and nothing runs away.
        assert_eq!(frame_budget("Hmm."), 100);
        assert_eq!(frame_budget(&"word ".repeat(10_000)), 1_500);
    }

    /// Where the reference voice and its transcript live.
    fn reference_dir() -> &'static Path {
        Path::new(r"C:\zen-ai\model\Qwen TTS")
    }

    #[test]
    fn both_halves_of_the_voice_reference_are_present() {
        // Synthesis still runs with only the audio, just worse - so a missing transcript would
        // surface as a voice that sounds slightly off, with nothing pointing at the cause.
        if !reference_dir().is_dir() {
            return;
        }
        assert!(
            reference_dir().join("user_ref_voice.wav").is_file(),
            "the reference voice is what the assistant sounds like"
        );
        let transcript = reference_dir().join("user_ref_text.txt");
        assert!(transcript.is_file());
        let text = std::fs::read_to_string(&transcript).expect("readable");
        assert!(
            !text.trim().is_empty(),
            "the transcript must say what the reference audio says"
        );
    }

    #[test]
    fn the_reference_voice_is_long_enough_to_clone_from() {
        // A speaker embedding taken from a very short clip is unstable, and every reply the
        // assistant ever speaks is built on it.
        let wav = reference_dir().join("user_ref_voice.wav");
        if !wav.is_file() {
            return;
        }
        let samples = read_reference_wav(&wav).expect("reference reads");
        let seconds = samples.len() as f32 / TTS_SAMPLE_RATE as f32;
        assert!(
            seconds >= 3.0,
            "reference voice is only {seconds:.1}s; embeddings from clips this short are unstable"
        );
    }

    #[test]
    fn synthesis_and_capture_run_at_different_rates() {
        // Mixing these up is silent: audio still plays, at the wrong pitch and speed.
        assert_ne!(TTS_SAMPLE_RATE, crate::audio::SAMPLE_RATE);
    }
}
