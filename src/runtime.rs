// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Process entry: argument handling, the native worker branch, and launching the app.
//!
//! Zen is one executable with three ways in. Re-executed as `--native-worker` it hosts an
//! isolated ASR or TTS library. Given `--self-test` it exercises the native path and exits.
//! Otherwise it opens the desktop application, whose embedded webview owns capture and
//! playback while this process owns VAD, recognition, generation and synthesis.
use crate::{
    asr::Recognizer,
    audio::Loudness,
    bridge::{validate_prompt, EngineOptions},
    engine::{LlamaConfig, LlamaEngine, SlotKind},
    resample::StreamResampler,
    tts::{Synthesizer, TTS_SAMPLE_RATE},
};
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

type Error = Box<dyn std::error::Error + Send + Sync>;

/// The talker prompt is embedded so a stock install has a voice without carrying a file
/// beside the binary, and so the string is byte-identical every run, which is what lets
/// llama-server reuse its cached prompt instead of reprocessing it.
const PERSONA: &str = crate::bridge::DEFAULT_PERSONA;

#[derive(Debug)]
struct Options {
    root: PathBuf,
    filter: bool,
    self_test: bool,
    run_for: Option<Duration>,
    gain: f32,
    reply_tokens: usize,
    system_prompt: String,
}

/// The installation root: the directory holding `bin`, `lib` and `model`.
///
/// Found relative to the executable so the whole install can be copied or moved and still
/// start from a double click. `zen.exe` beside those directories is the shipped layout; the
/// walk upwards is what lets a freshly built `target/release/zen.exe` run in place - and the
/// measurement tools in `examples/` find the models the same way.
pub fn discover_root() -> PathBuf {
    if let Some(root) = std::env::var_os("ZEN_ROOT") {
        return PathBuf::from(root);
    }
    let exe = std::env::current_exe().ok();
    root_near(exe.as_deref().and_then(Path::parent), |dir| {
        ["bin", "lib", "model"]
            .iter()
            .all(|name| dir.join(name).is_dir())
    })
}

/// The first directory at or above `start` that holds a complete install.
///
/// With none, `start` itself. What is missing is then reported inside the folder Zen was run
/// from, which is where anyone would go to fix it. This used to fall back to the folder on the
/// machine Zen was developed on, so an install anywhere else failed with paths its owner had
/// never seen.
fn root_near(start: Option<&Path>, complete: impl Fn(&Path) -> bool) -> PathBuf {
    let mut dir = start;
    // Six levels covers `target/<profile>/` and a deps directory with room to spare.
    for _ in 0..6 {
        let Some(candidate) = dir else { break };
        if complete(candidate) {
            return candidate.to_path_buf();
        }
        dir = candidate.parent();
    }
    start.map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

impl Default for Options {
    fn default() -> Self {
        Self {
            root: discover_root(),
            filter: true,
            self_test: false,
            run_for: None,
            gain: Loudness::DEFAULT_GAIN,
            reply_tokens: 512,
            system_prompt: PERSONA.to_string(),
        }
    }
}

impl From<Options> for EngineOptions {
    fn from(o: Options) -> Self {
        Self {
            root: o.root,
            system_prompt: crate::bridge::compose_prompt(&o.system_prompt),
            reply_tokens: o.reply_tokens,
            filter: o.filter,
            gain: o.gain,
            run_for: o.run_for,
        }
    }
}

/// Printed by `--version`, so a downloaded binary can say which one it is.
const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = r"Zen local voice assistant

  zen [OPTIONS]                  Open the Zen window
  zen --self-test                Exercise native ASR, LLM, TTS and cancellation
  zen --version                  Show the version
  zen --help                     Show this message

  --root PATH                    Installation root holding bin, lib and model
                                 (found next to the executable by default)
  --system-prompt TEXT           Replace the persona. The voice rules are kept
  --system-prompt-file PATH      Read instructions from a UTF-8 file
  --reply-tokens N               Reply budget, 64..1024 (default 512)
  --gain N                       Output gain, 0.5..4.0 (default 2.0)
  --no-filter                    Skip transcript repair
  --run-for-seconds N            Stop after N seconds

Instructions can also be set in Settings, which keeps them on this computer and uses them
from the next conversation. A prompt given on the command line may stay in your shell's
history.
";

fn parse(args: impl Iterator<Item = String>) -> Result<Option<Options>, String> {
    let mut args = args;
    let mut o = Options::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--no-filter" => o.filter = false,
            "--self-test" => o.self_test = true,
            "--system-prompt" => o.system_prompt = args.next().ok_or("missing system prompt")?,
            "--system-prompt-file" => {
                let path = args.next().ok_or("missing prompt file path")?;
                if std::fs::metadata(&path)
                    .map_err(|_| "cannot read prompt file")?
                    .len()
                    > 8192
                {
                    return Err("prompt file exceeds 8192 bytes".into());
                }
                o.system_prompt =
                    std::fs::read_to_string(path).map_err(|_| "cannot read UTF-8 prompt file")?;
            }
            "--gain" => {
                let n = args
                    .next()
                    .ok_or("missing gain")?
                    .parse::<f32>()
                    .map_err(|_| "invalid gain")?;
                if !(Loudness::MIN_GAIN..=Loudness::MAX_GAIN).contains(&n) {
                    return Err(format!(
                        "gain must be {}..{}",
                        Loudness::MIN_GAIN,
                        Loudness::MAX_GAIN
                    ));
                }
                o.gain = n;
            }
            "--reply-tokens" => {
                let n = args
                    .next()
                    .ok_or("missing token count")?
                    .parse::<usize>()
                    .map_err(|_| "invalid token count")?;
                if !(64..=1024).contains(&n) {
                    return Err("reply tokens must be 64..1024".into());
                }
                o.reply_tokens = n;
            }
            "--root" => o.root = PathBuf::from(args.next().ok_or("--root needs a path")?),
            "--run-for-seconds" => {
                let n = args
                    .next()
                    .ok_or("missing duration")?
                    .parse::<u64>()
                    .map_err(|_| "invalid duration")?;
                if n == 0 {
                    return Err("duration must be positive".into());
                }
                o.run_for = Some(Duration::from_secs(n));
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    validate_prompt(&o.system_prompt)?;
    Ok(Some(o))
}

/// Report a fatal error to whoever started Zen.
///
/// The release build is a windowed binary, so a double click has nowhere for `eprintln!` to
/// go and a failure to start would simply look like nothing happening. When there is no
/// standard error to write to, say it in a dialog instead.
fn fatal(message: &str) -> ExitCode {
    eprintln!("Zen stopped: {message}");
    #[cfg(windows)]
    if !has_stderr() {
        show_error_dialog(message);
    }
    ExitCode::FAILURE
}

/// Whether this process inherited a usable standard error, as it does from a terminal but
/// not from Explorer.
#[cfg(windows)]
fn has_stderr() -> bool {
    use std::os::windows::io::AsRawHandle;
    let handle = std::io::stderr().as_raw_handle();
    !handle.is_null() && handle as isize != -1
}

#[cfg(windows)]
fn show_error_dialog(message: &str) {
    use windows::core::PCWSTR;
    use windows::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, MB_ICONERROR, MB_OK, MB_SETFOREGROUND,
    };
    let wide = |text: &str| {
        text.encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<u16>>()
    };
    let body = wide(message);
    let title = wide("Zen could not start");
    unsafe {
        MessageBoxW(
            None,
            PCWSTR(body.as_ptr()),
            PCWSTR(title.as_ptr()),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
        );
    }
}

/// The window's event loop must own the main thread, so the async runtime is built here
/// and handed to Tauri rather than wrapping this function in `#[tokio::main]`.
pub fn main_entry() -> ExitCode {
    let mut internal = std::env::args().skip(1);
    if internal.next().as_deref() == Some("--native-worker") {
        return match crate::native::worker_entry(&internal.next().unwrap_or_default()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Native worker: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        println!("zen {VERSION}");
        return ExitCode::SUCCESS;
    }
    let options = match parse(std::env::args().skip(1)) {
        Ok(Some(o)) => o,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => return fatal(&e.to_string()),
    };
    tauri::async_runtime::set(runtime.handle().clone());

    let result = if options.self_test {
        runtime.block_on(run_self_test(options))
    } else {
        crate::app::run(options.into())
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fatal(&e.to_string()),
    }
}

/// Load the real engines, prove the native path end to end, and stop. No audio device is
/// opened and no window appears, so this runs unattended.
async fn run_self_test(o: Options) -> Result<(), Error> {
    println!("Starting Zen from {}", o.root.display());
    let llama = Arc::new(LlamaEngine::new(LlamaConfig::from_zen_root(&o.root))?);
    if let Err(e) = llama.start().await {
        let _ = llama.stop().await;
        return Err(e.into());
    }
    println!("Loading isolated speech recognition...");
    let asr: Arc<dyn Recognizer> = match crate::native::NativeEngine::load("asr", &o.root) {
        Ok(engine) => Arc::new(engine),
        Err(error) => {
            let _ = llama.stop().await;
            return Err(error.into());
        }
    };
    println!("Loading isolated speech synthesis...");
    let voice: Arc<dyn Synthesizer> = match crate::native::NativeEngine::load("tts", &o.root) {
        Ok(engine) => Arc::new(engine),
        Err(error) => {
            let _ = llama.stop().await;
            return Err(error.into());
        }
    };
    let result = self_test(&llama, asr.as_ref(), voice.as_ref()).await;
    let _ = llama.stop().await;
    result
}

async fn self_test(
    llama: &LlamaEngine,
    asr: &dyn Recognizer,
    voice: &dyn Synthesizer,
) -> Result<(), Error> {
    let start = Instant::now();
    let mut audio = Vec::new();
    let mut first = None;
    let mut previous_chunk = None;
    let mut max_chunk_gap = Duration::ZERO;
    let mut required_buffer = Duration::ZERO;
    voice.synthesize_cancellable(
        "Hello. My name is Zen. I am ready to help you.",
        &AtomicBool::new(false),
        &mut |chunk| {
            let arrived = start.elapsed();
            let first_arrival = *first.get_or_insert(arrived);
            if let Some(previous) = previous_chunk.replace(arrived) {
                max_chunk_gap = max_chunk_gap.max(arrived - previous);
            }
            let delivered =
                Duration::from_secs_f64(audio.len() as f64 / f64::from(TTS_SAMPLE_RATE));
            required_buffer =
                required_buffer.max((arrived - first_arrival).saturating_sub(delivered));
            audio.extend_from_slice(chunk);
            true
        },
    )?;
    if audio.is_empty() {
        return Err("TTS self-test produced no audio".into());
    }
    println!(
        "TTS: first chunk {} ms, {:.2} s audio, {:.2} s wall",
        first.unwrap().as_millis(),
        audio.len() as f64 / f64::from(TTS_SAMPLE_RATE),
        start.elapsed().as_secs_f64()
    );
    println!(
        "TTS delivery: largest chunk gap {} ms, startup buffer needed {} ms",
        max_chunk_gap.as_millis(),
        required_buffer.as_millis()
    );
    let mut converter = StreamResampler::new(TTS_SAMPLE_RATE, crate::audio::SAMPLE_RATE)?;
    let mut capture = Vec::new();
    converter.push(&audio, &mut capture)?;
    converter.finish(&mut capture)?;
    let transcript = asr.transcribe(&capture)?;
    let recognized = transcript.text.to_lowercase();
    if !recognized.contains("name") || !recognized.contains("help") {
        return Err(format!(
            "ASR self-test did not recognize the test phrase: {}",
            transcript.text
        )
        .into());
    }
    println!(
        "ASR: {:.0} ms, transcript: {}",
        transcript.latency_ms, transcript.text
    );
    // The fact under test has to be one the persona does not already answer. Asking the
    // assistant the user's name cannot work: the talker prompt states it outright, so the
    // model answers from the system prompt and the test reads a correct reply as a broken
    // engine. Anything the prompt is silent about proves the earlier turns actually arrived.
    let messages = vec![
        ("system", crate::bridge::compose_prompt(PERSONA)),
        ("user", "I left my keys in the blue drawer.".into()),
        ("assistant", "The blue drawer. Noted.".into()),
        ("user", "Which drawer did I say? Answer briefly.".into()),
    ];
    let mut streamed = String::new();
    let response = llama
        .client()
        .stream_completion_with(SlotKind::Talker, &messages, 64, 0.1, |token| {
            streamed.push_str(&token);
            true
        })
        .await?
        .text;
    if response != streamed {
        return Err(format!(
            "streamed tokens did not match the completed reply: {streamed:?} then {response:?}"
        )
        .into());
    }
    if !response.to_lowercase().contains("blue") {
        return Err(format!("the model did not see the conversation history: {response}").into());
    }
    println!("Model: {response}");

    // The talker slot, on the shipped persona. Only the rules with a mechanical shape are
    // asserted - digits, opening filler, a closing offer of more help, a claim to have done
    // something it cannot do. Whether a reply is *warm* is not testable and is not tested;
    // what is testable is that the instruction reached the model at all, which is the thing
    // that silently stops being true when the prompt is edited.
    let spoken = talker_reply(
        llama,
        "How many days does February have in a leap year?",
        160,
    )
    .await?;
    let lowered = spoken.to_lowercase().replace('\u{2019}', "'");
    for opener in [
        "certainly",
        "sure",
        "great question",
        "of course",
        "absolutely",
        "arya",
    ] {
        if lowered.trim_start().starts_with(opener) {
            return Err(format!("the reply opened with filler: {spoken}").into());
        }
    }
    if spoken.chars().any(|c| c.is_ascii_digit()) {
        return Err(format!("the reply used digits where a speaker says words: {spoken}").into());
    }
    for trailer in ["let me know", "anything else", "feel free"] {
        if lowered.contains(trailer) {
            return Err(format!("the reply closed by offering more help: {spoken}").into());
        }
    }
    println!("Talker: {spoken}");

    // The assistant has no tools. Saying so is fine; implying it acted is the failure, and it
    // is the one a listener cannot detect - nothing happens either way.
    let refusal = talker_reply(llama, "Set a timer for ten minutes.", 160).await?;
    let lowered = refusal.to_lowercase().replace('\u{2019}', "'");
    if [
        "i've set",
        "i have set",
        "i set a",
        "timer is set",
        "setting a timer",
    ]
    .iter()
    .any(|claim| lowered.contains(claim))
    {
        return Err(format!("the assistant implied it set a timer: {refusal}").into());
    }
    if ![
        "can't",
        "cannot",
        "can not",
        "not able",
        "don't have",
        "do not have",
        "no way",
    ]
    .iter()
    .any(|admission| lowered.contains(admission))
    {
        return Err(format!("the assistant did not say it cannot do that: {refusal}").into());
    }
    println!("Talker: {refusal}");

    // Brevity must lose to an explicit request. Asked for every month, the persona this
    // replaced answered that it depends and then declined to list them, which reads as rude
    // and leaves the question unanswered.
    let listed = talker_reply(
        llama,
        "How many days does each month have? Tell me all twelve.",
        400,
    )
    .await?;
    let lowered = listed.to_lowercase();
    if let Some(missing) = ["january", "february", "june", "december"]
        .iter()
        .find(|month| !lowered.contains(**month))
    {
        return Err(
            format!("the reply left out {missing} when asked for all twelve: {listed}").into(),
        );
    }
    println!("Talker listed every month, {} characters", listed.len());

    // The filter slot, on the shipped instruction and through the same parser and guard the
    // session uses. Its failure mode is silent from anywhere else: it answers the question, or
    // loses part of it, instead of copying it down. That does not show up in an offline test,
    // because it is not a property of the code but of this model reading this prompt. (A
    // transcript with nothing in it - "uh um uh" - ends the turn before the filter is asked.)
    for (transcript, words) in [
        ("wut is the wether tooday", &["weather"][..]),
        ("can you turn on the kitchen lights", &["kitchen", "light"][..]),
        // Longer than the flat 128-token budget the filter used to be given. Past that the
        // `CLEAN:` line ran out partway through and the rest of the question was simply
        // gone - accepted downstream, because every word still in it had genuinely been
        // said. The last words are what this checks, since truncation only ever loses those.
        (
            "so i was going through the notes from the meeting yesterday and there were a couple of things i wanted to check with you about the schedule for next month because the room booking looked like it might overlap with the other team and i could not tell from the calendar whether that had been sorted out already or whether somebody still needs to call the front desk about it before friday",
            &["front desk", "friday"][..],
        ),
    ] {
        let verdict = filter_once(llama, transcript).await?;
        let text = crate::reply::accept_verdict(transcript, verdict);
        let lowered = text.to_lowercase();
        if let Some(missing) = words.iter().find(|word| !lowered.contains(**word)) {
            return Err(format!("the filter lost {missing:?} out of {transcript:?}: {text}").into());
        }
        println!("Filter: {transcript:?} -> {text}");
    }

    let cancelled = AtomicBool::new(false);
    let mut cancellation_started = None;
    let result = voice.synthesize_cancellable(
        "This reply should stop when interrupted. The remaining words must never reach playback.",
        &cancelled,
        &mut |_| {
            cancellation_started = Some(Instant::now());
            cancelled.store(true, Ordering::Release);
            false
        },
    );
    if !matches!(result, Err(crate::tts::TtsError::Cancelled)) || cancellation_started.is_none() {
        return Err("native synthesis did not acknowledge cancellation".into());
    }
    println!(
        "TTS cancellation acknowledged in {} ms",
        cancellation_started.unwrap().elapsed().as_millis()
    );
    let mut resumed_samples = 0;
    voice.synthesize_cancellable(
        "I am listening again.",
        &AtomicBool::new(false),
        &mut |chunk| {
            resumed_samples += chunk.len();
            true
        },
    )?;
    if resumed_samples == 0 {
        return Err("native synthesis failed to resume after cancellation".into());
    }
    println!("Self-test passed, including synthesis after interruption. Speaker acoustics and audible interruption timing require a live session.");
    Ok(())
}

/// One reply from the talker slot on the shipped persona, with no history behind it.
async fn talker_reply(
    llama: &LlamaEngine,
    said: &str,
    tokens: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let messages = vec![
        ("system", crate::bridge::compose_prompt(PERSONA)),
        ("user", said.to_string()),
    ];
    Ok(llama
        .client()
        .stream_completion_with(SlotKind::Talker, &messages, tokens, 0.7, |_| true)
        .await?
        .text
        .trim()
        .to_string())
}

/// One filter request, exactly as a live turn makes it.
async fn filter_once(
    llama: &LlamaEngine,
    transcript: &str,
) -> Result<crate::reply::FilterVerdict, Box<dyn std::error::Error + Send + Sync>> {
    let messages = vec![
        ("system", crate::bridge::FILTER_RULES.to_string()),
        (
            "user",
            serde_json::json!({ "transcript": transcript }).to_string(),
        ),
    ];
    let reply = llama
        .client()
        .stream_completion_with(
            SlotKind::Filter,
            &messages,
            crate::remote::filter_budget(transcript),
            0.1,
            |_| true,
        )
        .await?;
    if reply.truncated {
        return Err(format!("the filter ran out of budget on {transcript:?}").into());
    }
    Ok(crate::reply::parse_filter_verdict(&reply.text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(args: &[&str]) -> Result<Option<Options>, String> {
        parse(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn removed_network_options_are_rejected_rather_than_ignored() {
        for arg in [
            "--serve",
            "--listen",
            "--tls-cert",
            "--tls-key",
            "--no-earcons",
        ] {
            let error = options(&[arg]).expect_err("must not be silently accepted");
            assert!(error.contains("unknown argument"), "{arg}: {error}");
        }
    }

    #[test]
    fn bounded_options_hold_their_ranges() {
        // The pause is learned, never set by hand.
        assert!(options(&["--endpoint-ms", "720"]).is_err());
        assert!(options(&["--reply-tokens", "63"]).is_err());
        assert!(options(&["--reply-tokens", "1025"]).is_err());
        assert!(options(&["--run-for-seconds", "0"]).is_err());
    }

    #[test]
    fn the_install_is_found_above_the_executable_or_reported_where_it_is() {
        let exe_dir = Path::new(r"D:\Apps\Zen\target\release");
        let found = root_near(Some(exe_dir), |dir| dir == Path::new(r"D:\Apps\Zen"));
        assert_eq!(found, Path::new(r"D:\Apps\Zen"));
        // Nothing complete anywhere: the folder Zen is in, never one from another machine.
        assert_eq!(root_near(Some(exe_dir), |_| false), exe_dir);
        assert_eq!(root_near(None, |_| false), Path::new("."));
    }

    #[test]
    fn an_empty_system_prompt_is_refused() {
        assert!(options(&["--system-prompt", "   "]).is_err());
        assert!(options(&["--system-prompt", "Be brief."]).is_ok());
    }

    #[test]
    fn the_root_is_found_beside_the_executable_and_can_still_be_overridden() {
        let temp = std::env::temp_dir().join(format!("zen-root-{}", std::process::id()));
        let nested = temp.join("target").join("release");
        for name in ["bin", "lib", "model"] {
            std::fs::create_dir_all(temp.join(name)).unwrap();
        }
        std::fs::create_dir_all(&nested).unwrap();
        let complete = |dir: &Path| {
            ["bin", "lib", "model"]
                .iter()
                .all(|name| dir.join(name).is_dir())
        };
        // The walk upwards is what lets a build output directory find the install it sits in.
        assert!(!complete(&nested));
        assert!(nested.ancestors().any(complete));
        assert_eq!(
            nested.ancestors().find(|dir| complete(dir)).unwrap(),
            temp.as_path()
        );
        // An explicit root always wins over discovery.
        assert_eq!(
            options(&["--root", r"D:\elsewhere"]).unwrap().unwrap().root,
            PathBuf::from(r"D:\elsewhere")
        );
        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn the_default_prompt_is_embedded_and_valid() {
        let o = options(&[]).unwrap().unwrap();
        assert_eq!(o.system_prompt, PERSONA);
        assert!(validate_prompt(&o.system_prompt).is_ok());
    }
}
