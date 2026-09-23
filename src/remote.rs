// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Remote audio adapter for the shared Session state machine.
//! The browser owns echo cancellation and acknowledges playback, never generated text.
use crate::{
    asr::{AsrOutcome, AsrWorker, Recognizer},
    audio::{
        CaptureEvent, CapturePipeline, ChunkConfig, EndpointPolicy, Loudness, SegmenterConfig,
    },
    bridge::{Control, EngineOptions, Error, Input, Transport},
    conversation::{Conversation, WindowBudget},
    engine::{LlamaConfig, LlamaEngine, SlotKind},
    input::Utterance,
    reply::{heard_part, pause_after_ms, ChunkLimits},
    session::{Generation, Session, Task},
    tts::{Synthesizer, TTS_SAMPLE_RATE},
    turn::{Phase, TurnTimeouts},
    voice::{VoiceEvent, VoiceWorker},
};
use futures_util::FutureExt;
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

enum ModelEvent {
    Filter(Generation, u64, String),
    /// The filter ran out of token budget partway through its line.
    FilterTruncated(Generation, u64),
    FilterFailed(Generation, u64),
    Token(Generation, String),
    Done(Generation),
    Failed(Generation),
}
struct Phrase {
    id: u64,
    text: String,
    /// Synthesis has finished, so `sent` is the phrase's full length.
    ended: bool,
    /// Samples sent to the page, and how many of them it has confirmed playing.
    sent: usize,
    heard: usize,
    /// Length of the prefix already reported as heard.
    credited: usize,
}
struct AudioCredit {
    sequence: u64,
    phrase: u64,
    samples: usize,
}

/// Where the time went between the end of a question and the first sound of its answer.
/// Reported once per turn, when that sound starts, so a slow reply can be traced to its stage.
#[derive(Default)]
struct TurnTiming {
    generation: Option<Generation>,
    ended: Option<Instant>,
    transcribed: Option<Instant>,
    requested: Option<Instant>,
    first_word: Option<Instant>,
    /// Longest any piece of the question waited for the recogniser to be free.
    queue_ms: u64,
    /// The question was typed, so there was nothing to recognise.
    typed: bool,
}

impl TurnTiming {
    /// The speaker has finished. Stages timed against an earlier endpoint of the same question -
    /// before they carried on talking - no longer apply.
    fn ended(&mut self, generation: Generation) {
        *self = Self {
            generation: Some(generation),
            ended: Some(Instant::now()),
            queue_ms: self.queue_ms,
            ..Self::default()
        };
    }

    fn mark(stage: &mut Option<Instant>) {
        stage.get_or_insert_with(Instant::now);
    }

    fn report(&mut self, generation: Generation) -> Option<serde_json::Value> {
        if self.generation != Some(generation) {
            return None;
        }
        let ended = self.ended.take()?;
        let now = Instant::now();
        let ms = |from: Option<Instant>, to: Option<Instant>| match (from, to) {
            (Some(from), Some(to)) => json!(to.saturating_duration_since(from).as_millis() as u64),
            _ => serde_json::Value::Null,
        };
        Some(json!({
            "type": "timing",
            "generation": generation.value(),
            "total_ms": now.saturating_duration_since(ended).as_millis() as u64,
            "recognize_ms": if self.typed {
                serde_json::Value::Null
            } else {
                ms(Some(ended), self.transcribed)
            },
            "queue_ms": self.queue_ms,
            // Typed words are taken as written, so neither stage ran.
            "repair_ms": if self.typed {
                serde_json::Value::Null
            } else {
                ms(self.transcribed, self.requested)
            },
            "think_ms": ms(self.requested, self.first_word),
            "speak_ms": ms(self.first_word, Some(now)),
        }))
    }
}

/// Tokens the filter may spend repairing one transcript.
///
/// It has to write the whole transcript back out, so a fixed budget is really a limit on how
/// much anyone may say in one breath. A turn holds up to three minutes of speech; at the flat
/// 128 this used to be, the `CLEAN:` line ran out partway through anything over roughly forty
/// seconds, and the remainder of the question was simply gone - accepted, because every word
/// still in it had genuinely been said.
///
/// Sized from characters rather than words: a transcript in a script written without spaces is
/// one "word" however long it is, and budgeting from that would cap a Chinese question at the
/// floor. The ceiling is a latency guard rather than a correctness one - past it the turn is
/// answered from the raw transcript, which is unpolished but complete.
pub(crate) fn filter_budget(raw: &str) -> usize {
    (raw.chars().count() + 64).clamp(128, 512)
}

/// The local weekday and hour. The user's, not UTC: a greeting that calls midnight morning is
/// worse than none.
#[cfg(windows)]
fn local_clock() -> Option<(u16, u16)> {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    let now = unsafe { GetLocalTime() };
    Some((now.wDayOfWeek, now.wHour))
}

#[cfg(not(windows))]
fn local_clock() -> Option<(u16, u16)> {
    None
}

/// When the session is starting, the way a person would put it: "a Friday evening".
fn moment() -> String {
    let Some((day, hour)) = local_clock() else {
        return "today".into();
    };
    let day = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    .get(day as usize)
    .copied()
    .unwrap_or("today");
    match hour {
        5..=11 => format!("a {day} morning"),
        12..=16 => format!("a {day} afternoon"),
        17..=21 => format!("a {day} evening"),
        _ => format!("late on a {day} night"),
    }
}

/// The opening line when the model cannot give one: fixed, and chosen from the clock.
fn greeting_line() -> String {
    match local_clock().map(|(_, hour)| hour) {
        Some(5..=11) => "Good morning. What's on your mind?",
        Some(12..=16) => "Good afternoon. What's on your mind?",
        Some(17..=21) => "Good evening. What's on your mind?",
        _ => "Hello. What's on your mind?",
    }
    .to_string()
}

/// How long the opening line may take to write before the fixed one is said instead. Measured:
/// a line of this length takes a few hundred milliseconds once the model is loaded.
const GREETING_BUDGET: Duration = Duration::from_secs(4);

/// What Zen opens with, asked of the model in its own voice, so the session does not start with
/// the same sentence every time.
///
/// A session that begins in silence gives no sign that anything is listening. The line is said
/// through the same phrase path a reply takes, so it answers to the same generation fence and
/// typing or the stop button cut it short. Speech does not: nothing is listened to until it has
/// been said (see `RemoteRunner::opening`). It is not recorded in the conversation - nothing was
/// asked - and the request is the conversation's own system prompt with one instruction after
/// it, so the prompt it leaves cached is the one the first real turn begins with.
///
/// The wording was measured: asked to "ask what is on his mind", every line came back as that
/// phrase; asked for a question of its own and to make it different each time, 24 of 24 were
/// distinct and all one or two short sentences, at temperature 0.9.
fn request_greeting(llama: Arc<LlamaEngine>, system: String) -> tokio::task::JoinHandle<String> {
    let instruction = format!(
        "(Arya has just opened Zen. It is {}. Say hello the way you would, in your own words: \
         one short, warm sentence and one easy question to start him talking. Make it \
         different each time, and do not ask what is on his mind. Only the greeting.)",
        moment()
    );
    tokio::spawn(async move {
        let messages = [("system", system), ("user", instruction)];
        let request =
            llama
                .client()
                .stream_completion_with(SlotKind::Talker, &messages, 48, 0.9, |_| true);
        match tokio::time::timeout(GREETING_BUDGET, request).await {
            Ok(Ok(reply)) => usable_greeting(&reply.text).unwrap_or_else(greeting_line),
            _ => greeting_line(),
        }
    })
}

/// The model's opening line, if it is one: a sentence or two of plain speech. Anything longer,
/// empty, or cut off by the token limit is not said.
fn usable_greeting(text: &str) -> Option<String> {
    let text = crate::reply::to_speakable(text.trim().trim_matches('"'));
    let ends = text.ends_with(['.', '?', '!']);
    (ends && (8..=200).contains(&text.len())).then_some(text)
}

/// How long nobody may speak or type, with Zen saying nothing either, before the page is told the
/// session has gone quiet. The page turns the microphone off then, unless the listener has asked
/// it not to.
const QUIET_AFTER: Duration = Duration::from_secs(120);

/// How long Zen's voice can still come back from the room after the last of it has played: the
/// longer end of the 0.3-0.6 s reverberation time of an ordinary room.
const ROOM_TAIL: Duration = Duration::from_millis(600);

/// Whether what the microphone hears is listened to yet. See `RemoteRunner::opening`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Opening {
    /// The greeting is being said.
    Greeting,
    /// The greeting is over, or was cut short; listening starts at this instant.
    Until(Instant),
    Open,
}

/// How long recognition may take for the audio just captured.
///
/// A fixed budget is really a cap on how long anyone may talk: say four sentences and the turn
/// is thrown away for being slow, having already been heard. Sized from the audio instead, and
/// generously - this guards against a recogniser that has stopped making progress, it is not a
/// performance target.
///
/// Measured on the reference machine, recognition costs about 300 ms plus 630 ms for every
/// second of speech: 1.0 s of audio in 763 ms, 7.3 s in 3.8 s, 33.9 s in 21.3 s. Most of that
/// is already paid while the speaker is still talking, since each pause hands off a piece, so
/// a full second of budget per second of audio leaves real headroom for the tail.
fn transcribe_budget_ms(audio_ms: usize) -> u64 {
    const FIXED_MS: u64 = 3_000;
    const PER_SECOND_MS: u64 = 1_000;
    let budget = FIXED_MS + (audio_ms as u64) * PER_SECOND_MS / 1_000;
    budget.clamp(5_000, 60_000)
}

pub(crate) async fn run(
    options: &EngineOptions,
    prompt: &str,
    transport: Transport,
    input: mpsc::Receiver<(u64, Input)>,
) -> Result<(), Error> {
    let llama = Arc::new(LlamaEngine::new(LlamaConfig::from_zen_root(&options.root))?);
    let result =
        std::panic::AssertUnwindSafe(run_owned(options, prompt, transport, input, llama.clone()))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err("session worker panicked".into()));
    // A new device must not inherit KV or native speech state. Stop the model before unlocking.
    llama.stop().await?;
    result
}

async fn run_owned(
    options: &EngineOptions,
    prompt: &str,
    transport: Transport,
    mut input: mpsc::Receiver<(u64, Input)>,
    llama: Arc<LlamaEngine>,
) -> Result<(), Error> {
    transport.send(json!({"type":"state","phase":"loading"}));
    tokio::select! {
        result = llama.start() => { result?; }
        _ = transport.until_expired() => return Ok(()),
    }
    let root = options.root.clone();
    // Native libraries have blocking initialization and may not share GGML in one process.
    let (asr, voice, capture) = tokio::task::spawn_blocking(move || -> Result<_, Error> {
        let asr: Arc<dyn Recognizer> = Arc::new(crate::native::NativeEngine::load("asr", &root)?);
        let voice: Arc<dyn Synthesizer> =
            Arc::new(crate::native::NativeEngine::load("tts", &root)?);
        // Chrome supplies AEC/NS/AGC and the capture graph filters the speech band, so this
        // side only decides when someone is speaking.
        let capture = CapturePipeline::new(SegmenterConfig::default())?;
        Ok((asr, voice, capture))
    })
    .await??;
    if transport.status().2 {
        return Ok(());
    }
    let asr = AsrWorker::spawn(asr, 4, ChunkConfig::default())?;
    let voice = VoiceWorker::spawn(voice)?;
    let (model_tx, model_rx) = mpsc::channel(256);
    let mut budget = WindowBudget::for_slot(llama.config().tokens_per_slot);
    budget.reply_tokens = options.reply_tokens;
    let mut runner = RemoteRunner {
        session: Session::new(
            Conversation::new(prompt, budget),
            TurnTimeouts::default(),
            ChunkLimits::default(),
        ),
        asr,
        voice,
        capture,
        input: Utterance::default(),
        llama,
        transport: transport.clone(),
        model_tx,
        model_rx,
        model: None,
        cancelled: Arc::new(AtomicBool::new(false)),
        epoch: Instant::now(),
        filter: options.filter,
        reply_tokens: options.reply_tokens,
        default_prompt: options.system_prompt.clone(),
        loudness: Loudness::new(options.gain),
        phrases: VecDeque::new(),
        next_phrase: 0,
        timing: TurnTiming::default(),
        quiet_told: false,
        active_at: Instant::now(),
        greeting: None,
        opening: Opening::Open,
        active_phrase: 0,
        audio: VecDeque::new(),
        next_audio: 0,
        last_audio_ack: 0,
        last_phrase_ack: 0,
        last_progress: Instant::now(),
        last_capture: Instant::now(),
        last_state: None,
        last_endpoint: None,
        transcribe_deadline: None,
        pending_speech: VecDeque::new(),
    };
    let mut connection = transport.status();
    transport.send(json!({"type":"ready"}));
    // Open with a voice rather than silence, and listen only once it has been said.
    runner.greeting = Some(request_greeting(
        runner.llama.clone(),
        runner.session.conversation().system().to_owned(),
    ));
    runner.opening = Opening::Greeting;
    let mut timer = tokio::time::interval(Duration::from_millis(10));
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let result = loop {
        timer.tick().await;
        let status = transport.status();
        if status.2 {
            break Ok(());
        }
        if (status.0, status.1) != (connection.0, connection.1) {
            let tasks = runner.session.interrupt(runner.now());
            runner.dispatch(tasks);
            runner.capture.reset_capture();
            if status.1 {
                transport.send(json!({"type":"ready"}));
            }
            connection = status;
        }
        for _ in 0..32 {
            let Ok((revision, event)) = input.try_recv() else {
                break;
            };
            if !transport.accepts(revision) {
                continue;
            }
            if runner.receive(event).is_err() {
                runner.fail("invalid_input");
            }
        }
        runner.poll_workers();
        runner.speak_greeting();
        if status.1 {
            let timeouts = runner.session.timeouts_hit();
            let tasks = runner.session.poll(runner.now());
            if runner.session.timeouts_hit() != timeouts {
                transport.send(json!({"type":"error","code":"turn_timeout"}));
            }
            runner.dispatch(tasks);
            if runner.input.expects_audio()
                && runner.last_capture.elapsed() > Duration::from_secs(3)
            {
                runner.fail("audio_stalled");
            }
            runner.enforce_transcribe_deadline();
            runner.update_quiet();
            if !runner.audio.is_empty() && runner.last_progress.elapsed() > Duration::from_secs(15)
            {
                runner.fail("playback_stalled");
            }
            runner.publish_state();
            runner.publish_endpoint();
        }
        if runner.asr.is_finished() || runner.voice.is_finished() || runner.voice.stalled() {
            break Err("speech worker stopped responding".into());
        }
    };
    let job = runner.model.take();
    runner.cancel_work();
    if let Some(job) = job {
        job.abort();
        let _ = job.await;
    }
    // Threads holding someone's audio are waited for, not detached. Every native request has a
    // finite deadline, so this wait ends; the ceiling is there so that a worker which breaks
    // that promise cannot keep Zen from restarting or quitting, since both wait on this.
    tokio::task::spawn_blocking(move || {
        runner.cancelled.store(true, Ordering::Release);
        if !stop_workers(&mut runner.voice, &mut runner.asr, WORKER_STOP_LIMIT) {
            eprintln!(
                "A native worker did not stop within {} s; it is left to finish on its own",
                WORKER_STOP_LIMIT.as_secs()
            );
        }
    })
    .await?;
    result
}

/// How long a session end may wait for its workers. A cancelled native request answers within
/// its 2 s cancellation deadline, after a cancel write that may itself wait 3 s; twice that
/// leaves room for scheduling without stalling the next session behind a wedged one.
const WORKER_STOP_LIMIT: Duration = Duration::from_secs(10);

/// Stops both workers, polling each in short turns so a slow one does not hold up the other.
/// Returns whether both finished before `limit`.
fn stop_workers(voice: &mut VoiceWorker, asr: &mut AsrWorker, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    let (mut voice_done, mut asr_done) = (false, false);
    loop {
        let turn = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(250));
        voice_done = voice_done || voice.shutdown(turn);
        asr_done = asr_done || asr.shutdown(turn);
        if voice_done && asr_done {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
    }
}

struct RemoteRunner {
    session: Session,
    asr: AsrWorker,
    voice: VoiceWorker,
    capture: CapturePipeline,
    input: Utterance,
    llama: Arc<LlamaEngine>,
    transport: Transport,
    model_tx: mpsc::Sender<ModelEvent>,
    model_rx: mpsc::Receiver<ModelEvent>,
    model: Option<tokio::task::JoinHandle<()>>,
    cancelled: Arc<AtomicBool>,
    epoch: Instant,
    filter: bool,
    reply_tokens: usize,
    default_prompt: String,
    loudness: Loudness,
    phrases: VecDeque<Phrase>,
    next_phrase: u64,
    active_phrase: u64,
    audio: VecDeque<AudioCredit>,
    next_audio: u64,
    last_audio_ack: u64,
    last_phrase_ack: u64,
    last_progress: Instant,
    last_capture: Instant,
    last_state: Option<(&'static str, u64)>,
    /// Last endpoint reported to the page, so a learned change is sent once.
    last_endpoint: Option<usize>,
    /// Deadline for recognizing the complete utterance.
    transcribe_deadline: Option<u64>,
    pending_speech: VecDeque<(Generation, String)>,
    timing: TurnTiming,
    /// The page has been told the session has gone quiet, and nothing has happened since.
    quiet_told: bool,
    /// When Zen last had anything to do: someone talking to it, or it answering.
    active_at: Instant,
    /// The opening line, while the model is still writing it.
    greeting: Option<tokio::task::JoinHandle<String>>,
    /// Nothing the microphone hears is taken as speech until the greeting has been said.
    ///
    /// The browser's echo canceller has to hear some of Zen's voice before it can take it back
    /// out of the microphone - WebRTC's AEC3 stays in a cautious initial state for its first
    /// 2.5 s of playback - and the greeting is the first thing Zen says. Listened to, part of it
    /// could leak back, be taken for someone talking, cut the greeting off, and be answered as
    /// if the listener had said it. The microphone stays open meanwhile, so the canceller still
    /// learns from the greeting; what it hears is just not used. The page plays its ready cue
    /// when the greeting ends, and listening starts once the room has fallen quiet.
    opening: Opening,
}

impl Drop for RemoteRunner {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(job) = self.model.take() {
            job.abort();
        }
    }
}
impl RemoteRunner {
    fn now(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }
    /// Tell the page what the endpoint has settled on. Adaptive mode moves it as it learns,
    /// and a number that changes without explanation is worse than no number at all.
    fn publish_endpoint(&mut self) {
        let ms = self.capture.endpoint_ms();
        if self.last_endpoint != Some(ms) {
            self.last_endpoint = Some(ms);
            self.transport.send(json!({"type":"endpoint","ms":ms}));
        }
    }
    fn publish_state(&mut self) {
        let phase = self.session.phase();
        let shown = if self.opening == Opening::Greeting {
            // Said outside any turn, but said all the same, and nothing is listened to meanwhile.
            "speaking"
        } else {
            phase.as_str()
        };
        let state = (shown, self.session.generation().value());
        if self.last_state != Some(state) {
            self.transport
                .send(json!({"type":"state","phase":state.0,"generation":state.1}));
            self.last_state = Some(state);
        }
    }
    /// Tells the page, once, when Zen has had nothing to do for `QUIET_AFTER`: nobody speaking or
    /// typing to it, and nothing of its own being said.
    fn update_quiet(&mut self) {
        let busy = self.session.phase() != Phase::Idle
            || self.opening == Opening::Greeting
            || !self.phrases.is_empty()
            || !self.audio.is_empty()
            || !self.pending_speech.is_empty();
        if busy {
            self.active();
        } else if !self.quiet_told && self.active_at.elapsed() >= QUIET_AFTER {
            self.quiet_told = true;
            self.transport.send(json!({"type":"quiet"}));
        }
    }

    /// Something happened: the quiet timer starts again.
    fn active(&mut self) {
        self.active_at = Instant::now();
        self.quiet_told = false;
    }

    /// Says the opening line once the model has written it, unless the listener has already
    /// taken over.
    fn speak_greeting(&mut self) {
        let Some(finished) = self
            .greeting
            .as_mut()
            .filter(|job| job.is_finished())
            .and_then(|job| job.now_or_never())
        else {
            return;
        };
        self.greeting = None;
        if self.opening != Opening::Greeting {
            return;
        }
        let text = finished.unwrap_or_else(|_| greeting_line());
        let generation = self.session.generation();
        self.dispatch(vec![Task::Speak { generation, text }]);
    }

    /// Whether what the microphone hears counts yet. See `opening`.
    fn listening(&mut self) -> bool {
        match self.opening {
            Opening::Open => true,
            Opening::Greeting => false,
            Opening::Until(at) if Instant::now() < at => false,
            Opening::Until(_) => {
                self.opening = Opening::Open;
                true
            }
        }
    }

    fn cancel_work(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancelled = Arc::new(AtomicBool::new(false));
        if let Some(job) = self.model.take() {
            job.abort();
        }
        self.input.cancel();
        self.transcribe_deadline = None;
        // A greeting cut short - by typing, the stop button, a failure - still leaves its echo
        // in the room for a moment. Nobody needs the cue: they have already taken over. One
        // still being written is not said at all.
        if let Some(job) = self.greeting.take() {
            job.abort();
        }
        if self.opening == Opening::Greeting {
            self.opening = Opening::Until(Instant::now() + ROOM_TAIL);
        }
        self.phrases.clear();
        self.audio.clear();
        self.pending_speech.clear();
        self.timing = TurnTiming::default();
        self.transport
            .send(json!({"type":"clear","generation":self.session.generation().value()}));
    }
    /// Sends one block of finished audio to the page and records what it is owed for it.
    ///
    /// Every sample the page is sent passes through here, so the credit the engine waits on
    /// before sending more can never disagree with what was actually sent.
    fn send_block(&mut self, generation: Generation, samples: &[f32]) {
        // Rounded, not truncated: truncation maps everything within one step either side of zero
        // to zero, a dead band in the quietest part of the signal.
        let bytes: Vec<_> = samples
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
            .collect();
        if self.audio.is_empty() {
            self.last_progress = Instant::now();
        }
        self.next_audio += 1;
        if let Some(phrase) = self.phrases.back_mut() {
            phrase.sent += samples.len();
        }
        self.audio.push_back(AudioCredit {
            sequence: self.next_audio,
            phrase: self.active_phrase,
            samples: samples.len(),
        });
        self.transport.send_audio(
            generation.value(),
            self.active_phrase,
            self.next_audio,
            &bytes,
        );
    }

    /// Counts a block the page has played towards its phrase, and tells the session once more
    /// of that phrase is certainly heard.
    fn credit_heard(&mut self, generation: Generation, played: AudioCredit) {
        let Some(phrase) = self.phrases.iter_mut().find(|p| p.id == played.phrase) else {
            return;
        };
        phrase.heard += played.samples;
        let ms = |samples: usize| samples * 1000 / TTS_SAMPLE_RATE as usize;
        // The limiter delays everything by its look-ahead, so the first samples are silence.
        let heard_ms = ms(phrase.heard.saturating_sub(Loudness::LATENCY));
        let total_ms = phrase.ended.then(|| ms(phrase.sent));
        let part = heard_part(&phrase.text, heard_ms, total_ms);
        if part.len() > phrase.credited {
            phrase.credited = part.len();
            self.session.on_heard(generation, part);
        }
    }

    fn fail(&mut self, code: &'static str) {
        self.transport.send(json!({"type":"error","code":code}));
        let tasks = self
            .session
            .on_failure(self.session.generation(), code.into(), self.now());
        self.dispatch(tasks);
        self.capture.reset_capture();
    }
    /// A partial hypothesis can omit a negation or the actual question. Report incomplete
    /// recognition instead of promoting it to a complete user turn.
    fn enforce_transcribe_deadline(&mut self) {
        let Some(deadline) = self.transcribe_deadline else {
            return;
        };
        if self.now() < deadline {
            return;
        }
        self.transcribe_deadline = None;
        if self.input.waiting_for_recognition() {
            self.fail("recognition_incomplete");
        }
    }

    fn dispatch(&mut self, tasks: Vec<Task>) {
        let mut tasks: VecDeque<_> = tasks.into();
        while let Some(task) = tasks.pop_front() {
            match task {
                Task::Cancel { .. } => self.cancel_work(),
                Task::StopSpeaking | Task::Transcribe { .. } => {}
                Task::ShowCut { generation, heard } => {
                    self.transport.send(json!({
                        "type": "heard",
                        "generation": generation.value(),
                        "text": heard,
                    }));
                }
                Task::DisplayTranscript { generation, text } => {
                    if self.session.is_current(generation) {
                        self.transport.send(json!({
                            "type": "transcript",
                            "text": text,
                            "generation": generation.value(),
                            "filtered": true,
                        }));
                    }
                }
                Task::Filter {
                    generation,
                    revision,
                    raw,
                } => {
                    if raw.len() > 32_768 {
                        self.fail("transcript_too_long");
                        continue;
                    }
                    if !self.filter {
                        tasks.extend(self.session.use_raw_transcript(
                            generation,
                            revision,
                            self.now(),
                        ));
                        continue;
                    }
                    let budget = filter_budget(&raw);
                    let messages = vec![
                        ("system", crate::bridge::FILTER_RULES.to_string()),
                        ("user", json!({"transcript":raw}).to_string()),
                    ];
                    self.start_model(generation, messages, Some((budget, revision)));
                }
                Task::Reply {
                    generation,
                    messages,
                } => {
                    TurnTiming::mark(&mut self.timing.requested);
                    self.start_model(generation, messages, None)
                }
                Task::Speak { generation, text } => {
                    if !self.session.is_current(generation) {
                        continue;
                    }
                    if self.pending_speech.len() >= 128 {
                        self.fail("synthesis_backpressure");
                    } else {
                        self.pending_speech.push_back((generation, text));
                    }
                }
            }
        }
    }
    /// Runs one model request. `filter_budget` present means this is the filter slot, and
    /// carries the tokens it may spend on this particular transcript.
    fn start_model(
        &mut self,
        generation: Generation,
        messages: Vec<(&'static str, String)>,
        // Present for a repair: how many tokens it may spend, and which version of the question
        // it is repairing. The revision travels back out with the result, because aborting this
        // job does not unsend a result already in the channel.
        filter_budget: Option<(usize, u64)>,
    ) {
        if let Some(job) = self.model.take() {
            job.abort();
        }
        let llama = self.llama.clone();
        let tx = self.model_tx.clone();
        let filter = filter_budget.is_some();
        let revision = filter_budget.map_or(0, |(_, revision)| revision);
        let max_tokens = filter_budget.map_or(self.reply_tokens, |(budget, _)| budget);
        // The filter's deadline follows its budget for the same reason the budget follows the
        // transcript: a long turn has more to repeat back, and a fixed eight seconds would
        // abandon the turn outright on exactly the utterances the larger budget was added for.
        let deadline = if filter {
            Duration::from_secs((8 + max_tokens as u64 / 64).min(20))
        } else {
            Duration::from_secs(30)
        };
        self.model = Some(tokio::spawn(async move {
            let tokens = tx.clone();
            let request = llama.client().stream_completion_with(
                if filter {
                    SlotKind::Filter
                } else {
                    SlotKind::Talker
                },
                &messages,
                max_tokens,
                if filter { 0.1 } else { 0.7 },
                move |text| filter || tokens.try_send(ModelEvent::Token(generation, text)).is_ok(),
            );
            let event = match tokio::time::timeout(deadline, request).await {
                // A filter line cut off by the budget is the speaker's own question with the
                // end missing, and it passes every resemblance check because everything left
                // in it really was said. It has to be refused here, where the reason is known.
                Ok(Ok(reply)) if filter && reply.truncated => {
                    ModelEvent::FilterTruncated(generation, revision)
                }
                Ok(Ok(reply)) if filter => ModelEvent::Filter(generation, revision, reply.text),
                Ok(Ok(_)) => ModelEvent::Done(generation),
                _ if filter => ModelEvent::FilterFailed(generation, revision),
                _ => ModelEvent::Failed(generation),
            };
            let _ = tx.send(event).await;
        }));
    }
    fn capture_event(&mut self, event: CaptureEvent) -> Result<(), Error> {
        match event {
            CaptureEvent::Started => {
                // The question already holds all one turn can, and is being recognised. It is
                // answered as it stands; what is said over it now cannot join it.
                if self.session.phase() == Phase::Transcribing && self.input.is_full() {
                    return Ok(());
                }
                // If recognition of the previous utterance is still outstanding, the session
                // keeps the same turn, so the jobs already in flight have to be kept with it.
                let continuing = self.session.phase() == Phase::Transcribing
                    && self.input.generation() == Some(self.session.generation())
                    && !self.input.is_full();
                let tasks = self.session.on_speech(self.now());
                self.dispatch(tasks);
                if continuing {
                    if let Some(job) = self.model.take() {
                        job.abort();
                    }
                    // The deadline was set for the utterance as it stood at the endpoint that
                    // has just been undone; a new one is set when this one ends.
                    self.transcribe_deadline = None;
                    self.input.reopen();
                } else {
                    self.input.begin(self.session.generation());
                    self.timing = TurnTiming::default();
                }
            }
            CaptureEvent::Segment(segment) => {
                // No turn is open, or this one has already been closed off - by its endpoint, or
                // because it could take no more. Either way this audio belongs to the next turn.
                if self.input.generation().is_none() || self.input.is_closed() {
                    return Ok(());
                }
                let ms = segment.duration_ms();
                let overlaps = segment.overlaps_previous;
                let accepted = match self.asr.submit_cancellable(segment, self.cancelled.clone()) {
                    Ok(sequence) => self.input.add(sequence, ms, overlaps).is_ok(),
                    Err(_) => false,
                };
                if !accepted {
                    // Recognition is behind, or the turn has outgrown what one utterance holds.
                    // Neither is worth what reporting it as an error costs: the turn, including
                    // every word already recognised. End it here instead and answer from the
                    // pieces that were accepted.
                    self.end_turn();
                }
            }
            CaptureEvent::Ended => self.end_turn(),
        }
        Ok(())
    }
    /// Close the utterance to further audio and answer from what it holds.
    fn end_turn(&mut self) {
        if !self.input.expects_audio() {
            return;
        }
        self.input.close();
        self.transcribe_deadline = Some(self.now() + transcribe_budget_ms(self.input.audio_ms()));
        let tasks = self.session.on_turn_ended(self.now());
        self.dispatch(tasks);
        self.timing.ended(self.session.generation());
        self.finalize();
    }

    fn finalize(&mut self) {
        if let Some((g, text)) = self.input.take_ready() {
            self.transcribe_deadline = None;
            let unheard = self.input.unheard_ms() > 0;
            if unheard && text.trim().is_empty() {
                // Nothing came through, but something was plainly said. Ask for it again
                // rather than letting the turn end in silence.
                let tasks = self.session.on_unheard(g, self.now());
                self.dispatch(tasks);
                return;
            }
            if unheard {
                // Some of it came through. Answer what was heard, and say that part was not.
                self.transport
                    .send(json!({"type":"notice","code":"partly_unheard","generation":g.value()}));
            }
            TurnTiming::mark(&mut self.timing.transcribed);
            let tasks = self.session.on_transcript(g, text, self.now());
            self.dispatch(tasks);
        }
    }
    fn receive(&mut self, input: Input) -> Result<(), Error> {
        let generation = self.session.generation();
        match input {
            Input::Audio(bytes) => {
                let samples: Vec<_> = bytes
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                self.last_capture = Instant::now();
                if !self.listening() {
                    return Ok(());
                }
                for event in self.capture.push_events(&samples)? {
                    self.capture_event(event)?;
                }
            }
            Input::Control(Control::Text { text }) => {
                if text.trim().is_empty() || text.len() > 8_192 || text.contains('\0') {
                    return Err("invalid text".into());
                }
                self.active();
                self.capture.reset_capture();
                let tasks = self.session.interrupt(self.now());
                self.dispatch(tasks);
                let tasks = self.session.on_speech(self.now());
                self.dispatch(tasks);
                let tasks = self.session.on_turn_ended(self.now());
                self.dispatch(tasks);
                self.timing.ended(self.session.generation());
                self.timing.typed = true;
                TurnTiming::mark(&mut self.timing.transcribed);
                let tasks = self
                    .session
                    .on_typed(self.session.generation(), text, self.now());
                self.dispatch(tasks);
            }
            Input::Control(Control::LearnedPause { ms }) => {
                // What the last session learned about how this person pauses. Starting from it
                // rather than from the default means the first few turns of every session are
                // not spent relearning it.
                if !(1..=10_000).contains(&ms) {
                    return Err("learned pause must be 1..10000 ms".into());
                }
                self.capture
                    .set_endpoint(EndpointPolicy::adaptive_from_ms(ms));
                self.last_endpoint = None;
            }
            Input::Control(Control::Active {}) => self.active(),
            Input::Control(Control::Interrupt {}) => {
                let tasks = self.session.interrupt(self.now());
                self.dispatch(tasks);
                self.capture.reset_capture();
            }
            Input::Control(Control::ClearHistory { system_prompt }) => {
                self.active();
                let prompt = match system_prompt.as_deref() {
                    Some(custom) => {
                        crate::bridge::validate_prompt(custom)?;
                        crate::bridge::compose_prompt(custom)
                    }
                    None => self.default_prompt.clone(),
                };
                let tasks = self.session.clear_history(Some(prompt), self.now());
                self.dispatch(tasks);
                self.capture.reset_capture();
                self.input = Utterance::default();
                self.transport.send(json!({"type":"history_cleared"}));
            }
            Input::Control(Control::EndAudio {}) => {
                // Flush preserves the unpadded fractional frame and trailing consonant.
                if let Some(segment) = self.capture.flush() {
                    self.capture_event(CaptureEvent::Segment(segment))?;
                }
                self.capture_event(CaptureEvent::Ended)?;
                self.capture.reset_capture();
            }
            Input::Control(Control::AudioPlayed {
                generation: g,
                sequence,
            }) if g == generation.value() => {
                if sequence <= self.last_audio_ack {
                    return Ok(());
                }
                if !self.audio.front().is_some_and(|a| a.sequence == sequence) {
                    return Err("out of order audio acknowledgement".into());
                }
                let played = self.audio.pop_front().unwrap();
                self.last_audio_ack = sequence;
                self.last_progress = Instant::now();
                self.credit_heard(generation, played);
            }
            Input::Control(Control::PlaybackStarted {
                generation: g,
                phrase,
            }) if g == generation.value() => {
                if self.phrases.front().is_some_and(|p| p.id == phrase) {
                    self.session.on_playback_started(generation, self.now());
                    if let Some(report) = self.timing.report(generation) {
                        self.transport.send(report);
                    }
                }
            }
            Input::Control(Control::Played {
                generation: g,
                phrase,
            }) if g == generation.value() => {
                if phrase <= self.last_phrase_ack {
                    return Ok(());
                }
                if !self
                    .phrases
                    .front()
                    .is_some_and(|p| p.id == phrase && p.ended)
                    || self.audio.iter().any(|a| a.phrase == phrase)
                {
                    return Err("invalid phrase acknowledgement".into());
                }
                let completed = self.phrases.pop_front().unwrap();
                self.last_phrase_ack = phrase;
                if self.opening == Opening::Greeting
                    && self.phrases.is_empty()
                    && self.pending_speech.is_empty()
                {
                    // Said in full. The page cues that it is the listener's turn.
                    self.opening = Opening::Until(Instant::now() + ROOM_TAIL);
                    self.transport.send(json!({"type":"greeted"}));
                }
                self.session.on_spoken(generation, completed.text);
                let tasks = self.session.on_playback_finished(generation, self.now());
                self.dispatch(tasks);
            }
            Input::Control(_) => {} // A cancelled generation can never credit playback.
        }
        Ok(())
    }
    fn poll_workers(&mut self) {
        for outcome in self.asr.poll() {
            match outcome {
                AsrOutcome::Complete {
                    sequence,
                    transcript,
                    unheard_ms,
                    queued_ms,
                } => {
                    if self.input.contains(sequence) {
                        self.timing.queue_ms = self.timing.queue_ms.max(queued_ms);
                    }
                    if unheard_ms > 0 {
                        self.input.unheard(sequence, unheard_ms);
                    }
                    if self.input.complete(sequence, transcript.text) {
                        self.transport.send(json!({"type":"transcript_partial","generation":self.session.generation().value(),"text":self.input.preview()}));
                        self.finalize();
                    }
                }
                AsrOutcome::Partial {
                    sequence,
                    chunk,
                    text,
                    ..
                } => {
                    if let Some(text) = self.input.partial(sequence, chunk, text) {
                        self.transport.send(json!({"type":"transcript_partial","generation":self.session.generation().value(),"text":text}));
                    }
                }
                AsrOutcome::Failed { sequence, .. } if self.input.contains(sequence) => {
                    self.fail("recognition_failed")
                }
                _ => {}
            }
        }
        for _ in 0..256 {
            let Ok(event) = self.model_rx.try_recv() else {
                break;
            };
            let tasks = match event {
                ModelEvent::Filter(g, revision, text) => {
                    self.session.on_filter(g, revision, text, self.now())
                }
                ModelEvent::FilterTruncated(g, revision)
                | ModelEvent::FilterFailed(g, revision) => {
                    self.session.use_raw_transcript(g, revision, self.now())
                }
                ModelEvent::Token(g, text) => {
                    if self.session.is_current(g) {
                        TurnTiming::mark(&mut self.timing.first_word);
                    }
                    self.session.on_reply_token(g, &text, self.now())
                }
                ModelEvent::Done(g) => self.session.on_reply_complete(g, self.now()),
                ModelEvent::Failed(g) if self.session.is_current(g) => {
                    self.fail("model_failed");
                    Vec::new()
                }
                _ => Vec::new(),
            };
            self.dispatch(tasks);
        }
        while self.voice.can_submit() {
            let Some((generation, text)) = self.pending_speech.pop_front() else {
                break;
            };
            if self
                .voice
                .submit(generation, text, self.cancelled.clone())
                .is_err()
            {
                self.fail("synthesis_backpressure");
            }
        }
        while self.audio.iter().map(|a| a.samples).sum::<usize>() < TTS_SAMPLE_RATE as usize * 2 {
            let Ok(event) = self.voice.events.try_recv() else {
                break;
            };
            match event {
                VoiceEvent::Begin(g, text) if self.session.is_current(g) => {
                    // Each phrase is its own stream on the page, with a pause before it. The
                    // limiter must not carry the last phrase's tail, or its gain reduction,
                    // across that pause.
                    self.loudness.reset();
                    self.next_phrase += 1;
                    self.active_phrase = self.next_phrase;
                    self.transport.send(json!({"type":"phrase_start","generation":g.value(),"phrase":self.active_phrase,"text":text}));
                    self.phrases.push_back(Phrase {
                        id: self.active_phrase,
                        text,
                        ended: false,
                        sent: 0,
                        heard: 0,
                        credited: 0,
                    });
                }
                VoiceEvent::Audio(g, mut samples) if self.session.is_current(g) => {
                    if samples.iter().any(|s| !s.is_finite()) {
                        self.fail("invalid_synthesis_audio");
                        continue;
                    }
                    self.loudness.process(&mut samples);
                    self.send_block(g, &samples);
                }
                VoiceEvent::End(g, text) if self.session.is_current(g) => {
                    // Synthesis for this phrase is done, so the limiter has no more input to
                    // hold its tail against. Send it before the page is told the phrase ended,
                    // or the ending fade is applied to audio whose last millisecond is missing.
                    let tail = self.loudness.finish();
                    if !tail.is_empty() {
                        self.send_block(g, &tail);
                    }
                    let pause = pause_after_ms(&text);
                    if let Some(phrase) = self.phrases.back_mut() {
                        phrase.text = text.clone();
                        phrase.ended = true;
                    }
                    self.transport.send(json!({"type":"phrase_end","generation":g.value(),"phrase":self.active_phrase,"text":text,"pause_ms":pause}));
                }
                VoiceEvent::Failed(g, _) if self.session.is_current(g) => {
                    self.fail("synthesis_failed")
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{greeting_line, moment, stop_workers, transcribe_budget_ms, usable_greeting};
    use crate::{
        asr::{AsrError, AsrWorker, Recognizer, Transcript},
        audio::ChunkConfig,
        session::Generation,
        tts::{Synthesizer, TtsError},
        voice::VoiceWorker,
    };
    use std::{
        sync::{atomic::AtomicBool, mpsc, Arc, Mutex},
        time::{Duration, Instant},
    };

    /// A synthesizer stuck in native code: it ignores cancellation until the test lets it go.
    struct Wedged(Mutex<mpsc::Receiver<()>>, mpsc::SyncSender<()>);

    impl Synthesizer for Wedged {
        fn synthesize_cancellable(
            &self,
            _: &str,
            _: &AtomicBool,
            _: &mut dyn FnMut(&[f32]) -> bool,
        ) -> Result<(), TtsError> {
            let _ = self.1.send(());
            let _ = self.0.lock().unwrap().recv();
            Ok(())
        }
    }

    struct Silent;

    impl Recognizer for Silent {
        fn transcribe(&self, _: &[f32]) -> Result<Transcript, AsrError> {
            Err(AsrError::NoResult)
        }
    }

    impl Synthesizer for Silent {
        fn synthesize_cancellable(
            &self,
            _: &str,
            _: &AtomicBool,
            _: &mut dyn FnMut(&[f32]) -> bool,
        ) -> Result<(), TtsError> {
            Ok(())
        }
    }

    fn asr() -> AsrWorker {
        AsrWorker::spawn(Arc::new(Silent), 4, ChunkConfig::default()).unwrap()
    }

    #[test]
    fn idle_workers_stop_at_once() {
        let mut voice = VoiceWorker::spawn(Arc::new(Silent)).unwrap();
        let mut asr = asr();
        let started = Instant::now();
        assert!(stop_workers(&mut voice, &mut asr, Duration::from_secs(10)));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn a_wedged_worker_cannot_hold_the_session_end_past_its_limit() {
        // Zen's next session and its exit both wait for this, so an unbounded wait here is a
        // window stuck on "loading" until the process is killed.
        let (release, wait) = mpsc::channel();
        let (entered, running) = mpsc::sync_channel(1);
        let engine = Arc::new(Wedged(Mutex::new(wait), entered));
        let mut voice = VoiceWorker::spawn(engine).unwrap();
        voice
            .submit(
                Generation::default(),
                "Hello.".into(),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        running.recv_timeout(Duration::from_secs(5)).unwrap();
        let mut asr = asr();
        let started = Instant::now();
        assert!(!stop_workers(
            &mut voice,
            &mut asr,
            Duration::from_millis(300)
        ));
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(300),
            "gave up after {waited:?}"
        );
        assert!(waited < Duration::from_secs(2), "held on for {waited:?}");
        release.send(()).unwrap();
        assert!(voice.shutdown(Duration::from_secs(5)));
    }

    #[test]
    fn the_recognition_budget_follows_the_audio() {
        // Short turns get the floor, long ones get room, and nothing waits forever.
        assert_eq!(transcribe_budget_ms(0), 5_000);
        assert_eq!(transcribe_budget_ms(2_000), 5_000);
        assert_eq!(transcribe_budget_ms(30_000), 33_000);
        assert_eq!(transcribe_budget_ms(180_000), 60_000);
    }

    #[test]
    fn the_budget_clears_what_recognition_actually_costs() {
        // Measured on the reference machine: about 300 ms fixed and 630 ms for every second of
        // speech. This budget exists to catch a recogniser that has stopped making progress, so
        // it has to sit clear of one that is merely working - and the earlier figure it was
        // sized from, 240 ms per second, was optimistic by a factor of two and a half.
        //
        // Past about a minute and a half of speech the ceiling binds and the budget is shorter
        // than a from-scratch recognition. That is deliberate: by then the pieces captured at
        // each pause have been coming back for minutes, and answering from them beats silence.
        for seconds in [1_u64, 5, 10, 20, 40, 57] {
            let measured = 300 + 630 * seconds;
            let budget = transcribe_budget_ms(seconds as usize * 1_000);
            assert!(
                budget > measured,
                "{seconds} s of speech costs about {measured} ms to recognise, budget is {budget} ms"
            );
        }
    }

    #[test]
    fn only_a_whole_short_line_of_speech_opens_the_session() {
        // Measured, from the model.
        assert_eq!(
            usable_greeting("Hey Arya, happy Friday! How has your week been treating you so far?"),
            Some("Hey Arya, happy Friday! How has your week been treating you so far?".into())
        );
        assert_eq!(
            usable_greeting("\"Morning, Arya. What should we start with?\""),
            Some("Morning, Arya. What should we start with?".into())
        );
        // Cut off by the token limit, empty, or a speech rather than a greeting.
        assert_eq!(
            usable_greeting("Hello Arya, I was just thinking about how"),
            None
        );
        assert_eq!(usable_greeting(""), None);
        assert_eq!(usable_greeting(&"This goes on and on. ".repeat(12)), None);
    }

    #[test]
    fn the_session_starts_from_a_moment_a_person_would_name() {
        let moment = moment();
        assert!(
            moment == "today"
                || [" morning", " afternoon", " evening", " night"]
                    .iter()
                    .any(|part| moment.ends_with(part)),
            "{moment}"
        );
        assert!(greeting_line().ends_with('?'));
    }
}
