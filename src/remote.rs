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
    reply::{pause_after_ms, ChunkLimits},
    session::{Generation, Session, Task},
    tts::{Synthesizer, TTS_SAMPLE_RATE},
    turn::TurnTimeouts,
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
    Filter(Generation, String),
    /// The filter ran out of token budget partway through its line.
    FilterTruncated(Generation),
    Token(Generation, String),
    Done(Generation),
    Failed(Generation),
}
struct Phrase {
    id: u64,
    text: Option<String>,
}
struct AudioCredit {
    sequence: u64,
    phrase: u64,
    samples: usize,
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

/// The opening line, chosen from the clock on this machine.
///
/// A session that begins in silence gives no sign that anything is listening. This is said
/// through the same phrase path a reply takes, so it is interruptible and answers to the same
/// generation fence: speak over it and it stops like anything else. It is deliberately not
/// recorded in the conversation - nothing was asked, and the model did not say it.
#[cfg(windows)]
fn greeting_line() -> String {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    // The hour is the user's, not UTC: a greeting that calls midnight morning is worse than none.
    let hour = unsafe { GetLocalTime() }.wHour;
    match hour {
        5..=11 => "Good morning. What's on your mind?",
        12..=16 => "Good afternoon. What's on your mind?",
        17..=21 => "Good evening. What's on your mind?",
        _ => "Hello. What's on your mind?",
    }
    .to_string()
}

#[cfg(not(windows))]
fn greeting_line() -> String {
    "Hello. What's on your mind?".to_string()
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
    let endpoint = options.endpoint;
    // Native libraries have blocking initialization and may not share GGML in one process.
    let (asr, voice, capture) = tokio::task::spawn_blocking(move || -> Result<_, Error> {
        let asr: Arc<dyn Recognizer> = Arc::new(crate::native::NativeEngine::load("asr", &root)?);
        let voice: Arc<dyn Synthesizer> =
            Arc::new(crate::native::NativeEngine::load("tts", &root)?);
        // Chrome supplies AEC/NS/AGC and the capture graph filters the speech band, so this
        // side only decides when someone is speaking.
        let capture = CapturePipeline::new(SegmenterConfig {
            endpoint,
            ..SegmenterConfig::default()
        })?;
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
        active_phrase: 0,
        audio: VecDeque::new(),
        next_audio: 0,
        last_audio_ack: 0,
        last_phrase_ack: 0,
        last_progress: Instant::now(),
        last_capture: Instant::now(),
        last_state: None,
        last_endpoint: None,
        pending_speech: VecDeque::new(),
    };
    let mut connection = transport.status();
    transport.send(json!({"type":"ready"}));
    // Open with a voice rather than silence.
    let opening = Task::Speak {
        generation: runner.session.generation(),
        text: greeting_line(),
    };
    runner.dispatch(vec![opening]);
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
        if status.1 {
            let timeouts = runner.session.timeouts_hit();
            let tasks = runner.session.poll(runner.now());
            if runner.session.timeouts_hit() != timeouts {
                transport.send(json!({"type":"error","code":"turn_timeout"}));
            }
            runner.dispatch(tasks);
            if runner.input.generation().is_some()
                && runner.last_capture.elapsed() > Duration::from_secs(3)
            {
                runner.fail("audio_stalled");
            }
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
    // Do not detach threads with private data. Managed native requests have finite deadlines.
    tokio::task::spawn_blocking(move || {
        runner.cancelled.store(true, Ordering::Release);
        let mut voice_done = runner.voice.shutdown(Duration::from_secs(3));
        let mut asr_done = runner.asr.shutdown(Duration::from_secs(3));
        while !voice_done || !asr_done {
            if !voice_done {
                voice_done = runner.voice.shutdown(Duration::from_secs(1));
            }
            if !asr_done {
                asr_done = runner.asr.shutdown(Duration::from_secs(1));
            }
        }
    })
    .await?;
    result
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
    pending_speech: VecDeque<(Generation, String)>,
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
        let state = (
            self.session.phase().as_str(),
            self.session.generation().value(),
        );
        if self.last_state != Some(state) {
            self.transport
                .send(json!({"type":"state","phase":state.0,"generation":state.1}));
            self.last_state = Some(state);
        }
    }
    fn cancel_work(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancelled = Arc::new(AtomicBool::new(false));
        if let Some(job) = self.model.take() {
            job.abort();
        }
        self.input.cancel();
        self.phrases.clear();
        self.audio.clear();
        self.pending_speech.clear();
        self.transport
            .send(json!({"type":"clear","generation":self.session.generation().value()}));
    }
    fn fail(&mut self, code: &'static str) {
        self.transport.send(json!({"type":"error","code":code}));
        let tasks = self
            .session
            .on_failure(self.session.generation(), code.into(), self.now());
        self.dispatch(tasks);
        self.capture.reset_capture();
    }
    fn dispatch(&mut self, tasks: Vec<Task>) {
        let mut tasks: VecDeque<_> = tasks.into();
        while let Some(task) = tasks.pop_front() {
            match task {
                Task::Cancel { .. } => self.cancel_work(),
                Task::StopSpeaking | Task::Transcribe { .. } => {}
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
                Task::Filter { generation, raw } => {
                    if raw.len() > 32_768 {
                        self.fail("transcript_too_long");
                        continue;
                    }
                    if !self.filter {
                        tasks.extend(self.session.use_raw_transcript(generation, self.now()));
                        continue;
                    }
                    let budget = filter_budget(&raw);
                    let messages = vec![
                        ("system", include_str!("prompts/filter.txt").to_string()),
                        ("user", json!({"transcript":raw}).to_string()),
                    ];
                    self.start_model(generation, messages, Some(budget));
                }
                Task::Reply {
                    generation,
                    messages,
                } => self.start_model(generation, messages, None),
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
        filter_budget: Option<usize>,
    ) {
        if let Some(job) = self.model.take() {
            job.abort();
        }
        let llama = self.llama.clone();
        let tx = self.model_tx.clone();
        let filter = filter_budget.is_some();
        let max_tokens = filter_budget.unwrap_or(self.reply_tokens);
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
                    ModelEvent::FilterTruncated(generation)
                }
                Ok(Ok(reply)) if filter => ModelEvent::Filter(generation, reply.text),
                Ok(Ok(_)) => ModelEvent::Done(generation),
                _ => ModelEvent::Failed(generation),
            };
            let _ = tx.send(event).await;
        }));
    }
    fn capture_event(&mut self, event: CaptureEvent) -> Result<(), Error> {
        match event {
            CaptureEvent::Started => {
                let tasks = self.session.on_speech(self.now());
                self.dispatch(tasks);
                self.input.begin(self.session.generation());
            }
            CaptureEvent::Segment(segment) => {
                let ms = segment.duration_ms();
                if self.input.generation().is_none() {
                    return Ok(());
                }
                let sequence = self
                    .asr
                    .submit_cancellable(segment, self.cancelled.clone())?;
                self.input.add(sequence, ms)?;
            }
            CaptureEvent::Ended => {
                self.input.close();
                let tasks = self.session.on_turn_ended(self.now());
                self.dispatch(tasks);
                self.finalize();
            }
        }
        Ok(())
    }
    fn finalize(&mut self) {
        if let Some((g, text)) = self.input.take_ready() {
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
                for event in self.capture.push_events(&samples)? {
                    self.capture_event(event)?;
                }
            }
            Input::Control(Control::Text { text }) => {
                if text.trim().is_empty() || text.len() > 8_192 || text.contains('\0') {
                    return Err("invalid text".into());
                }
                self.capture.reset_capture();
                let tasks = self.session.interrupt(self.now());
                self.dispatch(tasks);
                let tasks = self.session.on_speech(self.now());
                self.dispatch(tasks);
                let tasks = self.session.on_turn_ended(self.now());
                self.dispatch(tasks);
                let tasks = self
                    .session
                    .on_transcript(self.session.generation(), text, self.now());
                self.dispatch(tasks);
            }
            Input::Control(Control::Tuning { endpoint_ms }) => {
                // `null` means "decide for me": the endpoint is learned from the speaker's
                // own pauses instead of being a number anyone has to guess at.
                let policy = match endpoint_ms {
                    None => EndpointPolicy::adaptive(),
                    Some(ms) if (450..=2000).contains(&ms) => {
                        EndpointPolicy::Fixed(SegmenterConfig::frames_for_ms(ms))
                    }
                    Some(_) => return Err("endpoint must be 450..2000 ms".into()),
                };
                self.capture.set_endpoint(policy);
                self.last_endpoint = None;
            }
            Input::Control(Control::Interrupt {}) => {
                let tasks = self.session.interrupt(self.now());
                self.dispatch(tasks);
                self.capture.reset_capture();
            }
            Input::Control(Control::ClearHistory { system_prompt }) => {
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
                // Complete the last fractional VAD frame before flushing the user's tail.
                for event in self.capture.push_events(&[0.0; 800])? {
                    self.capture_event(event)?;
                }
                if let Some(segment) = self.capture.flush() {
                    self.capture_event(CaptureEvent::Segment(segment))?;
                }
                self.capture_event(CaptureEvent::Ended)?;
                self.capture.reset_capture();
            }
            // Accepted for older pages, but energy hints are not confirmed speech.
            // Cancelling here loses the reply even when Silero rejects the sound as noise.
            Input::Control(Control::SpeechHint {}) => {}
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
                self.audio.pop_front();
                self.last_audio_ack = sequence;
                self.last_progress = Instant::now();
            }
            Input::Control(Control::PlaybackStarted {
                generation: g,
                phrase,
            }) if g == generation.value() => {
                if self.phrases.front().is_some_and(|p| p.id == phrase) {
                    self.session.on_playback_started(generation, self.now());
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
                    .is_some_and(|p| p.id == phrase && p.text.is_some())
                    || self.audio.iter().any(|a| a.phrase == phrase)
                {
                    return Err("invalid phrase acknowledgement".into());
                }
                let completed = self.phrases.pop_front().unwrap();
                self.last_phrase_ack = phrase;
                self.session.on_spoken(generation, completed.text.unwrap());
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
                } => {
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
                ModelEvent::Filter(g, text) => self.session.on_filter(g, text, self.now()),
                ModelEvent::FilterTruncated(g) => self.session.use_raw_transcript(g, self.now()),
                ModelEvent::Token(g, text) => self.session.on_reply_token(g, &text, self.now()),
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
                    self.next_phrase += 1;
                    self.active_phrase = self.next_phrase;
                    self.phrases.push_back(Phrase {
                        id: self.active_phrase,
                        text: None,
                    });
                    self.transport.send(json!({"type":"phrase_start","generation":g.value(),"phrase":self.active_phrase,"text":text}));
                }
                VoiceEvent::Audio(g, mut samples) if self.session.is_current(g) => {
                    if samples.iter().any(|s| !s.is_finite()) {
                        self.fail("invalid_synthesis_audio");
                        continue;
                    }
                    self.loudness.process(&mut samples);
                    let bytes: Vec<_> = samples
                        .iter()
                        .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
                        .collect();
                    if self.audio.is_empty() {
                        self.last_progress = Instant::now();
                    }
                    self.next_audio += 1;
                    self.audio.push_back(AudioCredit {
                        sequence: self.next_audio,
                        phrase: self.active_phrase,
                        samples: samples.len(),
                    });
                    self.transport.send_audio(
                        g.value(),
                        self.active_phrase,
                        self.next_audio,
                        &bytes,
                    );
                }
                VoiceEvent::End(g, text) if self.session.is_current(g) => {
                    let pause = pause_after_ms(&text);
                    if let Some(phrase) = self.phrases.back_mut() {
                        phrase.text = Some(text.clone());
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
