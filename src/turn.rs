// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Turn-taking: the state machine that makes the pipeline feel like a conversation.
//!
//! Everything below this is mechanism - audio in, text out, audio back. This module decides
//! *when* each of those is allowed to happen, which is what separates a voice assistant from a
//! transcription demo.
//!
//! ```text
//! IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> PREPARING -> SPEAKING
//!             ^              |           |           |           |
//!             +--------------+-----------+-----------+-----------+
//!                  confirmed speech interrupts any active work
//! ```
//!
//! Speech must first pass the capture detector. Confirmed speech cancels active work even
//! before playback starts, so the user can correct a question without waiting for a reply.
//! Preparing and speaking are separate: generated text is not proof that audio was heard.
//!
//! Time enters through an explicit millisecond clock rather than `Instant`, so every timeout is
//! reachable from a test without sleeping.

/// What the assistant is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Nothing heard recently. The detector is live but nothing is queued.
    Idle,
    /// A person is speaking; audio is accumulating.
    Listening,
    /// Speech-to-text is running. New confirmed speech interrupts it.
    Transcribing,
    /// The model is composing a reply. New confirmed speech interrupts it.
    Thinking,
    /// Text is ready and synthesis is preparing the first audio.
    Preparing,
    /// A reply is being spoken. Barge-in is armed.
    Speaking,
}

impl Phase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Listening => "listening",
            Phase::Transcribing => "transcribing",
            Phase::Thinking => "thinking",
            Phase::Preparing => "preparing",
            Phase::Speaking => "speaking",
        }
    }

    /// Whether a reply in this phase can be cut short by the user.
    /// Whether speech arriving now interrupts work that has already produced something.
    ///
    /// Transcribing is deliberately absent: nothing has been said back yet, so speech there
    /// continues the question rather than cutting anything off.
    pub const fn interruptible(self) -> bool {
        matches!(self, Phase::Thinking | Phase::Preparing | Phase::Speaking)
    }
}

/// Something that happened, from the audio pipeline or a worker.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The detector opened an utterance.
    SpeechStarted,
    /// A complete utterance is ready to transcribe.
    SegmentReady,
    /// Speech-to-text finished.
    TranscriptReady(String),
    /// Speech-to-text produced nothing usable.
    TranscriptEmpty,
    /// A response has text ready for synthesis, including a clarification.
    ReplyReady,
    /// The first audio of a reply is about to play.
    ReplyStarted,
    /// The reply finished playing.
    ReplyFinished,
    /// A worker failed.
    Failed(String),
}

/// What the host should do as a result of an event or a timeout.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Hand the captured utterance to speech-to-text.
    Transcribe,
    /// Ask the model for a reply.
    Think(String),
    /// Abandon the in-flight model request.
    CancelThinking,
    /// Stop playback immediately.
    CancelSpeaking,
}

/// Timeouts, in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct TurnTimeouts {
    /// Silence in [`Phase::Listening`] before giving up on an utterance.
    pub listening_ms: u64,
    /// How long speech-to-text may take before the turn is abandoned.
    ///
    /// A backstop, not the working deadline, and it has to outlast one: the host applies a
    /// deadline sized from the audio actually captured and answers from the words recognised
    /// so far, where this discards the turn outright. It only catches the case where nothing
    /// was recognised at all.
    ///
    /// Measured on the reference machine, recognition costs about 300 ms plus 630 ms for every
    /// second of speech - 1.0 s of audio in 763 ms, 7.3 s in 3.8 s, 33.9 s in 21.3 s - and a
    /// turn may hold three minutes of it, so this sits well clear of the working deadline.
    pub transcribe_ms: u64,
    /// How long the model may take before the turn is abandoned.
    pub thinking_ms: u64,
    /// Hard cap on a single spoken reply.
    pub speaking_ms: u64,
}

impl Default for TurnTimeouts {
    fn default() -> Self {
        Self {
            listening_ms: 15_000,
            transcribe_ms: 120_000,
            thinking_ms: 30_000,
            speaking_ms: 120_000,
        }
    }
}

/// Drives one conversation's turn-taking.
///
/// Pure logic: it holds no audio, spawns nothing, and never blocks. Callers feed it events and a
/// clock and act on what comes back, which is what makes every timeout and every race reachable
/// from a test.
#[derive(Debug)]
pub struct TurnMachine {
    phase: Phase,
    timeouts: TurnTimeouts,
    /// When the current phase began.
    entered_ms: u64,
    interruptions: usize,
    timeouts_hit: usize,
}

impl TurnMachine {
    pub fn new(timeouts: TurnTimeouts) -> Self {
        Self {
            phase: Phase::Idle,
            timeouts,
            entered_ms: 0,
            interruptions: 0,
            timeouts_hit: 0,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn interruptions(&self) -> usize {
        self.interruptions
    }

    pub fn timeouts_hit(&self) -> usize {
        self.timeouts_hit
    }

    fn enter(&mut self, phase: Phase, now_ms: u64) {
        self.phase = phase;
        self.entered_ms = now_ms;
    }

    fn elapsed(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.entered_ms)
    }

    /// Feeds one event, returning what to do about it.
    pub fn handle(&mut self, event: Event, now_ms: u64) -> Vec<Action> {
        match (self.phase, event) {
            // --- speech detected -------------------------------------------------------------
            (Phase::Idle, Event::SpeechStarted) => {
                self.enter(Phase::Listening, now_ms);
                Vec::new()
            }
            // Already listening, or a follow-up inside the window: no sound, just keep going.
            (Phase::Listening, Event::SpeechStarted) => Vec::new(),

            // The user talking over a reply is the whole point of barge-in. Stop everything
            // immediately - playback first, because that is what they can hear.
            (Phase::Speaking | Phase::Preparing | Phase::Thinking, Event::SpeechStarted) => {
                self.interruptions += 1;
                self.enter(Phase::Listening, now_ms);
                vec![Action::CancelSpeaking, Action::CancelThinking]
            }

            // Nothing has been said back yet - recognition of what was just heard is still
            // running - so more speech is the rest of the same question, not an interruption.
            // Starting over here discards every word already recognised, and on a long question
            // that is the whole question: recognising thirty seconds of speech takes longer
            // than the pause a speaker leaves before adding to it.
            (Phase::Transcribing, Event::SpeechStarted) => {
                self.enter(Phase::Listening, now_ms);
                Vec::new()
            }

            // Busy. Input is captured but must not steer the turn.

            // --- utterance complete ----------------------------------------------------------
            (Phase::Listening, Event::SegmentReady) => {
                self.enter(Phase::Transcribing, now_ms);
                vec![Action::Transcribe]
            }
            (_, Event::SegmentReady) => Vec::new(),

            // --- transcription ---------------------------------------------------------------
            (Phase::Transcribing, Event::TranscriptReady(text)) => {
                self.enter(Phase::Thinking, now_ms);
                vec![Action::Think(text)]
            }
            (Phase::Transcribing, Event::TranscriptEmpty) => {
                // Nothing was said worth answering. Return to listening without a sound, so a
                // cough or a passing noise does not produce a visible reaction.
                self.enter(Phase::Listening, now_ms);
                Vec::new()
            }
            (_, Event::TranscriptReady(_) | Event::TranscriptEmpty) => Vec::new(),

            // --- reply -----------------------------------------------------------------------
            (Phase::Thinking | Phase::Transcribing, Event::ReplyReady) => {
                self.enter(Phase::Preparing, now_ms);
                Vec::new()
            }
            (_, Event::ReplyReady) => Vec::new(),
            (Phase::Thinking | Phase::Preparing, Event::ReplyStarted) => {
                self.enter(Phase::Speaking, now_ms);
                Vec::new()
            }
            (_, Event::ReplyStarted) => Vec::new(),

            (Phase::Speaking | Phase::Preparing, Event::ReplyFinished) => {
                // Straight to listening, not idle: this is the follow-up window, and it is what
                // lets the next turn happen without being re-addressed.
                self.enter(Phase::Listening, now_ms);
                Vec::new()
            }
            (_, Event::ReplyFinished) => Vec::new(),

            // --- failure ---------------------------------------------------------------------
            (_, Event::Failed(_reason)) => {
                self.enter(Phase::Listening, now_ms);
                vec![Action::CancelThinking, Action::CancelSpeaking]
            }
        }
    }

    /// Advances the clock, abandoning any phase that overran its budget.
    ///
    /// Must be called regularly; nothing else notices that a worker has stopped responding.
    /// A timeout is reported by the phase it leaves behind, not by the returned actions: two
    /// of them have nothing to cancel, and callers watch the phase either way.
    pub fn poll(&mut self, now_ms: u64) -> Vec<Action> {
        let elapsed = self.elapsed(now_ms);
        match self.phase {
            Phase::Listening if elapsed >= self.timeouts.listening_ms => {
                self.enter(Phase::Idle, now_ms);
                Vec::new()
            }
            Phase::Transcribing if elapsed >= self.timeouts.transcribe_ms => {
                self.timeouts_hit += 1;
                self.enter(Phase::Listening, now_ms);
                Vec::new()
            }
            Phase::Thinking | Phase::Preparing if elapsed >= self.timeouts.thinking_ms => {
                self.timeouts_hit += 1;
                self.enter(Phase::Listening, now_ms);
                vec![Action::CancelThinking]
            }
            Phase::Speaking if elapsed >= self.timeouts.speaking_ms => {
                // A reply this long is a runaway, not an answer.
                self.timeouts_hit += 1;
                self.enter(Phase::Listening, now_ms);
                vec![Action::CancelSpeaking]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> TurnMachine {
        TurnMachine::new(TurnTimeouts::default())
    }

    /// Drives a machine from idle up to the start of a spoken reply.
    fn to_speaking(machine: &mut TurnMachine, at: u64) {
        machine.handle(Event::SpeechStarted, at);
        machine.handle(Event::SegmentReady, at + 100);
        machine.handle(Event::TranscriptReady("hello".into()), at + 400);
        machine.handle(Event::ReplyStarted, at + 700);
        assert_eq!(machine.phase(), Phase::Speaking);
    }

    #[test]
    fn a_full_turn_walks_idle_to_speaking_and_back_to_listening() {
        let mut machine = machine();
        assert_eq!(machine.phase(), Phase::Idle);

        assert!(machine.handle(Event::SpeechStarted, 0).is_empty());
        assert_eq!(machine.phase(), Phase::Listening);

        assert_eq!(
            machine.handle(Event::SegmentReady, 500),
            vec![Action::Transcribe]
        );
        assert_eq!(machine.phase(), Phase::Transcribing);

        assert_eq!(
            machine.handle(Event::TranscriptReady("what time is it".into()), 900),
            vec![Action::Think("what time is it".into())]
        );
        assert_eq!(machine.phase(), Phase::Thinking);

        machine.handle(Event::ReplyStarted, 1_400);
        assert_eq!(machine.phase(), Phase::Speaking);

        machine.handle(Event::ReplyFinished, 3_000);
        // Listening, not idle - this is the follow-up window.
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn confirmed_speech_while_thinking_cancels_the_old_request() {
        // The rule that keeps a half-heard word from cancelling an answer the user is waiting on.
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        machine.handle(Event::SegmentReady, 500);
        machine.handle(Event::TranscriptReady("hello".into()), 900);
        assert_eq!(machine.phase(), Phase::Thinking);

        assert!(machine
            .handle(Event::SpeechStarted, 1_000)
            .contains(&Action::CancelThinking));
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn speech_while_speaking_interrupts_immediately() {
        // Playback is cancelled before anything else, because it is what the user can hear.
        let mut machine = machine();
        to_speaking(&mut machine, 0);

        let actions = machine.handle(Event::SpeechStarted, 1_000);
        assert_eq!(actions[0], Action::CancelSpeaking);
        assert_eq!(machine.phase(), Phase::Listening);
        assert_eq!(machine.interruptions(), 1);
    }

    #[test]
    fn work_that_has_produced_something_is_interruptible() {
        assert!(Phase::Speaking.interruptible());
        // Transcribing is not interrupted by speech: it has produced nothing to cut off, and
        // the words it has recognised so far belong to the same question.
        for phase in [Phase::Idle, Phase::Listening, Phase::Transcribing] {
            assert!(
                !phase.interruptible(),
                "{phase:?} must not be interruptible"
            );
        }
        for phase in [Phase::Thinking, Phase::Preparing] {
            assert!(phase.interruptible());
        }
    }

    #[test]
    fn speech_during_recognition_continues_the_turn_instead_of_restarting_it() {
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        machine.handle(Event::SegmentReady, 1_000);
        assert_eq!(machine.phase(), Phase::Transcribing);
        // The speaker adds to the question while it is still being recognised.
        let actions = machine.handle(Event::SpeechStarted, 1_200);
        assert_eq!(machine.phase(), Phase::Listening);
        assert!(
            actions.is_empty(),
            "nothing is playing or composing, so there is nothing to cancel: {actions:?}"
        );
        assert_eq!(
            machine.interruptions(),
            0,
            "continuing a question is not an interruption"
        );
    }

    #[test]
    fn an_empty_transcript_returns_to_listening_without_a_sound() {
        // A cough or a door closing must not produce a visible or audible reaction.
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        machine.handle(Event::SegmentReady, 500);
        assert!(machine.handle(Event::TranscriptEmpty, 700).is_empty());
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn a_follow_up_needs_no_second_greeting() {
        let mut machine = machine();
        to_speaking(&mut machine, 0);
        machine.handle(Event::ReplyFinished, 3_000);
        // Already listening, so speech starting again produces no opening sound.
        assert!(machine.handle(Event::SpeechStarted, 3_500).is_empty());
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn listening_gives_up_after_a_long_silence() {
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        assert!(machine.poll(1_000).is_empty());
        assert!(machine.poll(15_000).is_empty());
        assert_eq!(
            machine.phase(),
            Phase::Idle,
            "silence must eventually close the turn"
        );
    }

    #[test]
    fn a_stalled_transcription_does_not_wedge_the_turn() {
        // Without this the assistant sits silent forever and looks broken.
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        machine.handle(Event::SegmentReady, 100);
        // This is the backstop for a recogniser that returned nothing at all. The working
        // deadline is the host's, sized from the audio captured, and it answers from a partial
        // transcript rather than discarding what was said.
        assert!(
            machine.poll(60_000).is_empty(),
            "a long turn is not a stall, and this has to outlast the host deadline it backs up"
        );
        machine.poll(120_100);
        assert_eq!(machine.phase(), Phase::Listening);
        assert_eq!(machine.timeouts_hit(), 1);
    }

    #[test]
    fn a_stalled_model_is_cancelled_rather_than_waited_on() {
        let mut machine = machine();
        machine.handle(Event::SpeechStarted, 0);
        machine.handle(Event::SegmentReady, 100);
        machine.handle(Event::TranscriptReady("hello".into()), 200);
        let actions = machine.poll(30_300);
        assert!(actions.contains(&Action::CancelThinking));
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn a_runaway_reply_is_cut_off() {
        let mut machine = machine();
        to_speaking(&mut machine, 0);
        assert_eq!(machine.poll(200_000), vec![Action::CancelSpeaking]);
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn a_worker_failure_cancels_everything_in_flight() {
        let mut machine = machine();
        to_speaking(&mut machine, 0);
        let actions = machine.handle(Event::Failed("asr crashed".into()), 1_000);
        assert!(actions.contains(&Action::CancelThinking));
        assert!(actions.contains(&Action::CancelSpeaking));
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn events_arriving_in_the_wrong_phase_are_ignored_rather_than_corrupting_state() {
        // Workers are asynchronous, so a late transcript can arrive after a barge-in already
        // moved the turn on. It must not resurrect a cancelled reply.
        let mut machine = machine();
        to_speaking(&mut machine, 0);
        machine.handle(Event::SpeechStarted, 1_000);
        assert_eq!(machine.phase(), Phase::Listening);

        assert!(machine
            .handle(Event::TranscriptReady("stale".into()), 1_100)
            .is_empty());
        assert!(machine.handle(Event::ReplyStarted, 1_200).is_empty());
        assert_eq!(machine.phase(), Phase::Listening);
    }

    #[test]
    fn phases_are_named_for_logs_and_the_user_interface() {
        assert_eq!(Phase::Speaking.as_str(), "speaking");
        assert_eq!(Phase::Transcribing.as_str(), "transcribing");
    }
}
