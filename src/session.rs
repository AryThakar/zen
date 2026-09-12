// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Pure turn orchestration. Every asynchronous job keeps the generation issued at capture.
use crate::{
    conversation::{Conversation, SpokenReply},
    reply::{
        accept_verdict, parse_filter_verdict, to_speakable, ChunkLimits, FilterVerdict,
        ReplyChunker,
    },
    turn::{Action, Event, Phase, TurnMachine, TurnTimeouts},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(u64);
impl Generation {
    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    Transcribe {
        generation: Generation,
    },
    Filter {
        generation: Generation,
        raw: String,
    },
    /// The transcript accepted by the repair layer. The page must never display the raw
    /// recogniser text as a committed user turn because it may still be in another language.
    DisplayTranscript {
        generation: Generation,
        text: String,
    },
    Reply {
        generation: Generation,
        messages: Vec<(&'static str, String)>,
    },
    Speak {
        generation: Generation,
        text: String,
    },
    StopSpeaking,
    Cancel {
        through: Generation,
    },
}

pub struct Session {
    turn: TurnMachine,
    conversation: Conversation,
    chunker: ReplyChunker,
    limits: ChunkLimits,
    reply: SpokenReply,
    generation: Generation,
    raw_transcript: String,
    input_open: bool,
    filter_pending: bool,
    generation_done: bool,
    pending_phrases: usize,
}

impl Session {
    pub fn new(conversation: Conversation, timeouts: TurnTimeouts, limits: ChunkLimits) -> Self {
        Self {
            turn: TurnMachine::new(timeouts),
            conversation,
            chunker: ReplyChunker::new(limits),
            limits,
            reply: SpokenReply::new(),
            generation: Generation(0),
            raw_transcript: String::new(),
            input_open: false,
            filter_pending: false,
            generation_done: false,
            pending_phrases: 0,
        }
    }
    pub fn phase(&self) -> Phase {
        self.turn.phase()
    }
    pub fn generation(&self) -> Generation {
        self.generation
    }
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }
    pub fn is_current(&self, generation: Generation) -> bool {
        generation == self.generation
    }
    /// Turns the user cut short. Counted by the turn machine, which is the only place that can
    /// tell a barge-in from an ordinary new turn.
    pub fn interruptions(&self) -> usize {
        self.turn.interruptions()
    }
    /// Turns abandoned because a stage stopped responding. Excludes the benign listening-to-idle
    /// transition, which is a silence, not a fault.
    pub fn timeouts_hit(&self) -> usize {
        self.turn.timeouts_hit()
    }

    fn advance(&mut self) -> Task {
        let through = self.generation;
        self.generation = Generation(
            self.generation
                .0
                .checked_add(1)
                .expect("generation exhausted"),
        );
        self.chunker = ReplyChunker::new(self.limits);
        self.reply = SpokenReply::new();
        self.raw_transcript.clear();
        self.filter_pending = false;
        self.generation_done = false;
        self.pending_phrases = 0;
        self.input_open = false;
        Task::Cancel { through }
    }

    pub fn on_speech(&mut self, now_ms: u64) -> Vec<Task> {
        if self.input_open {
            return Vec::new();
        }
        // Recognition of what was just said is still running, so this is one question
        // continuing. The generation has to stay: advancing it cancels the recognition jobs
        // still in flight and throws away every word they had already produced.
        let mut tasks = Vec::new();
        if self.turn.phase() != Phase::Transcribing {
            if self.turn.phase().interruptible() {
                self.commit_reply(true);
            }
            tasks.push(self.advance());
        }
        self.input_open = true;
        let actions = self.turn.handle(Event::SpeechStarted, now_ms);
        tasks.extend(self.translate(actions));
        tasks
    }

    pub fn on_turn_ended(&mut self, now_ms: u64) -> Vec<Task> {
        if !self.input_open {
            return Vec::new();
        }
        self.input_open = false;
        let actions = self.turn.handle(Event::SegmentReady, now_ms);
        self.translate(actions)
    }

    pub fn on_transcript(&mut self, generation: Generation, raw: String, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation)
            || self.phase() != Phase::Transcribing
            || self.filter_pending
        {
            return Vec::new();
        }
        if raw.trim().is_empty() {
            self.turn.handle(Event::TranscriptEmpty, now_ms);
            return vec![self.advance()];
        }
        self.raw_transcript = raw.trim().to_string();
        self.filter_pending = true;
        vec![Task::Filter {
            generation,
            raw: self.raw_transcript.clone(),
        }]
    }

    pub fn on_filter(
        &mut self,
        generation: Generation,
        response: String,
        now_ms: u64,
    ) -> Vec<Task> {
        let verdict = parse_filter_verdict(&response);
        self.apply_verdict(generation, verdict, now_ms)
    }

    /// Take the turn from the raw transcript, with no repair applied.
    ///
    /// Two callers, for the same reason. Repair may be switched off outright; or the filter may
    /// have run out of token budget partway through its line, in which case what came back is
    /// the speaker's own question with the end cut off, and no resemblance check downstream can
    /// tell - every word left in it really was said. The raw transcript is unpunctuated and
    /// possibly still in another language, and it is complete, which is the property a question
    /// cannot do without.
    ///
    /// The verdict is handed over directly rather than formatted into a `CLEAN:` line and read
    /// back, because that round trip loses everything after the first newline.
    pub fn use_raw_transcript(&mut self, generation: Generation, now_ms: u64) -> Vec<Task> {
        let raw = self.raw_transcript.trim().to_string();
        self.apply_verdict(generation, FilterVerdict::Clean(raw), now_ms)
    }

    fn apply_verdict(
        &mut self,
        generation: Generation,
        verdict: FilterVerdict,
        now_ms: u64,
    ) -> Vec<Task> {
        if !self.is_current(generation)
            || self.phase() != Phase::Transcribing
            || !self.filter_pending
        {
            return Vec::new();
        }
        self.filter_pending = false;
        match accept_verdict(&self.raw_transcript, verdict) {
            // A clarification is the only thing that will be said this turn, so it has to
            // exist. `emit` drops text that is empty or has nothing speakable in it, and the
            // model does sometimes answer with a bare marker and no question - which left the
            // turn in Preparing with nothing to play, nothing to cancel and nothing to
            // acknowledge, until the thinking timeout fired thirty seconds later. Falling
            // back to the transcript keeps the conversation moving.
            FilterVerdict::Ask(question) => match self.emit(generation, question) {
                Some(task) => {
                    self.turn.handle(Event::ReplyReady, now_ms);
                    self.generation_done = true;
                    vec![task]
                }
                None => {
                    let raw = std::mem::take(&mut self.raw_transcript);
                    self.conversation.record_user(raw.clone());
                    let actions = self.turn.handle(Event::TranscriptReady(raw), now_ms);
                    self.translate(actions)
                }
            },
            FilterVerdict::Clean(text) => {
                let display = text.clone();
                self.conversation.record_user(text.clone());
                let actions = self.turn.handle(Event::TranscriptReady(text), now_ms);
                let mut tasks = vec![Task::DisplayTranscript {
                    generation,
                    text: display,
                }];
                tasks.extend(self.translate(actions));
                tasks
            }
        }
    }

    pub fn on_reply_token(&mut self, generation: Generation, text: &str, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation)
            || self.generation_done
            || !matches!(
                self.phase(),
                Phase::Thinking | Phase::Preparing | Phase::Speaking
            )
        {
            return Vec::new();
        }
        self.reply.generated(text);
        let mut tasks = Vec::new();
        let mut chunk = self.chunker.push(text);
        while let Some(ready) = chunk {
            self.turn.handle(Event::ReplyReady, now_ms);
            tasks.extend(self.emit(generation, ready));
            chunk = self.chunker.push("");
        }
        tasks
    }

    pub fn on_reply_complete(&mut self, generation: Generation, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation)
            || self.generation_done
            || !matches!(
                self.phase(),
                Phase::Thinking | Phase::Preparing | Phase::Speaking
            )
        {
            return Vec::new();
        }
        let mut tasks = Vec::new();
        if let Some(tail) = self.chunker.flush() {
            self.turn.handle(Event::ReplyReady, now_ms);
            tasks.extend(self.emit(generation, tail));
        }
        self.generation_done = true;
        if self.pending_phrases == 0 && self.reply.is_empty() {
            return self.on_failure(
                generation,
                "The model returned no speakable response".into(),
                now_ms,
            );
        }
        tasks.extend(self.on_playback_finished(generation, now_ms));
        tasks
    }

    /// Called when the device callback has consumed the first audio for this reply.
    pub fn on_playback_started(&mut self, generation: Generation, now_ms: u64) {
        if self.is_current(generation) {
            self.turn.handle(Event::ReplyStarted, now_ms);
        }
    }

    /// Only the playback ledger may acknowledge a phrase, after its final sample was rendered.
    pub fn on_spoken(&mut self, generation: Generation, text: String) {
        if self.is_current(generation) && self.pending_phrases > 0 {
            self.pending_phrases -= 1;
            self.reply.played(text);
        }
    }

    pub fn on_playback_finished(&mut self, generation: Generation, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation) || !self.generation_done || self.pending_phrases != 0 {
            return Vec::new();
        }
        self.commit_reply(false);
        let actions = self.turn.handle(Event::ReplyFinished, now_ms);
        let cancel = self.advance();
        let mut tasks = vec![cancel];
        tasks.extend(self.translate(actions));
        tasks
    }

    /// Cancel a turn without inventing an input utterance. Used for transport loss and stop.
    /// Cancel anything in flight and empty the conversation window. The turn machine
    /// returns to idle through the same interruption path a user barge-in takes, so no
    /// half-finished turn can land afterwards and be credited to a history that no
    /// longer exists.
    pub fn clear_history(&mut self, system: Option<String>, now_ms: u64) -> Vec<Task> {
        let tasks = self.interrupt(now_ms);
        if let Some(system) = system {
            self.conversation.set_system(system);
        }
        self.conversation.clear_history();
        tasks
    }

    pub fn interrupt(&mut self, now_ms: u64) -> Vec<Task> {
        self.commit_reply(true);
        self.turn.handle(Event::Failed(String::new()), now_ms);
        vec![self.advance(), Task::StopSpeaking]
    }

    fn commit_reply(&mut self, interrupted: bool) {
        let reply = std::mem::take(&mut self.reply);
        // Nothing was ever spoken in answer to this question, and the turn is being abandoned.
        // Leaving it in the window stacks unanswered questions up until a later reply answers
        // one of the older ones instead of the one just asked.
        if interrupted && reply.is_empty() {
            self.conversation.drop_unanswered_question();
            return;
        }
        self.conversation.record_reply(reply, interrupted);
    }

    pub fn on_failure(&mut self, generation: Generation, reason: String, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation) {
            return Vec::new();
        }
        self.commit_reply(true);
        let actions = self.turn.handle(Event::Failed(reason), now_ms);
        let cancel = self.advance();
        let mut tasks = vec![cancel];
        tasks.extend(self.translate(actions));
        tasks
    }

    pub fn poll(&mut self, now_ms: u64) -> Vec<Task> {
        // A listening timeout is a silence timeout, not a limit on someone still speaking.
        if self.input_open {
            return Vec::new();
        }
        // A timeout is a phase change, not a set of actions: giving up on a stalled
        // recogniser has nothing to cancel, so the returned list can legitimately be empty.
        let before = self.phase();
        let actions = self.turn.poll(now_ms);
        if self.phase() == before {
            return Vec::new();
        }
        self.commit_reply(true);
        let cancel = self.advance();
        let mut tasks = vec![cancel, Task::StopSpeaking];
        tasks.extend(self.translate(actions));
        tasks
    }

    fn emit(&mut self, generation: Generation, chunk: String) -> Option<Task> {
        let text = to_speakable(&chunk);
        if text.trim().is_empty() {
            return None;
        }
        self.pending_phrases += 1;
        Some(Task::Speak { generation, text })
    }

    fn translate(&mut self, actions: Vec<Action>) -> Vec<Task> {
        let generation = self.generation;
        actions
            .into_iter()
            .filter_map(|action| match action {
                Action::Transcribe => Some(Task::Transcribe { generation }),
                Action::Think(_) => Some(Task::Reply {
                    generation,
                    messages: self.conversation.messages(),
                }),
                Action::CancelSpeaking => Some(Task::StopSpeaking),
                Action::CancelThinking => None,
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::WindowBudget;

    #[test]
    fn transport_interrupt_preserves_heard_history_and_allows_a_fresh_utterance() {
        let mut session = session();
        let (old, text) = to_speaking(&mut session, 0);
        session.on_spoken(old, text.clone());
        session.interrupt(700);
        assert!(!session.is_current(old));
        assert_eq!(session.phase(), Phase::Listening);
        assert_eq!(session.conversation().turns().last().unwrap().text, text);
        session.on_speech(800);
        session.on_turn_ended(900);
        assert_eq!(session.phase(), Phase::Transcribing);
        assert!(session.on_reply_complete(old, 1000).is_empty());
    }
    #[test]
    fn speech_while_recognition_is_running_keeps_the_same_turn() {
        // The generation is what the recognition jobs are filed under. Advancing it here would
        // cancel the jobs still running on the first half of a long question and discard every
        // word they had already produced, so the speaker adding to their own question must not
        // start a new turn.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(1_000);
        assert_eq!(session.phase(), Phase::Transcribing);
        let during = session.generation();

        let tasks = session.on_speech(1_200);

        assert_eq!(session.phase(), Phase::Listening);
        assert_eq!(
            session.generation(),
            during,
            "the turn continues, so the jobs in flight stay addressed to it"
        );
        assert!(
            !tasks.iter().any(|task| matches!(task, Task::Cancel { .. })),
            "cancelling would discard the recognition already done: {tasks:?}"
        );
        // And it still finishes as one question.
        session.on_turn_ended(2_000);
        assert_eq!(session.phase(), Phase::Transcribing);
        let tasks = session.on_transcript(session.generation(), "both halves".into(), 2_100);
        assert!(tasks.iter().any(|task| matches!(task, Task::Filter { .. })));
    }

    #[test]
    fn a_filter_line_cut_off_by_its_budget_falls_back_to_the_whole_transcript() {
        // The filter has to write the transcript back out, so running out of tokens leaves a
        // fluent prefix of the speaker's own question. Nothing downstream can catch that -
        // every word still in it really was said - so the truncation has to be refused here.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let g = session.generation();
        let raw = "remind me what we decided about the schedule for next week and whether the                    room is still booked";
        session.on_transcript(g, raw.into(), 200);
        let tasks = session.use_raw_transcript(g, 300);
        let shown = tasks.iter().find_map(|task| match task {
            Task::DisplayTranscript { text, .. } => Some(text.clone()),
            _ => None,
        });
        assert_eq!(shown.as_deref(), Some(raw));
        assert_eq!(session.conversation().turns().last().unwrap().text, raw);
    }

    #[test]
    fn a_coalesced_model_event_is_drained_into_bounded_phrases() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let g = session.generation();
        session.on_transcript(g, "hello".into(), 200);
        session.on_filter(g, "CLEAN: hello".into(), 300);
        let text = "one two three four five six seven eight nine ten. ".repeat(50);
        let mut tasks = session.on_reply_token(g, &text, 400);
        tasks.extend(session.on_reply_complete(g, 500));
        let phrases: Vec<_> = tasks
            .into_iter()
            .filter_map(|t| {
                if let Task::Speak { text, .. } = t {
                    Some(text)
                } else {
                    None
                }
            })
            .collect();
        assert!(phrases.len() > 5, "got {} phrases", phrases.len());
        assert!(phrases.iter().all(|p| p.split_whitespace().count() <= 60));
        assert_eq!(
            phrases.join(" ").split_whitespace().collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
    }

    fn session() -> Session {
        Session::new(
            Conversation::new("You are Zen.", WindowBudget::for_slot(8_192)),
            TurnTimeouts::default(),
            ChunkLimits::default(),
        )
    }

    /// Drives a session until a phrase has actually been handed to synthesis.
    ///
    /// Pushing a token or two is not enough: the chunker holds text until a phrase is ready, so a
    /// helper that stopped early would leave the session in a state no real run reaches.
    fn to_speaking(session: &mut Session, at: u64) -> (Generation, String) {
        session.on_speech(at);
        session.on_turn_ended(at + 100);
        let generation = session.generation();
        session.on_transcript(generation, "hello there".into(), at + 200);
        session.on_filter(generation, "CLEAN: hello there".into(), at + 300);
        // A short reply is held until generation finishes and then spoken as a single phrase.
        // That is the ordinary case, and the reason it sounds like one thought rather than two
        // fragments read out in turn, so the helper drives that path rather than a long one.
        let mut spoken = None;
        for piece in "Hello there and how can I help you today? ".split_inclusive(' ') {
            for task in session.on_reply_token(generation, piece, at + 400) {
                if let Task::Speak { text, .. } = task {
                    spoken = Some(text);
                }
            }
        }
        assert_eq!(
            session.phase(),
            Phase::Thinking,
            "a reply this short must not have been cut into phrases"
        );
        for task in session.on_reply_complete(generation, at + 450) {
            if let Task::Speak { text, .. } = task {
                spoken = Some(text);
            }
        }
        assert_eq!(session.phase(), Phase::Preparing);
        session.on_playback_started(generation, at + 500);
        assert_eq!(session.phase(), Phase::Speaking);
        (
            generation,
            spoken.expect("a phrase must have reached synthesis"),
        )
    }

    #[test]
    fn a_turn_flows_from_speech_to_a_spoken_reply() {
        let mut session = session();
        session.on_speech(0);
        assert_eq!(session.phase(), Phase::Listening);

        let tasks = session.on_turn_ended(100);
        assert!(tasks.contains(&Task::Transcribe {
            generation: session.generation()
        }));

        let generation = session.generation();
        let tasks = session.on_transcript(generation, "turn on the light".into(), 900);
        assert!(matches!(tasks.first(), Some(Task::Filter { .. })));

        let tasks = session.on_filter(generation, "CLEAN: turn on the light".into(), 950);
        assert!(tasks.iter().any(|task| matches!(task, Task::Reply { .. })));
        assert_eq!(session.phase(), Phase::Thinking);
    }

    #[test]
    fn a_late_transcript_from_a_cancelled_turn_is_dropped() {
        // The race this module exists for. Speech-to-text takes hundreds of milliseconds, so a
        // result from an abandoned turn always arrives afterwards and would otherwise start a
        // reply to a question the user already moved on from.
        let mut session = session();
        let (stale, _) = to_speaking(&mut session, 0);
        session.on_speech(1_000); // interruption advances the generation
        assert_ne!(session.generation(), stale);

        let tasks = session.on_transcript(stale, "from the old turn".into(), 1_100);
        assert!(tasks.is_empty(), "a spent generation must produce no work");
    }

    #[test]
    fn an_interruption_asks_for_in_flight_work_to_be_abandoned() {
        // Ignoring a late result is not enough: the worker is still burning a core the recogniser
        // needs.
        let mut session = session();
        to_speaking(&mut session, 0);
        let tasks = session.on_speech(1_000);
        assert!(
            tasks.iter().any(|task| matches!(task, Task::Cancel { .. })),
            "an interruption must request cancellation, got {tasks:?}"
        );
        assert!(tasks.contains(&Task::StopSpeaking));
    }

    #[test]
    fn an_interruption_records_only_what_was_played() {
        let mut session = session();
        let (generation, spoken) = to_speaking(&mut session, 0);
        session.on_spoken(generation, spoken.clone());
        // More was generated, but playback never reached it.
        session.on_speech(1_000);

        let last = session.conversation().turns().last().unwrap();
        assert_eq!(last.text, spoken);
        assert!(last.interrupted);
    }

    #[test]
    fn a_completed_reply_is_recorded_as_complete() {
        let mut session = session();
        let (generation, spoken) = to_speaking(&mut session, 0);
        session.on_spoken(generation, spoken.clone());
        session.on_playback_finished(generation, 2_000);

        let last = session.conversation().turns().last().unwrap();
        assert_eq!(last.text, spoken);
        assert!(!last.interrupted);
    }

    #[test]
    fn the_assistant_hearing_itself_is_not_treated_as_a_new_utterance() {
        // Echo cancellation leaks under load. We know exactly what is being said, so this costs
        // one comparison and no model.
        let mut session = session();
        let (generation, spoken) = to_speaking(&mut session, 0);
        // The recogniser picks up the assistant's own phrase leaking past echo cancellation.
        let tasks = session.on_transcript(generation, spoken, 500);
        assert!(
            tasks.is_empty(),
            "the assistant's own words must not start a turn"
        );
    }

    #[test]
    fn a_genuine_utterance_during_playback_is_still_accepted() {
        // The echo check must not swallow real speech, or the user cannot interrupt with words.
        let mut session = session();
        to_speaking(&mut session, 0);
        session.on_speech(1000);
        session.on_turn_ended(1500);
        let tasks = session.on_transcript(
            session.generation(),
            "stop that and play music".into(),
            1600,
        );
        assert!(!tasks.is_empty());
    }

    #[test]
    fn an_unclear_utterance_is_answered_by_the_filter_itself() {
        // The clarification is spoken directly and the reply slot is never told: putting a
        // garbled transcript into the conversation would poison every later turn.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "mm uh hm".into(), 200);
        let tasks = session.on_filter(
            generation,
            "ASK: Sorry, did you mean the kitchen light?".into(),
            300,
        );

        assert!(tasks.iter().any(|task| matches!(task, Task::Speak { .. })));
        assert!(
            !tasks.iter().any(|task| matches!(task, Task::Reply { .. })),
            "an unclear utterance must never reach the reply slot"
        );
        assert_eq!(session.conversation().turn_count(), 0);
    }

    #[test]
    fn an_invented_correction_loses_to_the_raw_transcript() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "uh the the mm kitchen".into(), 200);
        session.on_filter(
            generation,
            "CLEAN: Please book me a table for two at eight o'clock".into(),
            300,
        );
        let recorded = &session.conversation().turns().next().unwrap().text;
        assert_eq!(recorded, "uh the the mm kitchen");
    }

    #[test]
    fn spoken_text_is_made_speakable_before_it_reaches_synthesis() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "how many".into(), 200);
        session.on_filter(generation, "CLEAN: how many".into(), 300);

        let mut spoken = Vec::new();
        for piece in ["There are 3 apples", " and 50% are ripe. ", "That is all."] {
            for task in session.on_reply_token(generation, piece, 400) {
                if let Task::Speak { text, .. } = task {
                    spoken.push(text);
                }
            }
        }
        let all = spoken.join(" ");
        assert!(!all.contains('3'), "digits reached synthesis: {all:?}");
        assert!(!all.contains('%'), "symbols reached synthesis: {all:?}");
    }

    #[test]
    fn a_timeout_abandons_the_turn_and_its_in_flight_work() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let stale = session.generation();
        let tasks = session.poll(120_100);
        assert!(tasks.iter().any(|task| matches!(task, Task::Cancel { .. })));
        assert!(!session.is_current(stale));
    }

    #[test]
    fn a_failure_advances_the_generation_so_nothing_stale_lands() {
        let mut session = session();
        let (generation, _) = to_speaking(&mut session, 0);
        session.on_failure(generation, "asr crashed".into(), 500);
        assert!(!session.is_current(generation));
        assert!(session
            .on_reply_token(generation, "too late", 600)
            .is_empty());
    }

    #[test]
    fn generations_only_move_forward() {
        let mut session = session();
        let first = session.generation();
        to_speaking(&mut session, 0);
        session.on_speech(1_000);
        let second = session.generation();
        session.on_failure(second, "boom".into(), 1_100);
        let third = session.generation();
        assert!(first < second && second < third);
    }

    #[test]
    fn interrupting_before_any_audio_played_leaves_no_half_exchange_behind() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "hello".into(), 200);
        session.on_filter(generation, "CLEAN: hello".into(), 300);
        // The user cut in while the model was still thinking, so nothing was ever said in
        // answer. Keeping the question would leave the window holding half an exchange.
        session.on_speech(400);
        assert_eq!(session.conversation().turn_count(), 0);
    }

    #[test]
    fn restating_a_question_does_not_stack_unanswered_ones_in_the_window() {
        // Ask, get no reply, ask again, ask again. Every question was recorded the moment its
        // transcript was accepted, so without dropping them the window ends up holding three
        // in a row and the next reply is composed against all of them - which is how a
        // question about a lion gets answered two turns after it was abandoned.
        let mut session = session();
        for (at, question) in [(0, "tell me a lion story"), (1_000, "tell me a life story")] {
            session.on_speech(at);
            session.on_turn_ended(at + 100);
            let generation = session.generation();
            session.on_transcript(generation, question.into(), at + 200);
            session.on_filter(generation, format!("CLEAN: {question}"), at + 300);
            assert_eq!(session.phase(), Phase::Thinking);
        }
        session.on_speech(2_000);
        session.on_turn_ended(2_100);
        let generation = session.generation();
        session.on_transcript(generation, "hello".into(), 2_200);
        let tasks = session.on_filter(generation, "CLEAN: hello".into(), 2_300);

        let messages = tasks
            .into_iter()
            .find_map(|task| match task {
                Task::Reply { messages, .. } => Some(messages),
                _ => None,
            })
            .expect("the last question is answered");
        let asked: Vec<_> = messages
            .iter()
            .filter(|(role, _)| *role == "user")
            .map(|(_, text)| text.as_str())
            .collect();
        assert_eq!(
            asked,
            ["hello"],
            "abandoned questions must not still be in the window"
        );
    }
}
