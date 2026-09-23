// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Pure turn orchestration. Every asynchronous job keeps the generation issued at capture.
use crate::{
    conversation::{Conversation, SpokenReply},
    reply::{
        accept_verdict, has_substance, parse_filter_verdict, to_speakable, ChunkLimits,
        FilterVerdict, ReplyChunker,
    },
    turn::{Action, Event, Phase, TurnMachine, TurnTimeouts},
};

/// What Zen says when the detector was sure it heard speech and recognition produced no words
/// for any of it.
pub const UNHEARD: &str = "Sorry, I missed that - could you say it again?";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
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
        /// Which version of the question this repairs. See Session::input_revision.
        revision: u64,
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
    /// A reply was cut off and kept up to that point. `heard` is what was certainly heard of
    /// the phrase playing at the cut, which may be nothing; the page adds it and marks the cut
    /// so the conversation it shows ends where the one the model is given does.
    ShowCut {
        generation: Generation,
        heard: String,
    },
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
    /// The part of the phrase now playing that has certainly been heard. Claimed only if the
    /// reply is cut off before that phrase finishes; see `reply::heard_part`.
    heard_part: String,
    /// Whether this generation's reply belongs in the conversation. A clarification is said
    /// and not kept: it asks about a transcript that was itself not kept.
    keep_reply: bool,
    generation: Generation,
    raw_transcript: String,
    input_open: bool,
    /// Bumped every time the question being transcribed grows, because the speaker paused and
    /// then carried on. Repair is asked about the question as it stood; when it grows, the
    /// answer that comes back is about a question nobody finished asking.
    input_revision: u64,
    filter_pending: bool,
    /// The revision the outstanding repair was asked about.
    filter_revision: u64,
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
            heard_part: String::new(),
            keep_reply: true,
            generation: Generation(0),
            raw_transcript: String::new(),
            input_open: false,
            input_revision: 0,
            filter_pending: false,
            filter_revision: 0,
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
    /// Which version of the question the outstanding repair was asked about.
    ///
    /// The runner carries this through on the task rather than reading it here; it is exposed
    /// so a test can answer the repair that is actually in flight without threading the task
    /// through every call.
    pub fn filter_revision(&self) -> u64 {
        self.filter_revision
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
        self.heard_part.clear();
        self.keep_reply = true;
        self.raw_transcript.clear();
        self.input_revision += 1;
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
                tasks.extend(self.commit_reply(true));
            }
            tasks.push(self.advance());
        } else {
            // The question is still being asked. Any repair already running was asked about the
            // part said so far, which is not the question that will need answering, so it is
            // abandoned here rather than left outstanding. Leaving it pending wedged the turn:
            // the result came back to a phase that would not take it and never cleared the
            // flag, and every transcript after that was refused on behalf of a repair that had
            // already finished. Heard as Zen going silent after a pause mid-sentence, and on
            // any answer long enough to contain one.
            self.input_revision += 1;
            self.filter_pending = false;
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
        // Nothing but "hmm", "uh" or a noise-word is someone thinking, or a hum the detector took
        // for a voice, and not something asked; a person would stay quiet. Passed on, the filter
        // wrote it down as said and Zen answered it.
        if !has_substance(&raw) {
            self.turn.handle(Event::TranscriptEmpty, now_ms);
            return vec![self.advance()];
        }
        self.raw_transcript = raw.trim().to_string();
        self.filter_pending = true;
        self.filter_revision = self.input_revision;
        vec![Task::Filter {
            generation,
            revision: self.filter_revision,
            raw: self.raw_transcript.clone(),
        }]
    }

    /// A typed question, taken as written.
    ///
    /// Repair exists for what recognition gets wrong, and typing has no recognition in it: the
    /// words are exactly the ones meant, and they are already on screen as they were typed. Sent
    /// through the filter anyway, measured in the running app on 2026-09-19, every typed turn
    /// waited 0.29-0.50 s for it before the model was even asked, and the filter was free to
    /// "correct" words nobody had misheard. A question typed in another language is answered in
    /// English all the same: that rule is the talker's own.
    pub fn on_typed(&mut self, generation: Generation, text: String, now_ms: u64) -> Vec<Task> {
        let text = text.trim().to_string();
        if !self.is_current(generation)
            || self.phase() != Phase::Transcribing
            || self.filter_pending
            || text.is_empty()
        {
            return Vec::new();
        }
        self.accept_question(generation, text, now_ms)
    }

    /// The speaker plainly said something, but recognition produced no words for any of it.
    ///
    /// Treated like the empty transcript it looks like, this ended the turn in silence: someone
    /// spoke, and nothing happened. So Zen asks them to say it again - said, and never recorded,
    /// because it asks about words the conversation never received.
    pub fn on_unheard(&mut self, generation: Generation, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation)
            || self.phase() != Phase::Transcribing
            || self.filter_pending
        {
            return Vec::new();
        }
        match self.emit(generation, UNHEARD.to_string()) {
            Some(task) => {
                self.turn.handle(Event::ReplyReady, now_ms);
                self.generation_done = true;
                self.keep_reply = false;
                vec![task]
            }
            None => Vec::new(),
        }
    }

    pub fn on_filter(
        &mut self,
        generation: Generation,
        revision: u64,
        response: String,
        now_ms: u64,
    ) -> Vec<Task> {
        let verdict = parse_filter_verdict(&response);
        self.apply_verdict(generation, revision, verdict, now_ms)
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
    pub fn use_raw_transcript(
        &mut self,
        generation: Generation,
        revision: u64,
        now_ms: u64,
    ) -> Vec<Task> {
        let raw = self.raw_transcript.trim().to_string();
        self.apply_verdict(generation, revision, FilterVerdict::Clean(raw), now_ms)
    }

    fn apply_verdict(
        &mut self,
        generation: Generation,
        revision: u64,
        verdict: FilterVerdict,
        now_ms: u64,
    ) -> Vec<Task> {
        // A repair asked about an older version of the question is not an answer to the current
        // one, however it arrives. Aborting the model job does not unsend a result already on
        // its way, so the identity travels with the request rather than being inferred from the
        // state it comes back to.
        if !self.is_current(generation) || !self.filter_pending || revision != self.filter_revision
        {
            return Vec::new();
        }
        // Whatever happens below, this repair is no longer outstanding. It must never be left
        // set on a path that returns early, or nothing said afterwards can be transcribed.
        self.filter_pending = false;
        if self.phase() != Phase::Transcribing {
            return Vec::new();
        }
        let question = accept_verdict(&self.raw_transcript, verdict);
        self.accept_question(generation, question, now_ms)
    }

    /// The question is settled: remember it, show it, and ask for the answer. One path for
    /// every way a question is accepted, so none of them can answer something the person was
    /// never shown having said.
    fn accept_question(&mut self, generation: Generation, text: String, now_ms: u64) -> Vec<Task> {
        self.conversation.record_user(text.clone());
        let actions = self
            .turn
            .handle(Event::TranscriptReady(text.clone()), now_ms);
        let mut tasks = vec![Task::DisplayTranscript { generation, text }];
        tasks.extend(self.translate(actions));
        tasks
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
            self.heard_part.clear();
            self.reply.played(text);
        }
    }

    /// How much of the phrase now playing has certainly been heard so far.
    pub fn on_heard(&mut self, generation: Generation, part: &str) {
        if self.is_current(generation) && self.pending_phrases > 0 {
            self.heard_part.clear();
            self.heard_part.push_str(part);
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

    /// Cancel a turn without inventing an input utterance: for the stop button, typing, a
    /// reconnected page and a fresh conversation. What was heard of the reply is kept.
    pub fn interrupt(&mut self, now_ms: u64) -> Vec<Task> {
        let mut tasks: Vec<Task> = self.commit_reply(true).into_iter().collect();
        self.turn.handle(Event::Failed(String::new()), now_ms);
        tasks.extend([self.advance(), Task::StopSpeaking]);
        tasks
    }

    /// Writes this generation's reply into the conversation, if it belongs there. An
    /// interrupted reply that was kept comes back as the task that shows the cut.
    fn commit_reply(&mut self, interrupted: bool) -> Option<Task> {
        let mut reply = std::mem::take(&mut self.reply);
        let heard = std::mem::take(&mut self.heard_part);
        // A clarification was never meant to be remembered. Recording it once it had played
        // left the model an answer to a question it was never shown - and, after a complete
        // reply, two assistant turns in a row.
        if !self.keep_reply {
            return None;
        }
        if interrupted {
            reply.played(heard.clone());
        }
        // Nothing was heard in answer to this question, so there is no assistant turn to write:
        // recording an empty one would tell the model it answered when the listener heard
        // silence. The question itself stays. It used to be deleted here to stop unanswered
        // questions stacking up, which also meant that speaking again before the first phrase
        // was acknowledged erased what had just been asked - the conversation forgot something
        // the person had actually said, within the same session. The stack is prevented in
        // `Conversation::record_user` instead, by carrying the question into the next turn.
        if reply.is_empty() {
            return None;
        }
        self.conversation.record_reply(reply, interrupted);
        interrupted.then_some(Task::ShowCut {
            generation: self.generation,
            heard,
        })
    }

    pub fn on_failure(&mut self, generation: Generation, reason: String, now_ms: u64) -> Vec<Task> {
        if !self.is_current(generation) {
            return Vec::new();
        }
        let mut tasks: Vec<Task> = self.commit_reply(true).into_iter().collect();
        let actions = self.turn.handle(Event::Failed(reason), now_ms);
        tasks.push(self.advance());
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
        let mut tasks: Vec<Task> = self.commit_reply(true).into_iter().collect();
        tasks.extend([self.advance(), Task::StopSpeaking]);
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
        let tasks = session.use_raw_transcript(g, session.filter_revision(), 300);
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
        session.on_filter(g, session.filter_revision(), "CLEAN: hello".into(), 300);
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
        assert!(phrases.len() > 3, "got {} phrases", phrases.len());
        assert!(phrases
            .iter()
            .all(|p| p.split_whitespace().count() <= ChunkLimits::default().hard));
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
        session.on_filter(
            generation,
            session.filter_revision(),
            "CLEAN: hello there".into(),
            at + 300,
        );
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
    fn a_typed_question_is_asked_as_written_without_repair() {
        // Typing has no recognition in it, so there is nothing to repair: the words go to the
        // model and onto the page exactly as typed, with no filter request in between.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(0);
        let generation = session.generation();
        let tasks = session.on_typed(generation, "  wut is teh capital of Japan?  ".into(), 10);
        assert!(!tasks.iter().any(|task| matches!(task, Task::Filter { .. })));
        assert!(tasks.contains(&Task::DisplayTranscript {
            generation,
            text: "wut is teh capital of Japan?".into()
        }));
        assert!(tasks.iter().any(|task| matches!(task, Task::Reply { .. })));
        assert_eq!(session.phase(), Phase::Thinking);
        assert_eq!(
            session
                .conversation()
                .turns()
                .last()
                .map(|turn| turn.text.as_str()),
            Some("wut is teh capital of Japan?")
        );
        // Nothing typed is nothing asked.
        let mut session = self::session();
        session.on_speech(0);
        session.on_turn_ended(0);
        assert!(session
            .on_typed(session.generation(), "   ".into(), 10)
            .is_empty());
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

        let tasks = session.on_filter(
            generation,
            session.filter_revision(),
            "CLEAN: turn on the light".into(),
            950,
        );
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
    fn a_request_to_repeat_words_that_were_said_is_overruled() {
        // Anything that reaches the filter has words in it, and asking someone to repeat words
        // they said gets the same words back. The turn goes ahead on what was heard.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "kitch en the lie".into(), 200);
        let tasks = session.on_filter(
            generation,
            session.filter_revision(),
            "ASK: Sorry, did you mean the kitchen light?".into(),
            300,
        );

        assert!(tasks.contains(&Task::DisplayTranscript {
            generation,
            text: "kitch en the lie".into()
        }));
        assert!(tasks.iter().any(|task| matches!(task, Task::Reply { .. })));
        assert!(!tasks.iter().any(|task| matches!(task, Task::Speak { .. })));
    }

    #[test]
    fn nothing_but_hesitation_or_noise_ends_the_turn_in_silence() {
        // Each one was answered before, as a question or with a request to repeat it.
        for heard in [
            "Hmm.",
            "mm uh hm",
            "Um... uhh.",
            "Hmmmm?",
            "Er, erm.",
            "mmph",
            "Shh.",
        ] {
            let mut session = session();
            session.on_speech(0);
            session.on_turn_ended(100);
            let generation = session.generation();
            let tasks = session.on_transcript(generation, heard.into(), 200);
            assert!(
                !tasks.iter().any(|task| matches!(
                    task,
                    Task::Filter { .. } | Task::Speak { .. } | Task::Reply { .. }
                )),
                "{heard:?} was taken as something said: {tasks:?}"
            );
            assert_ne!(session.phase(), Phase::Transcribing, "{heard:?}");
            assert_eq!(session.conversation().turn_count(), 0);
        }
    }

    #[test]
    fn hesitation_with_words_in_it_is_still_a_question() {
        for heard in [
            "Hmm, what about tomorrow?",
            "Uh-huh.",
            "Oh.",
            "Um, stop.",
            "Zen?",
        ] {
            let mut session = session();
            session.on_speech(0);
            session.on_turn_ended(100);
            let generation = session.generation();
            let tasks = session.on_transcript(generation, heard.into(), 200);
            assert!(
                tasks.iter().any(|task| matches!(task, Task::Filter { .. })),
                "{heard:?} was dropped"
            );
        }
    }

    #[test]
    fn a_clarification_with_nothing_in_it_answers_the_question_and_shows_it() {
        // The model sometimes answers with a bare marker. The turn goes ahead on the transcript,
        // and the transcript has to be shown like any other accepted question.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "what time is it".into(), 200);
        let tasks = session.on_filter(generation, session.filter_revision(), "ASK:".into(), 300);
        assert!(tasks.contains(&Task::DisplayTranscript {
            generation,
            text: "what time is it".into()
        }));
        assert!(tasks.iter().any(|task| matches!(task, Task::Reply { .. })));
        assert_eq!(session.phase(), Phase::Thinking);
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
            session.filter_revision(),
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
        session.on_filter(
            generation,
            session.filter_revision(),
            "CLEAN: how many".into(),
            300,
        );

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
    fn a_reply_cut_off_mid_phrase_keeps_the_clauses_that_were_heard() {
        // A phrase can run for half a minute. Cut off part-way, the listener has heard its
        // opening, and the model has to be told so or it starts the same answer again.
        let mut session = session();
        let (generation, _) = to_speaking(&mut session, 0);
        session.on_heard(generation, "Hello there");
        let tasks = session.interrupt(900);
        let last = session.conversation().turns().last().unwrap();
        assert_eq!(last.as_prompt_text(), "Hello there —");
        // And the page is told, before the cancellation, so what it shows matches.
        assert_eq!(
            tasks.first(),
            Some(&Task::ShowCut {
                generation,
                heard: "Hello there".into()
            })
        );
    }

    #[test]
    fn a_reply_nobody_heard_shows_no_cut() {
        let mut session = session();
        to_speaking(&mut session, 0);
        let tasks = session.interrupt(900);
        assert!(!tasks.iter().any(|t| matches!(t, Task::ShowCut { .. })));
    }

    #[test]
    fn a_clarification_is_said_but_never_kept() {
        // Documented as spoken and never recorded, it was recorded anyway once it finished
        // playing - an answer to a question the model was never shown.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        let tasks = session.on_unheard(generation, 200);
        let spoken = tasks
            .iter()
            .find_map(|t| match t {
                Task::Speak { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap();
        session.on_playback_started(generation, 400);
        session.on_heard(generation, "Sorry");
        session.on_spoken(generation, spoken);
        session.on_playback_finished(generation, 900);
        assert_eq!(session.conversation().turn_count(), 0);

        // Nor when it is cut off.
        session.on_speech(1_000);
        session.on_turn_ended(1_100);
        let generation = session.generation();
        session.on_unheard(generation, 1_200);
        session.on_playback_started(generation, 1_300);
        session.on_heard(generation, "Sorry, I missed that");
        let tasks = session.interrupt(1_400);
        assert_eq!(session.conversation().turn_count(), 0);
        assert!(!tasks.iter().any(|t| matches!(t, Task::ShowCut { .. })));
    }

    #[test]
    fn a_finished_phrase_replaces_the_estimate_of_it() {
        let mut session = session();
        let (generation, spoken) = to_speaking(&mut session, 0);
        session.on_heard(generation, "Hello there");
        session.on_spoken(generation, spoken.clone());
        session.on_reply_complete(generation, 800);
        session.on_playback_finished(generation, 900);
        let last = session.conversation().turns().last().unwrap();
        assert!(!last.interrupted);
        assert_eq!(
            last.as_prompt_text(),
            "Hello there and how can I help you today?"
        );
    }

    #[test]
    fn what_was_heard_of_an_abandoned_reply_is_not_credited_to_the_next() {
        let mut session = session();
        let (old, _) = to_speaking(&mut session, 0);
        session.interrupt(700);
        session.on_heard(old, "Stale words.");
        to_speaking(&mut session, 1_000);
        session.interrupt(1_700);
        assert!(session
            .conversation()
            .turns()
            .all(|turn| !turn.text.contains("Stale")));
    }

    #[test]
    fn interrupting_before_any_audio_played_keeps_what_was_asked() {
        // Cutting in while the model is still thinking must not cost the listener what they
        // already said. Nothing was heard in answer, so there is no assistant turn to write -
        // but the question itself was accepted and belongs in the conversation. It used to be
        // deleted here, so speaking again before the first phrase was acknowledged erased the
        // question entirely and Zen had no idea it had ever been asked.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "my sister is called Mira".into(), 200);
        session.on_filter(
            generation,
            session.filter_revision(),
            "CLEAN: my sister is called Mira".into(),
            300,
        );
        session.on_speech(400);
        let said: Vec<_> = session
            .conversation()
            .messages()
            .into_iter()
            .filter(|(role, _)| *role == "user")
            .map(|(_, text)| text)
            .collect();
        assert_eq!(said, ["my sister is called Mira"]);
    }

    #[test]
    fn clear_speech_with_no_words_asks_again_instead_of_going_silent() {
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        let tasks = session.on_unheard(generation, 200);
        assert_eq!(
            tasks,
            [Task::Speak {
                generation,
                text: to_speakable(UNHEARD),
            }],
            "the listener has to hear that they were not understood",
        );
        // Nothing was said by anyone, so nothing is remembered.
        assert_eq!(session.conversation().turn_count(), 0);
        // And a stale report cannot speak over a newer turn.
        session.on_speech(300);
        assert!(session.on_unheard(generation, 400).is_empty());
    }

    #[test]
    fn carrying_on_after_a_pause_does_not_wedge_the_turn() {
        // Reported as: talk, pause briefly, carry on - and Zen stops responding. Also on any
        // answer long enough to contain a pause.
        //
        // Repair of what had been said so far was still running when speech resumed. Its result
        // came back to a phase that would not take it, and the early return left the pending
        // flag set, so the transcript of the finished question was refused on behalf of a repair
        // that had already completed. The turn stayed in Transcribing until it timed out.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "what is the weather".into(), 200);
        let stale = session.filter_revision();

        // The speaker carries on before the repair comes back.
        session.on_speech(300);
        session.on_turn_ended(1_000);
        assert_eq!(session.phase(), Phase::Transcribing);

        // The repair of the first half arrives now. It answers a question nobody finished
        // asking, so it must be discarded - and it must not leave anything behind.
        assert!(session
            .on_filter(
                generation,
                stale,
                "CLEAN: What is the weather?".into(),
                1_100
            )
            .is_empty());

        // The whole question, now that they have actually stopped.
        let tasks = session.on_transcript(
            generation,
            "what is the weather in London tomorrow".into(),
            1_200,
        );
        let revision = match tasks.as_slice() {
            [Task::Filter { revision, raw, .. }] => {
                assert_eq!(raw, "what is the weather in London tomorrow");
                *revision
            }
            other => panic!("the finished question must be repaired, got {other:?}"),
        };
        assert_ne!(
            revision, stale,
            "and as a new request, not the abandoned one"
        );

        let tasks = session.on_filter(
            generation,
            revision,
            "CLEAN: What is the weather in London tomorrow?".into(),
            1_300,
        );
        assert!(
            tasks.iter().any(|task| matches!(task, Task::Reply { .. })),
            "the turn has to reach an answer: {tasks:?}",
        );
        assert_eq!(session.phase(), Phase::Thinking);
    }

    #[test]
    fn a_repair_of_an_abandoned_question_cannot_answer_the_finished_one() {
        // Aborting the model job does not unsend a result already on its way, so a stale repair
        // can arrive after the new one has been requested. Without an identity of its own it
        // would be taken for the answer to the current question and the wrong text spoken.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let generation = session.generation();
        session.on_transcript(generation, "book a table".into(), 200);
        let stale = session.filter_revision();
        session.on_speech(300);
        session.on_turn_ended(1_000);
        session.on_transcript(generation, "book a table for four".into(), 1_100);
        let current = session.filter_revision();
        assert_ne!(current, stale);

        // The abandoned repair lands late, while the current one is outstanding.
        assert!(session
            .on_filter(generation, stale, "CLEAN: Book a table.".into(), 1_150)
            .is_empty());
        let tasks = session.on_filter(
            generation,
            current,
            "CLEAN: Book a table for four.".into(),
            1_200,
        );
        let asked = tasks
            .iter()
            .find_map(|task| match task {
                Task::Reply { messages, .. } => messages
                    .iter()
                    .rev()
                    .find(|(role, _)| *role == "user")
                    .map(|(_, text)| text.clone()),
                _ => None,
            })
            .expect("the finished question is answered");
        assert_eq!(asked, "Book a table for four.");
    }

    #[test]
    fn something_said_before_an_interruption_is_still_there_to_be_asked_about() {
        // The complaint this exists for: tell Zen something, cut in before the first phrase of
        // its answer is acknowledged, then ask about what you said - and it had no idea. The
        // question had been deleted from the window on the grounds that it was unanswered, so
        // the fact in it was gone from the conversation entirely.
        let mut session = session();
        session.on_speech(0);
        session.on_turn_ended(100);
        let first = session.generation();
        session.on_transcript(first, "my sister is called Mira".into(), 200);
        session.on_filter(
            first,
            session.filter_revision(),
            "CLEAN: my sister is called Mira".into(),
            300,
        );
        // Cut in before anything was heard.
        session.on_speech(400);
        session.on_turn_ended(500);
        let second = session.generation();
        session.on_transcript(second, "what is my sister called".into(), 600);
        let tasks = session.on_filter(
            second,
            session.filter_revision(),
            "CLEAN: what is my sister called".into(),
            700,
        );

        let messages = tasks
            .into_iter()
            .find_map(|task| match task {
                Task::Reply { messages, .. } => Some(messages),
                _ => None,
            })
            .expect("the question is answered");
        assert!(
            messages.iter().any(|(_, text)| text.contains("Mira")),
            "the fact has to reach the model: {messages:?}",
        );
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
            session.on_filter(
                generation,
                session.filter_revision(),
                format!("CLEAN: {question}"),
                at + 300,
            );
            assert_eq!(session.phase(), Phase::Thinking);
        }
        session.on_speech(2_000);
        session.on_turn_ended(2_100);
        let generation = session.generation();
        session.on_transcript(generation, "hello".into(), 2_200);
        let tasks = session.on_filter(
            generation,
            session.filter_revision(),
            "CLEAN: hello".into(),
            2_300,
        );

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
            ["tell me a lion story
tell me a life story
hello"],
            "abandoned questions are carried into the next turn, not stacked beside it"
        );
    }
}
