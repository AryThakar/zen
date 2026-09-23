// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Rolling conversation for the reply slot.
//!
//! The invariant this module exists to hold:
//!
//! > **The transcript records what the user *heard*, never what the model *generated*.**
//!
//! When a reply is interrupted, the model may have produced two hundred words while only forty
//! reached the speaker. Storing all two hundred leaves the model believing it said things the
//! user never heard - so it will not repeat them, will refer back to them, and will answer
//! follow-up questions as though shared ground exists that does not. The conversation quietly
//! desynchronises from reality, and every later turn is built on that gap.
//!
//! So an assistant turn is assembled from the chunks text-to-speech actually played, committed as
//! they play, and truncated at whatever point the interruption landed.
//!
//! The second concern is the reply slot's KV cache ([`crate::engine::SlotKind::Talker`]). A
//! byte-identical prompt prefix across turns lets `llama-server` reuse the cached prefix instead
//! of re-prefilling the conversation - a stall the user hears as silence before the assistant
//! speaks. Two rules follow, and most of the design below is downstream of them:
//!
//! - the system prompt is fixed and never carries dynamic state,
//! - history is evicted in occasional large steps rather than trimmed every turn, because every
//!   eviction moves the boundary and costs a full re-prefill.

use std::collections::VecDeque;

/// Who said it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    User,
    Assistant,
}

impl Speaker {
    pub const fn role(self) -> &'static str {
        match self {
            Speaker::User => "user",
            Speaker::Assistant => "assistant",
        }
    }
}

/// One completed turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub speaker: Speaker,
    /// For the assistant, this is what was *spoken*, not what was generated.
    pub text: String,
    /// Whether the user cut this turn off partway through.
    pub interrupted: bool,
}

impl Utterance {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            speaker: Speaker::User,
            text: text.into(),
            interrupted: false,
        }
    }

    /// Text as the model should see it.
    ///
    /// An interrupted turn is marked rather than presented as a complete thought. Without the
    /// marker the model reads its own truncated sentence as finished and continues from a
    /// half-expressed idea; with it, it can see that it was cut off and pick the thread back up.
    ///
    /// The text before the dash is only what was actually heard, up to the last chunk the device
    /// acknowledged - never what was generated. The dash is the marker `core.txt` describes.
    pub fn as_prompt_text(&self) -> String {
        if self.interrupted && !self.text.is_empty() {
            format!("{} —", self.text.trim_end())
        } else {
            self.text.clone()
        }
    }

    /// Rough token cost. The engine's tokenizer is authoritative; this is for the cheap pass.
    pub fn estimated_tokens(&self) -> usize {
        // Four characters per token is the usual English approximation, plus a few for the role
        // and message framing the chat template adds.
        self.text.len() / 4 + 4
    }
}

/// An assistant reply while it is being spoken.
///
/// Chunks are committed here as text-to-speech plays them, so an interruption at any moment
/// leaves behind exactly what was heard.
#[derive(Debug, Clone, Default)]
pub struct SpokenReply {
    chunks: Vec<String>,
    /// The model's own words for the phrases handed to synthesis, before they were rewritten
    /// for the ear.
    generated: String,
}

impl SpokenReply {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one chunk as having reached the speaker.
    ///
    /// Called when playback of that chunk *finishes*, not when synthesis finishes. Synthesis runs
    /// ahead of playback by design, so recording on synthesis would credit the assistant with
    /// speech the user has not heard yet - which is the exact error this module prevents.
    pub fn played(&mut self, chunk: impl Into<String>) {
        let chunk = chunk.into();
        if !chunk.trim().is_empty() {
            self.chunks.push(chunk);
        }
    }

    /// Records the model's own text as it arrives, before any of it is made speakable.
    ///
    /// A completed reply is stored in the conversation exactly as the model wrote it. The
    /// spoken form differs - dashes become commas, numbers become words - and storing that
    /// instead hands the model tokens it never produced, which costs the server its cached
    /// prefix and makes it re-read the whole conversation on every single turn. Measured on a
    /// short history that was 858 tokens re-processed instead of 16.
    pub fn generated(&mut self, text: &str) {
        self.generated.push_str(text);
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.iter().all(|chunk| chunk.trim().is_empty())
    }

    /// Everything heard so far, joined.
    pub fn spoken_text(&self) -> String {
        let mut text = String::new();
        for chunk in &self.chunks {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(chunk);
        }
        text
    }

    /// Seals the reply into a turn.
    pub fn finish(self, interrupted: bool) -> Utterance {
        // Interrupted, only what was actually heard can be claimed - and the model has to see
        // where it was cut off. Complete, the two say the same thing, so the model's own
        // wording is stored because it is the one the server has already cached.
        let text = if interrupted || self.generated.trim().is_empty() {
            self.spoken_text()
        } else {
            self.generated.trim().to_string()
        };
        Utterance {
            speaker: Speaker::Assistant,
            text,
            interrupted,
        }
    }
}

/// Limits for the rolling window.
#[derive(Debug, Clone, Copy)]
pub struct WindowBudget {
    /// Total tokens the slot holds.
    pub capacity_tokens: usize,
    /// Held back for the reply.
    pub reply_tokens: usize,
    /// Held back for template overhead and counting error.
    pub safety_tokens: usize,
    /// Recent turns preferred during eviction, provided the hard budget still fits.
    ///
    /// The immediate exchange is what the next reply is about; dropping it to fit is worse than
    /// dropping older history.
    pub protected_turns: usize,
}

impl WindowBudget {
    /// Sized for one 8k slot.
    pub fn for_slot(capacity_tokens: usize) -> Self {
        Self {
            capacity_tokens,
            reply_tokens: 512,
            safety_tokens: 256,
            protected_turns: 4,
        }
    }

    /// Tokens available to the prompt.
    pub fn prompt_limit(&self) -> usize {
        self.capacity_tokens
            .saturating_sub(self.reply_tokens)
            .saturating_sub(self.safety_tokens)
    }

    /// How far down to evict, once evicting at all.
    ///
    /// Every eviction moves the prompt's prefix, which throws away the model server's cached
    /// KV and makes the next reply re-read the whole window before it can speak. The cost of
    /// a long conversation is therefore not how much gets evicted, it is how often. Trimming
    /// just past the limit leaves the window still full, so the next turn evicts again and
    /// the stall lands turn after turn. Dropping to half spends it once and buys a long quiet
    /// stretch, and the re-read it forces is half as long as well.
    pub fn eviction_target(&self) -> usize {
        self.prompt_limit() / 2
    }
}

/// What changed after recording a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowChange {
    pub evicted_turns: usize,
}

/// The reply slot's rolling conversation.
#[derive(Debug)]
pub struct Conversation {
    system: String,
    turns: VecDeque<Utterance>,
    budget: WindowBudget,
    evictions: usize,
}

impl Conversation {
    pub fn new(system: impl Into<String>, budget: WindowBudget) -> Self {
        Self {
            system: system.into(),
            turns: VecDeque::new(),
            budget,
            evictions: 0,
        }
    }

    /// Replace the instructions. Paired with `clear_history` this lets the persona
    /// change without unloading the model, which is the difference between starting
    /// fresh instantly and waiting for five gigabytes to load again.
    pub fn set_system(&mut self, system: impl Into<String>) {
        self.system = system.into();
    }

    /// Forget everything said so far and keep the instructions. This is the rolling
    /// window being emptied, not the session ending: the model stays loaded, so
    /// starting fresh costs nothing but the history itself.
    pub fn clear_history(&mut self) {
        self.turns.clear();
        self.evictions = 0;
    }

    /// The fixed prefix. Never carries timestamps, task state, or anything else that changes.
    pub fn system(&self) -> &str {
        &self.system
    }

    pub fn turns(&self) -> impl Iterator<Item = &Utterance> {
        self.turns.iter()
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn evictions(&self) -> usize {
        self.evictions
    }

    /// Longest a carried-forward user turn may grow, in characters. Generous - a minute of
    /// speech is roughly nine hundred - so it only ever trims a pathological run of barge-ins.
    const CARRIED_LIMIT: usize = 4000;

    /// Records what the person said, carrying forward anything they asked that was never
    /// answered.
    ///
    /// A user turn is stored as soon as the transcript is accepted, before any reply exists.
    /// If that turn is then abandoned - they cut in and asked something else, or the model
    /// failed - the question is left with nothing after it. Two or three of those and the
    /// window holds a stack of questions, so the next reply is composed against all of them at
    /// once and answers whichever the model finds most salient, which is rarely the one just
    /// asked.
    ///
    /// The old fix was to delete the unanswered question, which prevented the stack by
    /// throwing away what the person had told Zen: ask a question, speak again before the
    /// first phrase is acknowledged, and the first question was gone from the conversation
    /// entirely. Joining it to the next turn prevents the same stack without losing anything.
    /// They did say both things, in this order, and one user turn per exchange is what keeps
    /// the newest question the one being answered.
    pub fn record_user(&mut self, text: impl Into<String>) -> WindowChange {
        let text = text.into();
        if let Some(previous) = self.turns.back_mut() {
            if previous.speaker == Speaker::User {
                previous.text.push('\n');
                previous.text.push_str(&text);
                // Barge-in after barge-in must not grow one turn without limit. The newest
                // words are the ones being answered, so it is the oldest that go.
                let length = previous.text.chars().count();
                // Bound carried history, never trim the beginning of the new question.
                let limit = Self::CARRIED_LIMIT.max(text.chars().count());
                if length > limit {
                    previous.text = previous.text.chars().skip(length - limit).collect();
                }
                return self.enforce_budget();
            }
        }
        self.push(Utterance::user(text))
    }

    /// Commits a reply, storing only what was spoken.
    pub fn record_reply(&mut self, reply: SpokenReply, interrupted: bool) -> WindowChange {
        // A reply cut off before a single chunk played leaves no trace. Recording an empty
        // assistant turn would tell the model it answered when the user heard nothing.
        if reply.is_empty() {
            return WindowChange::default();
        }
        self.push(reply.finish(interrupted))
    }

    fn push(&mut self, utterance: Utterance) -> WindowChange {
        self.turns.push_back(utterance);
        self.enforce_budget()
    }

    /// Estimated prompt cost, system prompt included.
    pub fn estimated_tokens(&self) -> usize {
        self.system.len() / 4
            + 4
            + self
                .turns
                .iter()
                .map(Utterance::estimated_tokens)
                .sum::<usize>()
    }

    fn enforce_budget(&mut self) -> WindowChange {
        if self.estimated_tokens() <= self.budget.prompt_limit() {
            return WindowChange::default();
        }
        let target = self.budget.eviction_target();
        let mut evicted = 0;
        // Evict down past the limit to the target, so the next several turns fit without moving
        // the boundary again.
        while self.turns.len() > 1
            && ((self.turns.len() > self.budget.protected_turns
                && self.estimated_tokens() > target)
                || self.estimated_tokens() > self.budget.prompt_limit())
        {
            self.turns.pop_front();
            evicted += 1;
        }
        // An assistant message without its user question is not a valid retained exchange.
        while self.turns.len() > 1
            && self
                .turns
                .front()
                .is_some_and(|t| t.speaker == Speaker::Assistant)
        {
            self.turns.pop_front();
            evicted += 1;
        }
        if evicted > 0 {
            self.evictions += 1;
        }
        WindowChange {
            evicted_turns: evicted,
        }
    }

    /// The prompt, as chat messages.
    pub fn messages(&self) -> Vec<(&'static str, String)> {
        let mut messages = Vec::with_capacity(self.turns.len() + 1);
        messages.push(("system", self.system.clone()));
        for turn in &self.turns {
            messages.push((turn.speaker.role(), turn.as_prompt_text()));
        }
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation() -> Conversation {
        Conversation::new("You are Zen.", WindowBudget::for_slot(8_192))
    }

    #[test]
    fn a_completed_reply_is_stored_in_full() {
        let mut chat = conversation();
        chat.record_user("what is the weather");
        let mut reply = SpokenReply::new();
        reply.played("It is sunny today.");
        reply.played("Around twenty degrees.");
        chat.record_reply(reply, false);

        let last = chat.turns().last().unwrap();
        assert_eq!(last.text, "It is sunny today. Around twenty degrees.");
        assert!(!last.interrupted);
    }

    #[test]
    fn an_interrupted_reply_stores_only_what_was_played() {
        // The invariant this module exists for. The model generated three sentences; the user cut
        // in after one. Storing all three would leave the model believing it told the user things
        // they never heard.
        let mut chat = conversation();
        chat.record_user("tell me about mars");
        let mut reply = SpokenReply::new();
        reply.played("Mars is the fourth planet.");
        // Two more chunks were generated and synthesised, but never reached the speaker.
        chat.record_reply(reply, true);

        let last = chat.turns().last().unwrap();
        assert_eq!(last.text, "Mars is the fourth planet.");
        assert!(last.interrupted);
    }

    #[test]
    fn an_interrupted_turn_is_marked_so_the_model_knows_it_was_cut_off() {
        // Without a marker the model reads its own truncated sentence as a finished thought and
        // continues from a half-expressed idea.
        let mut reply = SpokenReply::new();
        reply.played("I was going to say");
        let turn = reply.finish(true);
        assert!(turn.as_prompt_text().ends_with('—'));
    }

    #[test]
    fn a_completed_turn_carries_no_marker() {
        let mut reply = SpokenReply::new();
        reply.played("All done.");
        let turn = reply.finish(false);
        assert_eq!(turn.as_prompt_text(), "All done.");
    }

    #[test]
    fn a_reply_interrupted_before_any_audio_played_leaves_no_turn() {
        // The user cut in during thinking, before a word was spoken. Recording an empty assistant
        // turn would tell the model it answered when the user heard nothing at all.
        let mut chat = conversation();
        chat.record_user("hello");
        let before = chat.turn_count();
        chat.record_reply(SpokenReply::new(), true);
        assert_eq!(chat.turn_count(), before);
    }

    #[test]
    fn chunks_are_credited_on_playback_not_on_synthesis() {
        // Synthesis runs ahead of playback by design. Crediting on synthesis would record speech
        // the user has not heard yet, which is the exact error being prevented.
        let mut reply = SpokenReply::new();
        reply.played("first");
        // "second" was synthesised here but playback never reached it.
        assert_eq!(reply.spoken_text(), "first");
    }

    #[test]
    fn blank_chunks_do_not_create_stray_spacing() {
        let mut reply = SpokenReply::new();
        reply.played("hello");
        reply.played("   ");
        reply.played("world");
        assert_eq!(reply.spoken_text(), "hello world");
    }

    #[test]
    fn the_system_prompt_leads_every_prompt_and_never_changes() {
        // The reply slot reuses its cached prefix only while these bytes are identical, so
        // anything dynamic here costs a full re-prefill every turn.
        let mut chat = conversation();
        let before = chat.messages()[0].clone();
        chat.record_user("hello");
        chat.record_reply(
            {
                let mut reply = SpokenReply::new();
                reply.played("hi");
                reply
            },
            false,
        );
        assert_eq!(chat.messages()[0], before);
        assert_eq!(before.0, "system");
    }

    #[test]
    fn an_ordinary_turn_evicts_nothing() {
        let mut chat = conversation();
        let change = chat.record_user("hello");
        assert_eq!(change, WindowChange::default());
    }

    #[test]
    fn carrying_an_unanswered_turn_never_trims_the_new_question() {
        let mut chat = conversation();
        chat.record_user("An earlier unanswered question");
        let newest = format!("Keep this opening detail. {}", "word ".repeat(1000));
        chat.record_user(newest.clone());
        assert_eq!(chat.turns().last().unwrap().text, newest);
    }

    #[test]
    fn a_full_window_evicts_the_oldest_turns() {
        let mut chat = Conversation::new("sys", WindowBudget::for_slot(2_048));
        let long = "word ".repeat(200);
        let mut evicted_at_least_once = false;
        for _ in 0..40 {
            chat.record_user(long.clone());
            let mut reply = SpokenReply::new();
            reply.played(&long);
            let change = chat.record_reply(reply, false);
            evicted_at_least_once |= change.evicted_turns > 0;
        }
        assert!(evicted_at_least_once);
        assert!(chat.estimated_tokens() <= chat.budget.prompt_limit());
    }

    #[test]
    fn eviction_happens_in_occasional_large_steps_not_every_turn() {
        // Each eviction moves the prefix boundary and costs a full re-prefill, heard as silence
        // before the assistant speaks. Trimming to the brim every turn would pay that constantly.
        let mut chat = Conversation::new("sys", WindowBudget::for_slot(2_048));
        let long = "word ".repeat(100);
        for _ in 0..60 {
            chat.record_user(long.clone());
        }
        assert!(
            chat.evictions() < 20,
            "evicted on {} occasions across 60 turns; the slack is not working",
            chat.evictions()
        );
    }

    #[test]
    fn recent_turns_cannot_override_the_hard_budget() {
        // The immediate exchange is what the next reply is about.
        let mut chat = Conversation::new("sys", WindowBudget::for_slot(2_048));
        let long = "word ".repeat(500);
        for _ in 0..30 {
            chat.record_user(long.clone());
            let mut reply = SpokenReply::new();
            reply.played(&long);
            chat.record_reply(reply, false);
        }
        assert!(chat.estimated_tokens() <= chat.budget.prompt_limit());
        assert!(chat.turn_count() > 0);
    }

    #[test]
    fn eviction_reports_that_the_cached_prefix_is_dead() {
        // The caller needs to know: this turn pays a re-prefill, and that is worth reporting
        // rather than silently absorbing.
        let mut chat = Conversation::new("sys", WindowBudget::for_slot(2_048));
        let long = "word ".repeat(300);
        let mut saw_invalidation = false;
        for _ in 0..30 {
            chat.record_user(long.clone());
            let mut reply = SpokenReply::new();
            reply.played(&long);
            saw_invalidation |= chat.record_reply(reply, false).evicted_turns > 0;
        }
        assert!(saw_invalidation);
    }

    #[test]
    fn the_interrupted_then_resumed_pattern_reads_correctly() {
        // The scenario as described: speak, pause, answer, interrupted, speak again. What the
        // model sees must match what the user actually experienced.
        let mut chat = conversation();
        chat.record_user("tell me a long story");
        let mut reply = SpokenReply::new();
        reply.played("Once upon a time there was a village.");
        chat.record_reply(reply, true);
        chat.record_user("actually make it shorter");

        let messages = chat.messages();
        assert_eq!(messages[0].0, "system");
        assert_eq!(messages[1].1, "tell me a long story");
        assert!(messages[2].1.starts_with("Once upon a time"));
        assert!(messages[2].1.ends_with('—'), "the cut-off must be visible");
        assert_eq!(messages[3].1, "actually make it shorter");
    }

    #[test]
    fn the_prompt_limit_leaves_room_for_a_reply() {
        let budget = WindowBudget::for_slot(8_192);
        assert!(budget.prompt_limit() < 8_192);
        assert!(budget.eviction_target() < budget.prompt_limit());
    }

    #[test]
    fn a_full_window_evicts_in_one_large_step_rather_than_every_turn() {
        // Each eviction costs a re-prefill the speaker hears as silence, so what matters is
        // how many happen across a long conversation, not how many turns each one drops.
        // Shaving the window back to just under its limit would evict on nearly every turn
        // from then on; going to half must not.
        let mut chat = Conversation::new("You are Zen.", WindowBudget::for_slot(8_192));
        let question = "word ".repeat(40);
        let answer = "reply ".repeat(60);
        let mut turns_after_first_eviction = 0;
        let mut evictions_after_first = 0;
        for _ in 0..200 {
            let evicted = chat.record_user(question.clone()).evicted_turns > 0;
            let mut reply = SpokenReply::new();
            reply.played(&answer);
            let evicted = evicted || chat.record_reply(reply, false).evicted_turns > 0;
            if chat.evictions() > 0 {
                turns_after_first_eviction += 1;
                evictions_after_first += usize::from(evicted);
            }
        }
        assert!(
            turns_after_first_eviction > 100,
            "the window must fill well before the conversation ends, or this proves nothing"
        );
        // Once evicting, a window trimmed to its limit evicts about every other turn. Half a
        // window of slack has to buy far more than that.
        let turns_per_eviction = turns_after_first_eviction / evictions_after_first.max(1);
        assert!(
            turns_per_eviction >= 10,
            "evicted once every {turns_per_eviction} turns ({evictions_after_first} times in {turns_after_first_eviction} turns); halving the window should buy many quiet turns"
        );
    }

    #[test]
    fn eviction_keeps_enough_history_to_stay_in_the_conversation() {
        // Half a window is a real trade: it has to leave enough turns behind that Zen still
        // knows what is being discussed.
        let mut chat = Conversation::new("You are Zen.", WindowBudget::for_slot(8_192));
        let question = "word ".repeat(40);
        let answer = "reply ".repeat(60);
        for _ in 0..200 {
            chat.record_user(question.clone());
            let mut reply = SpokenReply::new();
            reply.played(&answer);
            chat.record_reply(reply, false);
        }
        assert!(chat.evictions() > 0, "the window must have filled");
        assert!(
            chat.turn_count() >= 12,
            "only {} turns survived eviction",
            chat.turn_count()
        );
    }
}
