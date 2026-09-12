// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Whole-turn flows through the public `Session` API, with no models or audio devices:
//! recognition, generation and playback acknowledgements are fed in by hand so each ordering
//! and interruption can be driven deterministically.
use zen::{
    conversation::{Conversation, WindowBudget},
    input::Utterance,
    reply::ChunkLimits,
    session::{Generation, Session, Task},
    turn::{Phase, TurnTimeouts},
};
fn session() -> Session {
    Session::new(
        Conversation::new("You are Zen.", WindowBudget::for_slot(8192)),
        TurnTimeouts::default(),
        ChunkLimits::default(),
    )
}
/// A first stretch long enough to cross the chunker's opening gate, so it is handed to
/// synthesis while generation is still running. Short replies are deliberately held and
/// spoken whole, which is a different path and not the one these tests are about.
// Long enough to open the first-chunk gate, which now sits above an ordinary reply so that
// most replies are spoken whole rather than assembled out of separately synthesised pieces.
const LONG_FIRST: &str = "There is a lion out on the ridge at dusk and the whole herd below \
him has gone completely still, which is the part people never quite expect to hear about. They \
do not scatter and they do not run, because running is what gets noticed, and the one thing \
none of them wants tonight is to be the animal that moved first. ";
const TAIL: &str = "He does not need to move at all. ";

fn speak_tasks(tasks: Vec<Task>) -> Vec<String> {
    tasks
        .into_iter()
        .filter_map(|t| match t {
            Task::Speak { text, .. } => Some(text),
            _ => None,
        })
        .collect()
}

fn ask(s: &mut Session, text: &str, at: u64) -> (Generation, Vec<Task>) {
    s.on_speech(at);
    let g = s.generation();
    s.on_turn_ended(at + 700);
    s.on_transcript(g, text.into(), at + 800);
    (g, s.on_filter(g, format!("CLEAN: {text}"), at + 900))
}

#[test]
fn fast_asr_is_retained_until_endpoint_then_delivered_as_one_complete_question() {
    let mut s = session();
    let mut u = Utterance::default();
    s.on_speech(0);
    let g = s.generation();
    u.begin(g);
    u.add(1, 500).unwrap();
    u.add(2, 1000).unwrap();
    u.complete(1, "What is the weather".into());
    u.complete(2, "in London tomorrow?".into());
    assert!(u.take_ready().is_none());
    assert_eq!(s.phase(), Phase::Listening);
    u.close();
    s.on_turn_ended(1800);
    let (g, text) = u.take_ready().unwrap();
    let tasks = s.on_transcript(g, text, 1801);
    assert_eq!(
        tasks,
        vec![Task::Filter {
            generation: g,
            raw: "What is the weather in London tomorrow?".into()
        }]
    );
    let tasks = s.on_filter(
        g,
        "CLEAN: What is the weather in London tomorrow?".into(),
        1900,
    );
    assert!(tasks.iter().any(|t| matches!(t, Task::Reply { .. })));
    assert_eq!(s.phase(), Phase::Thinking);
    assert!(s.poll(10_000).is_empty());
}

#[test]
fn multi_phrase_reply_finishes_only_after_generation_and_all_audio_complete() {
    let mut s = session();
    let (g, _) = ask(&mut s, "Tell me something", 0);
    let first = speak_tasks(s.on_reply_token(g, LONG_FIRST, 1000));
    assert_eq!(first.len(), 1, "the opening stretch is released: {first:?}");
    assert!(speak_tasks(s.on_reply_token(g, TAIL, 1100)).is_empty());
    let second = speak_tasks(s.on_reply_complete(g, 1200));
    assert_eq!(second.len(), 1, "the tail is flushed at completion");
    s.on_playback_started(g, 1300);
    s.on_spoken(g, first[0].clone());
    assert!(s.on_playback_finished(g, 2000).is_empty());
    assert_eq!(s.phase(), Phase::Speaking);
    assert_eq!(s.conversation().turn_count(), 1);
    s.on_spoken(g, second[0].clone());
    s.on_playback_finished(g, 4000);
    assert_eq!(s.phase(), Phase::Listening);
    assert!(!s.is_current(g));
    assert_eq!(s.conversation().turn_count(), 2);
    let reply = s.conversation().turns().last().unwrap();
    assert!(reply.text.contains("ridge") && reply.text.contains("move"));
    assert!(!reply.interrupted);
}

#[test]
fn generation_finishing_after_audio_still_completes_and_retains_history() {
    let mut s = session();
    let (g, _) = ask(&mut s, "My name is Mira", 0);
    // Playback of the released stretch completes before generation does, which must not end
    // the turn while the model is still producing the rest of the answer.
    let spoken = speak_tasks(s.on_reply_token(g, LONG_FIRST, 1000));
    assert_eq!(spoken.len(), 1);
    s.on_playback_started(g, 1300);
    s.on_spoken(g, spoken[0].clone());
    assert!(s.on_playback_finished(g, 3000).is_empty());
    assert_eq!(s.phase(), Phase::Speaking);
    s.on_reply_complete(g, 3100);
    assert_eq!(s.phase(), Phase::Listening);
    let (next, tasks) = ask(&mut s, "What is my name?", 4000);
    assert_ne!(next, g);
    let messages = tasks
        .into_iter()
        .find_map(|t| match t {
            Task::Reply { messages, .. } => Some(messages),
            _ => None,
        })
        .unwrap();
    assert!(messages.contains(&("user", "My name is Mira".into())));
    assert!(messages
        .iter()
        .any(|(role, text)| *role == "assistant" && text.contains("ridge")));
    assert!(s.on_reply_token(g, "stale audio", 5000).is_empty());
}

#[test]
fn clarification_returns_to_listening_without_a_false_transcription_timeout() {
    let mut s = session();
    s.on_speech(0);
    s.on_turn_ended(700);
    let g = s.generation();
    // Actual noise. "unclear" is a word someone could have said, and a transcript with
    // words in it is no longer refused - that refusal is what left a speaker repeating a
    // greeting at a machine that kept asking them to say it again.
    s.on_transcript(g, "mm uh".into(), 800);
    s.on_filter(g, "ASK: Could you repeat that?".into(), 900);
    assert_eq!(s.phase(), Phase::Preparing);
    s.on_playback_started(g, 1200);
    assert_eq!(s.phase(), Phase::Speaking);
    s.on_spoken(g, "Could you repeat that?".into());
    s.on_playback_finished(g, 2000);
    assert_eq!(s.phase(), Phase::Listening);
    assert!(s.poll(11_000).is_empty());
}

#[test]
fn interruption_during_synthesis_rejects_old_results_and_credits_only_completed_phrases() {
    let mut s = session();
    let (g, _) = ask(&mut s, "Hello", 0);
    assert_eq!(speak_tasks(s.on_reply_token(g, LONG_FIRST, 1000)).len(), 1);
    assert_eq!(s.phase(), Phase::Preparing);
    let tasks = s.on_speech(1100);
    assert!(tasks.contains(&Task::StopSpeaking));
    assert!(tasks.contains(&Task::Cancel { through: g }));
    assert!(s.on_filter(g, "ASK: stale".into(), 1200).is_empty());
    assert!(s.on_reply_complete(g, 1300).is_empty());
    s.on_spoken(g, "never heard".into());
    // Nothing was spoken before the barge-in, so the abandoned question leaves nothing
    // behind either: the window holds exchanges, not halves of them.
    assert_eq!(s.conversation().turn_count(), 0);
}

#[test]
fn duplicate_filter_and_terminal_events_cannot_emit_a_second_reply() {
    let mut s = session();
    let (g, _) = ask(&mut s, "Hello", 0);
    assert!(s.on_filter(g, "CLEAN: Hello".into(), 950).is_empty());
    assert!(s.on_reply_token(g, "Hi.", 1000).is_empty());
    let done = s.on_reply_complete(g, 1100);
    assert_eq!(
        done.iter()
            .filter(|t| matches!(t, Task::Speak { .. }))
            .count(),
        1
    );
    assert!(s.on_reply_complete(g, 1200).is_empty());
    assert!(s.on_reply_token(g, "late", 1300).is_empty());
}
