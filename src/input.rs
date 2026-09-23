// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Collect recognition jobs in capture order, then finalize exactly once at the endpoint.
use crate::session::Generation;
use std::collections::{BTreeMap, BTreeSet};

/// Recognition jobs one utterance may hold.
const MAX_PIECES: usize = 128;
/// Audio one utterance may hold. Three minutes is far past anything anyone says in one turn;
/// it is here so a microphone left open in front of a television cannot grow without bound.
const MAX_AUDIO_MS: usize = 180_000;

#[derive(Default)]
pub struct Utterance {
    generation: Option<Generation>,
    pieces: BTreeMap<u64, Option<String>>,
    /// Pieces that begin with audio repeated from the one before: see
    /// [`crate::audio::SpeechSegment::overlaps_previous`].
    overlapping: BTreeSet<u64>,
    closed: bool,
    delivered: bool,
    audio_ms: usize,
    /// Clear speech that recognition returned no words for. See [`Utterance::unheard`].
    unheard_ms: usize,
    partials: BTreeMap<u64, BTreeMap<usize, String>>,
}

impl Utterance {
    pub fn begin(&mut self, generation: Generation) {
        *self = Self {
            generation: Some(generation),
            ..Self::default()
        };
    }

    /// Audio accepted into this utterance so far, in milliseconds.
    pub fn audio_ms(&self) -> usize {
        self.audio_ms
    }

    pub fn generation(&self) -> Option<Generation> {
        self.generation
    }

    /// Whether the endpoint has already closed this utterance.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Only an open capture needs a microphone heartbeat. Recognition and repair may
    /// continue after the user has muted the microphone.
    pub fn expects_audio(&self) -> bool {
        self.generation.is_some() && !self.closed
    }

    pub fn waiting_for_recognition(&self) -> bool {
        self.generation.is_some() && self.closed && !self.delivered
    }

    /// Whether this utterance has grown past what one turn may hold.
    ///
    /// Reopening a full utterance would lose the next piece; the runner reports incomplete
    /// input instead of answering a truncated question.
    pub fn is_full(&self) -> bool {
        self.pieces.len() >= MAX_PIECES || self.audio_ms >= MAX_AUDIO_MS
    }

    pub fn add(
        &mut self,
        sequence: u64,
        audio_ms: usize,
        overlaps_previous: bool,
    ) -> Result<(), &'static str> {
        if self.generation.is_none() || self.closed {
            return Err("no open utterance");
        }
        if self.pieces.len() >= MAX_PIECES || self.audio_ms.saturating_add(audio_ms) > MAX_AUDIO_MS
        {
            return Err("utterance exceeded the three-minute limit");
        }
        if self.pieces.contains_key(&sequence) {
            return Err("duplicate recognition job");
        }
        self.audio_ms += audio_ms;
        self.pieces.insert(sequence, None);
        if overlaps_previous {
            self.overlapping.insert(sequence);
        }
        Ok(())
    }

    /// Pieces' words in capture order, joined the way they were cut. A piece cut at a pause
    /// shares no audio with the next, so a word said on both sides of the seam was said twice
    /// and stays twice. One cut with no breath to cut at repeats a little audio into the next,
    /// so the words heard on both sides were said once.
    fn joined<'a>(&self, pieces: impl Iterator<Item = (u64, &'a str)>) -> String {
        let mut text = String::new();
        for (sequence, piece) in pieces {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if self.overlapping.contains(&sequence) && !text.is_empty() {
                text = crate::asr::join_overlapping(&[text, piece.to_string()]);
            } else {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
        }
        text
    }

    pub fn complete(&mut self, sequence: u64, text: String) -> bool {
        match self.pieces.get_mut(&sequence) {
            Some(piece @ None) => {
                *piece = Some(text);
                self.partials.remove(&sequence);
                true
            }
            _ => false,
        }
    }

    /// Records speech in one of this utterance's jobs that recognition could not turn into words.
    pub fn unheard(&mut self, sequence: u64, ms: usize) {
        if self.pieces.contains_key(&sequence) {
            self.unheard_ms = self.unheard_ms.saturating_add(ms);
        }
    }

    /// Milliseconds of this utterance that were clearly spoken but not recognised.
    pub fn unheard_ms(&self) -> usize {
        self.unheard_ms
    }

    pub fn contains(&self, sequence: u64) -> bool {
        self.pieces.contains_key(&sequence)
    }

    /// A display-only hypothesis. It never finalizes or enters conversation history.
    pub fn partial(&mut self, sequence: u64, chunk: usize, text: String) -> Option<String> {
        if !matches!(self.pieces.get(&sequence), Some(None)) || chunk >= 128 || text.len() > 32_768
        {
            return None;
        }
        self.partials
            .entry(sequence)
            .or_default()
            .insert(chunk, text);
        Some(self.preview())
    }

    /// Only show the recognized prefix; a later job cannot jump over an unfinished earlier one.
    pub fn preview(&self) -> String {
        if self.delivered {
            return String::new();
        }
        let mut pieces = Vec::new();
        for (sequence, text) in &self.pieces {
            if let Some(text) = text {
                pieces.push((*sequence, text.clone()));
            } else {
                if let Some(partials) = self.partials.get(sequence) {
                    let chunks: Vec<_> = partials
                        .iter()
                        .enumerate()
                        .take_while(|(expected, (actual, _))| *expected == **actual)
                        .map(|(_, (_, text))| text.clone())
                        .collect();
                    pieces.push((*sequence, crate::asr::join_overlapping(&chunks)));
                }
                break;
            }
        }
        self.joined(
            pieces
                .iter()
                .map(|(sequence, text)| (*sequence, text.as_str())),
        )
    }

    /// Accept more audio into the utterance the endpoint just closed.
    ///
    /// The speaker carried on while recognition was still running. Everything already
    /// submitted stays pending, and the new audio joins the same question rather than
    /// starting one of its own.
    pub fn reopen(&mut self) {
        self.closed = false;
        self.delivered = false;
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn cancel(&mut self) {
        *self = Self::default();
    }

    pub fn take_ready(&mut self) -> Option<(Generation, String)> {
        if !self.closed || self.delivered || self.pieces.values().any(Option::is_none) {
            return None;
        }
        let generation = self.generation?;
        let text = self.joined(
            self.pieces
                .iter()
                .filter_map(|(sequence, text)| Some((*sequence, text.as_deref()?))),
        );
        // Repair can still be interrupted by a continuation of this utterance. Keep the
        // recognized prefix until the runner begins or cancels a generation, but deliver
        // each endpoint only once.
        self.delivered = true;
        Some((generation, text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn generation() -> Generation {
        crate::session::Session::new(
            crate::conversation::Conversation::new(
                "Zen",
                crate::conversation::WindowBudget::for_slot(8192),
            ),
            crate::turn::TurnTimeouts::default(),
            crate::reply::ChunkLimits::default(),
        )
        .generation()
    }

    #[test]
    fn an_utterance_says_it_is_full_before_it_refuses_a_piece() {
        // The host has to be able to ask. Finding out by way of a refused piece used to reach
        // the speaker as invalid input, and took the whole turn - every word already recognised
        // in it included - along with the message.
        let mut utterance = Utterance::default();
        utterance.begin(generation());
        assert!(!utterance.is_full());
        for sequence in 0..MAX_PIECES as u64 {
            assert!(
                utterance.add(sequence, 100, false).is_ok(),
                "piece {sequence}"
            );
        }
        assert!(utterance.is_full());
        assert!(utterance.add(MAX_PIECES as u64, 100, false).is_err());
    }

    #[test]
    fn a_closed_utterance_reopens_for_the_rest_of_the_same_question() {
        // The speaker carried on while recognition of the first half was still running.
        let mut utterance = Utterance::default();
        utterance.begin(generation());
        utterance.add(0, 1_000, false).unwrap();
        utterance.close();
        assert!(utterance.is_closed());
        assert!(
            utterance.add(1, 1_000, false).is_err(),
            "closed means closed"
        );

        utterance.reopen();
        assert!(!utterance.is_closed());
        assert!(utterance.add(1, 1_000, false).is_ok());
        assert_eq!(utterance.audio_ms(), 2_000, "both halves are one question");

        utterance.complete(0, "first half".into());
        utterance.complete(1, "second half".into());
        utterance.close();
        let (_, text) = utterance.take_ready().expect("both pieces are in");
        assert_eq!(text, "first half second half");
    }

    #[test]
    fn early_recognition_waits_for_endpoint_and_keeps_all_phrases_in_capture_order() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(10, 1000, false).unwrap();
        u.add(11, 1000, false).unwrap();
        u.complete(11, "and tomorrow".into());
        u.complete(10, "weather today".into());
        assert!(u.take_ready().is_none());
        u.close();
        assert_eq!(u.take_ready().unwrap().1, "weather today and tomorrow");
        assert!(u.take_ready().is_none());
    }

    #[test]
    fn late_last_chunk_blocks_finalization_but_stale_jobs_cannot_complete_a_new_turn() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(1, 1000, false).unwrap();
        u.close();
        assert!(u.take_ready().is_none());
        u.begin(generation());
        u.add(2, 1000, false).unwrap();
        assert!(!u.complete(1, "old".into()));
        u.close();
        u.complete(2, "new".into());
        assert_eq!(u.take_ready().unwrap().1, "new");
    }

    #[test]
    fn repetitions_and_empty_turns_are_preserved_correctly() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(1, 1000, false).unwrap();
        u.add(2, 1000, false).unwrap();
        u.complete(1, "yes".into());
        u.complete(2, "yes please".into());
        u.close();
        assert_eq!(u.take_ready().unwrap().1, "yes yes please");
        u.begin(generation());
        u.close();
        assert_eq!(u.take_ready().unwrap().1, "");
    }

    #[test]
    fn a_seam_cut_with_no_breath_hears_its_repeated_words_once() {
        // The words on a hard cut are in both pieces, because the audio there is; at a pause
        // they are only repeated if they were said twice.
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(1, 10_000, false).unwrap();
        u.add(2, 4_000, true).unwrap();
        u.add(3, 1_000, false).unwrap();
        u.complete(1, "and the smell of fresh bread was drifting".into());
        u.complete(2, "was drifting out onto the street.".into());
        u.complete(3, "Street food, I mean.".into());
        u.close();
        assert_eq!(
            u.take_ready().unwrap().1,
            "and the smell of fresh bread was drifting out onto the street. Street food, I mean."
        );
    }

    #[test]
    fn partials_preserve_capture_order_and_never_commit_or_revive_stale_jobs() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(5, 1000, false).unwrap();
        u.add(6, 1000, false).unwrap();
        assert_eq!(u.partial(6, 0, "tomorrow".into()).unwrap(), "");
        assert_eq!(u.partial(5, 1, "today".into()).unwrap(), "");
        assert_eq!(u.partial(5, 0, "weather".into()).unwrap(), "weather today");
        u.close();
        assert!(u.take_ready().is_none());
        assert!(u.complete(5, "weather today and".into()));
        assert_eq!(u.preview(), "weather today and tomorrow");
        assert!(u.partial(5, 0, "obsolete".into()).is_none());
        u.complete(6, "tomorrow".into());
        assert_eq!(u.take_ready().unwrap().1, "weather today and tomorrow");
        assert!(u.partial(6, 0, "old".into()).is_none());
        assert_eq!(u.preview(), "");
    }
}
