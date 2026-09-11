// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Collect recognition jobs in capture order, then finalize exactly once at the endpoint.
use crate::session::Generation;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Utterance {
    generation: Option<Generation>,
    pieces: BTreeMap<u64, Option<String>>,
    closed: bool,
    audio_ms: usize,
    partials: BTreeMap<u64, BTreeMap<usize, String>>,
}

impl Utterance {
    pub fn begin(&mut self, generation: Generation) {
        *self = Self {
            generation: Some(generation),
            ..Self::default()
        };
    }

    pub fn generation(&self) -> Option<Generation> {
        self.generation
    }

    pub fn add(&mut self, sequence: u64, audio_ms: usize) -> Result<(), &'static str> {
        if self.generation.is_none() || self.closed {
            return Err("no open utterance");
        }
        if self.pieces.len() >= 128 || self.audio_ms.saturating_add(audio_ms) > 180_000 {
            return Err("utterance exceeded the three-minute limit");
        }
        if self.pieces.contains_key(&sequence) {
            return Err("duplicate recognition job");
        }
        self.audio_ms += audio_ms;
        self.pieces.insert(sequence, None);
        Ok(())
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
        let mut pieces = Vec::new();
        for (sequence, text) in &self.pieces {
            if let Some(text) = text {
                pieces.push(text.trim().to_owned());
            } else {
                if let Some(partials) = self.partials.get(sequence) {
                    let chunks: Vec<_> = partials
                        .iter()
                        .enumerate()
                        .take_while(|(expected, (actual, _))| *expected == **actual)
                        .map(|(_, (_, text))| text.clone())
                        .collect();
                    pieces.push(crate::asr::join_overlapping(&chunks));
                }
                break;
            }
        }
        pieces
            .into_iter()
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
    pub fn close(&mut self) {
        self.closed = true;
    }
    pub fn cancel(&mut self) {
        *self = Self::default();
    }

    pub fn take_ready(&mut self) -> Option<(Generation, String)> {
        if !self.closed || self.pieces.values().any(Option::is_none) {
            return None;
        }
        let generation = self.generation?;
        // Capture segments do not overlap. Preserve deliberate repetition between phrases.
        let text = self
            .pieces
            .values()
            .filter_map(Option::as_deref)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        self.cancel();
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
    fn early_recognition_waits_for_endpoint_and_keeps_all_phrases_in_capture_order() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(10, 1000).unwrap();
        u.add(11, 1000).unwrap();
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
        u.add(1, 1000).unwrap();
        u.close();
        assert!(u.take_ready().is_none());
        u.begin(generation());
        u.add(2, 1000).unwrap();
        assert!(!u.complete(1, "old".into()));
        u.close();
        u.complete(2, "new".into());
        assert_eq!(u.take_ready().unwrap().1, "new");
    }

    #[test]
    fn repetitions_and_empty_turns_are_preserved_correctly() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(1, 1000).unwrap();
        u.add(2, 1000).unwrap();
        u.complete(1, "yes".into());
        u.complete(2, "yes please".into());
        u.close();
        assert_eq!(u.take_ready().unwrap().1, "yes yes please");
        u.begin(generation());
        u.close();
        assert_eq!(u.take_ready().unwrap().1, "");
    }

    #[test]
    fn partials_preserve_capture_order_and_never_commit_or_revive_stale_jobs() {
        let mut u = Utterance::default();
        u.begin(generation());
        u.add(5, 1000).unwrap();
        u.add(6, 1000).unwrap();
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
