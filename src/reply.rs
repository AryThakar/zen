// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Text between the recogniser and the speaker.
//!
//! Three jobs, all of them text-in text-out and none of them touching audio or the network, so
//! every rule below is reachable from a test:
//!
//! 1. **Filter protocol** - reading the filter slot's verdict on a raw transcript, and refusing
//!    to accept a "correction" that invented words the speaker never said.
//! 2. **Speakable text** - turning a reply into something a synthesiser reads aloud correctly.
//! 3. **Output chunking** - cutting a token stream into phrases, with the first one short so
//!    audio starts quickly.

// ============================================================================================
// Filter slot protocol
// ============================================================================================

/// What the filter slot decided about an utterance.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterVerdict {
    /// Usable. Carries the cleaned transcript.
    Clean(String),
    /// Not usable. Carries the filter's own words asking for a repeat.
    ///
    /// Written by the model rather than pulled from a fixed list, so it can refer to what it did
    /// hear ("did you say the kitchen light, or the kitchen fan?"). A canned "sorry, say that
    /// again" gives the speaker nothing to correct and they usually just repeat themselves
    /// identically.
    Ask(String),
}

const CLEAN_MARKER: &str = "CLEAN";
const ASK_MARKER: &str = "ASK";

/// Reads the filter slot's reply.
///
/// An unmarked response is treated as a cleaned transcript rather than an error: small models
/// drop the prefix occasionally, and refusing the turn over formatting would make the assistant
/// ask for a repeat when it understood perfectly well. The invention guard below is what actually
/// protects the transcript, so being permissive here costs nothing.
pub fn parse_filter_verdict(response: &str) -> FilterVerdict {
    let trimmed = response.trim();
    for line in trimmed.lines() {
        let line = line.trim();
        if let Some(rest) = strip_marker(line, ASK_MARKER) {
            return FilterVerdict::Ask(rest.to_string());
        }
        if let Some(rest) = strip_marker(line, CLEAN_MARKER) {
            return FilterVerdict::Clean(rest.to_string());
        }
    }
    FilterVerdict::Clean(strip_quotes(trimmed).to_string())
}

/// Matches a leading `CLEAN`/`ASK` however the model punctuated it.
///
/// It is asked for a colon and usually writes one, but not always - measured emitting
/// `ASK Could you say that again?` with no colon at all. Insisting on the exact form meant
/// that answer matched neither marker and fell through to being read as a transcript, so the
/// model's request for a repeat became the words it thought the speaker had said.
fn strip_marker<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    let start = line.get(..marker.len())?;
    if !start.eq_ignore_ascii_case(marker) {
        return None;
    }
    let rest = &line[marker.len()..];
    // The marker has to be a word of its own, or `CLEANING` would parse as `CLEAN`.
    let separator = |c: char| c == ':' || c == '-' || c.is_whitespace();
    match rest.chars().next() {
        None => Some(""),
        Some(c) if separator(c) => Some(strip_quotes(rest.trim_start_matches(separator).trim())),
        Some(_) => None,
    }
}

fn strip_quotes(text: &str) -> &str {
    let text = text.trim();
    text.strip_prefix('"')
        .and_then(|inner| inner.strip_suffix('"'))
        .unwrap_or(text)
        .trim()
}

/// How alike two words are spelled, from 0.0 (nothing in common) to 1.0 (identical).
///
/// Edit distance normalised by the longer word, so the measure does not punish long words for
/// having more room to differ: "wether" and "weather" differ by one character out of seven and
/// score 0.86, while "book" and "the" share nothing and score 0.
fn spelling_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let longest = a.len().max(b.len());
    if longest == 0 {
        return 1.0;
    }
    // Levenshtein, one row at a time: only the previous row is ever needed.
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, from) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, to) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(from != to);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    1.0 - previous[b.len()] as f32 / longest as f32
}

/// Whether a cleaned word is accounted for by something the speaker actually said.
///
/// Exact matches always count. Beyond that a word counts when some raw word is spelled nearly the
/// same, which is what a genuine repair looks like - the filter is correcting how a word was
/// *heard*, so the corrected spelling stays close to the misheard one.
///
/// Fuzzy matching is withheld from very short words. At one or two characters almost anything is
/// within half of anything else ("me" against "mm", "a" against "uh"), so allowing it there would
/// let an invented sentence ground itself on coincidence.
fn is_grounded(word: &str, raw_words: &[String]) -> bool {
    const MIN_FUZZY_LENGTH: usize = 3;
    const MIN_SIMILARITY: f32 = 0.5;
    raw_words.iter().any(|raw| {
        raw == word
            || (word.chars().count() >= MIN_FUZZY_LENGTH
                && spelling_similarity(word, raw) >= MIN_SIMILARITY)
    })
}

/// Whether a cleaned transcript is plausibly the same utterance as the raw one.
///
/// The filter is asked to repair recognition errors, not to guess at meaning. A model given
/// mangled input will happily produce a fluent sentence that shares almost nothing with what was
/// said, and that sentence then becomes the user's words for the rest of the conversation.
///
/// The comparison has to be on *spelling*, not on word identity. Changing words is the filter's
/// entire job: an exact-match test scored "what is the weather today" against "wut is the wether
/// tooday" at two words out of five and threw the correct repair away, handing the assistant the
/// garbled text instead. Every word the filter fixed counted against it, so the guard fired
/// hardest on exactly the transcripts that needed repair most.
///
/// Measured as the share of cleaned words grounded in the raw transcript by [`is_grounded`].
/// Repairs keep each word recognisable; invention replaces them with unrelated ones.
pub fn resembles_original(raw: &str, cleaned: &str) -> bool {
    let raw_words: Vec<String> = normalise_words(raw);
    let cleaned_words: Vec<String> = normalise_words(cleaned);

    if cleaned_words.is_empty() {
        return raw_words.is_empty();
    }
    let raw_units = count_units(raw);
    // Refuse a "correction" far longer than the input. The filter elaborating is the same
    // failure as the filter inventing, and this is checked before the short-utterance exemption
    // below - otherwise a two-word command could be expanded into a whole sentence unchallenged.
    let growth = count_units(cleaned) as f32 / raw_units.max(1) as f32;
    if growth > 2.0 {
        return false;
    }
    // A short utterance can legitimately be rewritten in full ("hey zen" -> "Hey Zen"), and the
    // overlap measure is too noisy to judge at that length.
    if raw_units <= 2 {
        return true;
    }
    // A translation changes every word by design, so comparing spellings across scripts measures
    // nothing: "what can I do" shares no letters with the Devanagari it came from, and the guard
    // would reject every correct translation the filter produced. The length check above still
    // applies - and it is counted in units rather than words for this case above all, since a
    // sentence written without spaces is one word however long it is, and dividing by that made
    // every correct translation out of Chinese, Japanese or Thai look like pure invention.
    if !is_latin_script(raw) && is_latin_script(cleaned) {
        return true;
    }
    let grounded = cleaned_words
        .iter()
        .filter(|word| is_grounded(word, &raw_words))
        .count();
    grounded as f32 / cleaned_words.len() as f32 >= 0.5
}

/// Whether every letter in the text is written in the Latin alphabet.
///
/// Used only to notice that the filter has translated rather than repaired. The cutoff sits above
/// Latin Extended-B, so accented and extended Latin still count as Latin while Devanagari,
/// Cyrillic, Greek, Arabic, Hebrew and the CJK scripts do not.
fn is_latin_script(text: &str) -> bool {
    const LAST_LATIN: char = '\u{024F}';
    !text.chars().any(|c| c.is_alphabetic() && c > LAST_LATIN)
}

/// How many separable pieces of speech a transcript holds.
///
/// Whitespace-delimited words, except in the scripts that are written without spaces between
/// words, where each character counts for itself. Those are the whole reason this exists: a
/// Chinese question is one "word" however long it is, so measuring a translation of it against
/// a word count of one made every correct translation look like six words invented out of
/// nothing, and the guard threw away exactly the repairs it was least able to judge.
fn count_units(text: &str) -> usize {
    let mut units = 0;
    let mut in_word = false;
    for character in text.chars() {
        if !character.is_alphanumeric() {
            in_word = false;
        } else if is_unspaced_script(character) {
            // Roughly a word each, sometimes two; either way far closer than counting a whole
            // sentence as one.
            units += 1;
            in_word = false;
        } else if !in_word {
            units += 1;
            in_word = true;
        }
    }
    units
}

/// Whether a character belongs to a script written without spaces between words.
///
/// Only these are counted per character. Hangul is deliberately absent: Korean is written with
/// spaces, so counting its syllables would make one ordinary word look like a sentence.
fn is_unspaced_script(character: char) -> bool {
    matches!(character,
        '\u{0E00}'..='\u{0EFF}'      // Thai and Lao
        | '\u{1000}'..='\u{109F}'    // Myanmar
        | '\u{1780}'..='\u{17FF}'    // Khmer
        | '\u{3040}'..='\u{30FF}'    // Hiragana and katakana
        | '\u{3400}'..='\u{4DBF}'    // CJK extension A
        | '\u{4E00}'..='\u{9FFF}'    // CJK unified ideographs
        | '\u{F900}'..='\u{FAFF}') // CJK compatibility ideographs
}

fn normalise_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Applies the verdict, falling back to the raw transcript when the filter overreached.
/// Words that carry nothing on their own. A transcript of only these is worth asking about.
const FILLER: [&str; 12] = [
    "uh", "um", "umm", "uhh", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "mhm",
];

/// Whether the speaker said anything worth answering.
///
/// Deliberately generous. The cost of getting this wrong in one direction is a reply to a
/// cough; in the other it is telling someone who spoke perfectly clearly that they were not
/// understood, and there is nothing they can do about it but say the same word again.
///
/// A word counts when it could be one: it carries a vowel, or it is not written in the Latin
/// alphabet at all, where that test means nothing. "mmph" is a noise, "ok" and "why" are
/// answers, and a greeting in Devanagari is a greeting.
fn has_substance(raw: &str) -> bool {
    raw.split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .any(|word| {
            !word.is_empty()
                && !FILLER.contains(&word.as_str())
                && (!is_latin_script(&word)
                    || word.chars().any(|c| "aeiouy".contains(c) || c.is_numeric()))
        })
}

pub fn accept_verdict(raw: &str, verdict: FilterVerdict) -> FilterVerdict {
    match verdict {
        FilterVerdict::Clean(cleaned) if !resembles_original(raw, &cleaned) => {
            // The filter invented. The raw transcript is imperfect but it is what the speaker
            // actually said, which is the property that matters.
            FilterVerdict::Clean(raw.trim().to_string())
        }
        // The mirror of that, and the one that is actually maddening to sit through. The filter
        // is a small model reading one line out of context, and it will call a short or
        // unfamiliar utterance garbled - a greeting in another script, a bare "ok", a one-word
        // question. Spoken back, that is "could you say that again?" to someone who said it
        // perfectly clearly, and saying it again produces the same verdict, so the
        // conversation cannot move. It may only refuse a transcript with nothing in it.
        FilterVerdict::Ask(_) if has_substance(raw) => FilterVerdict::Clean(raw.trim().to_string()),
        other => other,
    }
}

// ============================================================================================
// Speakable text
// ============================================================================================

/// Characters a writer uses to mark a pause. A synthesiser has no glyph for them, so they are
/// turned into the punctuation it does read as a pause rather than removed.
const EM_DASH: char = '\u{2014}';
const EN_DASH: char = '\u{2013}';
const ELLIPSIS: char = '\u{2026}';

/// Rewrites a reply so a synthesiser reads it correctly.
///
/// The reply slot is instructed to write prose without digits or symbols, but a small model slips
/// and a slip is not a cosmetic problem: a synthesiser handed "50%" may read "percent sign", and
/// handed "**bold**" may read the asterisks aloud. This runs regardless of the prompt, for the
/// same reason thinking is disabled at the server as well as per request.
pub fn to_speakable(text: &str) -> String {
    let without_urls = strip_urls(text);
    let mut out = String::with_capacity(without_urls.len() + 16);
    let characters: Vec<char> = without_urls.chars().collect();
    let mut index = 0;

    while index < characters.len() {
        let character = characters[index];

        if character.is_ascii_digit() {
            let start = index;
            while index < characters.len()
                && (characters[index].is_ascii_digit()
                    || (characters[index] == '.'
                        && index + 1 < characters.len()
                        && characters[index + 1].is_ascii_digit()))
            {
                index += 1;
            }
            let number: String = characters[start..index].iter().collect();
            push_spaced(&mut out, &number_to_words(&number));
            continue;
        }

        // Dashes and ellipses are how a writer marks a pause, and a synthesiser reads a comma as
        // one. Dropping them cost the reply its phrasing - and an unspaced dash silently welded
        // the words on either side into one, which the voice then read as a single nonsense word.
        if matches!(character, EM_DASH | EN_DASH | ELLIPSIS) {
            // Never doubles up: a pause mark after existing punctuation would read as a stall.
            if out.trim_end().ends_with([',', '.', '!', '?', ':', ';']) {
                out.push(' ');
            } else if !out.trim().is_empty() {
                out.push_str(", ");
            }
            index += 1;
            continue;
        }

        let replacement = match character {
            '%' => Some("percent"),
            '&' => Some("and"),
            '+' => Some("plus"),
            '=' => Some("equals"),
            '@' => Some("at"),
            '$' => Some("dollars"),
            '£' => Some("pounds"),
            '€' => Some("euros"),
            '°' => Some("degrees"),
            '/' => Some("slash"),
            _ => None,
        };
        if let Some(word) = replacement {
            push_spaced(&mut out, word);
            index += 1;
            continue;
        }

        // Everything a synthesiser should never see. Markdown, brackets and code punctuation are
        // dropped rather than voiced.
        if matches!(
            character,
            '*' | '_' | '`' | '#' | '<' | '>' | '[' | ']' | '{' | '}' | '|' | '\\' | '~' | '^'
        ) {
            index += 1;
            continue;
        }

        if character.is_alphabetic()
            || character.is_whitespace()
            || matches!(character, '.' | ',' | '!' | '?' | '\'' | '-' | ':' | ';')
        {
            out.push(character);
        }
        index += 1;
    }

    collapse_whitespace(&out)
}

fn push_spaced(out: &mut String, word: &str) {
    if !out.is_empty() && !out.ends_with(' ') {
        out.push(' ');
    }
    out.push_str(word);
    out.push(' ');
}

fn strip_urls(text: &str) -> String {
    text.split_whitespace()
        .filter(|word| {
            !(word.starts_with("http://")
                || word.starts_with("https://")
                || word.starts_with("www."))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            space = true;
            continue;
        }
        // Punctuation attaches to the word before it, so a space inserted by a substitution does
        // not leave "hello ." behind.
        if !out.is_empty() && space && !matches!(character, '.' | ',' | '!' | '?' | ':' | ';') {
            out.push(' ');
        }
        space = false;
        out.push(character);
    }
    out.trim().to_string()
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Spells a number the way it would be read aloud.
pub fn number_to_words(number: &str) -> String {
    if let Some((whole, fraction)) = number.split_once('.') {
        let whole = integer_to_words(whole);
        let digits: Vec<String> = fraction
            .chars()
            .filter(|character| character.is_ascii_digit())
            .map(|digit| ONES[digit as usize - '0' as usize].to_string())
            .collect();
        return format!("{whole} point {}", digits.join(" "));
    }
    integer_to_words(number)
}

fn integer_to_words(digits: &str) -> String {
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return "zero".into();
    }
    // Beyond six digits, reading digit by digit is both correct and what a listener expects for
    // things like account or phone numbers.
    if trimmed.len() > 6 {
        return trimmed
            .chars()
            .map(|digit| ONES[digit as usize - '0' as usize])
            .collect::<Vec<_>>()
            .join(" ");
    }
    let value: u64 = trimmed.parse().unwrap_or(0);
    spell(value)
}

fn spell(value: u64) -> String {
    match value {
        0..=19 => ONES[value as usize].to_string(),
        20..=99 => {
            let tens = TENS[(value / 10) as usize];
            match value % 10 {
                0 => tens.to_string(),
                rest => format!("{tens} {}", ONES[rest as usize]),
            }
        }
        100..=999 => {
            let hundreds = format!("{} hundred", ONES[(value / 100) as usize]);
            match value % 100 {
                0 => hundreds,
                rest => format!("{hundreds} {}", spell(rest)),
            }
        }
        _ => {
            let thousands = format!("{} thousand", spell(value / 1000));
            match value % 1000 {
                0 => thousands,
                rest => format!("{thousands} {}", spell(rest)),
            }
        }
    }
}

// ============================================================================================
// Output chunking
// ============================================================================================

/// Word counts governing where the reply stream is cut.
#[derive(Debug, Clone, Copy)]
pub struct ChunkLimits {
    /// Words after which the first chunk is released at any punctuation.
    ///
    /// Lower than the steady-state figure, but not by much, and that is a correction. The
    /// original reasoning was that a shorter first chunk reaches the ear sooner - true of a
    /// synthesiser that returns a finished phrase, and false of this one, which streams: its
    /// time to first audio is set by fixed setup, around 640 ms, not by how many words it was
    /// given. So cutting the first chunk at six words bought a fraction of a second and paid
    /// for it with the prosody of the entire reply, because every boundary is a separate call
    /// that restarts the contour and resets the emphasis.
    ///
    /// Set above a typical short answer instead. A reply of one or two sentences now never
    /// reaches a boundary at all and is flushed whole at the end of generation, which is what
    /// makes it sound like one thought rather than two fragments read out in turn.
    pub first_soft: usize,
    /// Words after which the first chunk may also be cut at a comma.
    pub first_weak: usize,
    /// Words after which the first chunk is cut whether or not punctuation appeared.
    pub first_hard: usize,
    /// Steady-state: words after which a *sentence* end releases a chunk.
    ///
    /// Well above [`Self::first_soft`], and that gap is the whole point. Every chunk boundary
    /// is a separate call to the synthesiser, which knows nothing of the text either side of
    /// it: the contour restarts, the emphasis resets, and a reply delivered as four of these
    /// does not sound like one thought. Only the first chunk is worth that cost, because the
    /// listener is waiting in silence for it. After that there is audio playing to hide the
    /// synthesis behind, so the right move is to hand over as much as will still arrive in
    /// time, not as little as will do.
    pub soft: usize,
    /// Steady-state: words after which a comma or clause break will do.
    ///
    /// The gap between this and [`Self::soft`] is what keeps the rhythm human. Inside it only a
    /// finished sentence is worth stopping for; past it the phrase has grown long enough that
    /// holding on costs more than the interrupted contour does.
    pub weak: usize,
    /// Steady-state hard limit.
    pub hard: usize,
    /// Never emit a chunk shorter than this, however inviting the punctuation.
    pub minimum: usize,
}

impl Default for ChunkLimits {
    fn default() -> Self {
        Self {
            // Above a whole ordinary reply, so most are never cut at all. Measured in a live
            // session, a reply runs thirty to sixty words; at twenty-two this released the last
            // sentence end in the buffer, which was often only eight or sixteen words in, and
            // the rest of the reply followed as two or three more separate synthesis calls.
            first_soft: 55,
            first_weak: 70,
            first_hard: 85,
            soft: 60,
            // Far enough past `soft` that a sentence running long is given real room to finish.
            // Once audio is playing there is slack to spend: a chunk only has to be ready before
            // the previous one stops, and generation runs several times faster than speech.
            weak: 75,
            // The ceiling on how much a single interruption can cost. Only completed phrases
            // enter history, so a phrase cut short is a phrase the listener heard and the
            // transcript does not have. Ninety words is roughly thirty-five seconds: the price
            // of a reply that sounds like one thought instead of four.
            hard: 90,
            // A cut is gated on how much is buffered, but it lands on the last sentence end in
            // that buffer, which can be far earlier. Without a floor on the piece itself, the
            // gate opening at fifty-five words happily emitted the eight-word sentence that
            // happened to finish first, and the listener heard the reply as fragments.
            minimum: 12,
        }
    }
}

/// Cuts a streaming reply into speakable phrases.
#[derive(Debug)]
pub struct ReplyChunker {
    limits: ChunkLimits,
    buffer: String,
    emitted: usize,
}

impl ReplyChunker {
    pub fn new(limits: ChunkLimits) -> Self {
        Self {
            limits,
            buffer: String::new(),
            emitted: 0,
        }
    }

    fn is_first(&self) -> bool {
        self.emitted == 0
    }

    /// Adds generated text, returning a phrase once one is ready.
    pub fn push(&mut self, text: &str) -> Option<String> {
        self.buffer.push_str(text);
        let words = self.buffer.split_whitespace().count();
        let (soft, weak, hard) = if self.is_first() {
            (
                self.limits.first_soft,
                self.limits.first_weak,
                self.limits.first_hard,
            )
        } else {
            (self.limits.soft, self.limits.weak, self.limits.hard)
        };

        // A whole sentence is the best thing to hand a synthesiser, so it is tried first and from
        // the earliest point.
        if words >= soft {
            if let Some(cut) = self.cut_at(BoundaryStrength::Sentence, hard) {
                return Some(cut);
            }
        }
        // Only once the phrase is long enough to be awkward does a clause break or a comma become
        // worth the interrupted contour it costs.
        if words >= weak {
            if let Some(cut) = self.cut_at(BoundaryStrength::Weak, hard) {
                return Some(cut);
            }
        }
        if words >= hard {
            // No punctuation arrived in time. Cut at a word boundary rather than mid-word, which
            // a synthesiser would otherwise voice as a fragment.
            if let Some(cut) = self
                .word_limit(hard)
                .or_else(|| self.buffer.rfind(char::is_whitespace))
            {
                if self.buffer[..cut].split_whitespace().count() >= self.limits.minimum {
                    return Some(self.take(cut));
                }
            }
        }
        // Bound scripts without whitespace and single oversized model events as well as words.
        if self.buffer.len() > 1_024 {
            let end = nearest_boundary(&self.buffer, 1_024);
            let cut = self.buffer[..end]
                .rfind(char::is_whitespace)
                .filter(|p| *p > 0)
                .unwrap_or(end);
            return Some(self.take(cut));
        }
        None
    }

    /// Cuts at the last boundary of at least this strength, if doing so leaves a phrase worth
    /// speaking on its own.
    ///
    /// The minimum is what stops "Hello, how can I help?" from being delivered as a lone "Hello,"
    /// followed by a pause. A one- or two-word fragment costs a whole utterance's worth of
    /// startup and trailing silence to say almost nothing, and it is heard as a stutter.
    fn word_limit(&self, hard: usize) -> Option<usize> {
        let mut in_word = false;
        let mut words = 0;
        for (i, c) in self.buffer.char_indices() {
            if c.is_whitespace() {
                if in_word && words >= hard.max(1) {
                    return Some(i);
                }
                in_word = false;
            } else if !in_word {
                words += 1;
                in_word = true;
            }
        }
        None
    }

    fn cut_at(&mut self, strength: BoundaryStrength, hard: usize) -> Option<String> {
        let end = self.word_limit(hard).unwrap_or(self.buffer.len());
        let cut = last_boundary_of(&self.buffer[..end], strength)?;
        // A trailing period may be the first token of a decimal; wait for lookahead or EOF.
        if cut == self.buffer.len()
            && self.buffer.ends_with('.')
            && self
                .buffer
                .as_bytes()
                .get(cut.saturating_sub(2))
                .is_some_and(u8::is_ascii_digit)
        {
            return None;
        }
        let words = self.buffer[..nearest_boundary(&self.buffer, cut)]
            .split_whitespace()
            .count();
        (words >= self.limits.minimum).then(|| self.take(cut))
    }

    /// Releases whatever is buffered, for the end of a reply or a stalled stream.
    pub fn flush(&mut self) -> Option<String> {
        let remaining = self.buffer.trim().to_string();
        self.buffer.clear();
        if remaining.is_empty() {
            None
        } else {
            self.emitted += 1;
            Some(remaining)
        }
    }

    fn take(&mut self, cut: usize) -> String {
        let cut = nearest_boundary(&self.buffer, cut);
        let chunk = self.buffer[..cut].trim().to_string();
        self.buffer.drain(..cut);
        self.emitted += 1;
        chunk
    }
}

/// How good a place a piece of punctuation is to stop speaking.
///
/// Not a detail. Every chunk is synthesised as its own utterance, so the synthesiser gives each
/// one a complete intonation contour that falls at the end. Cut at a full stop and that contour
/// is the right one - the sentence really did end. Cut at a comma and the listener hears a
/// finished sentence in the middle of a clause, and a reply sliced at every comma arrives as a
/// row of identical falling fragments: the flat, mechanical cadence that makes an assistant sound
/// like a machine reading a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BoundaryStrength {
    /// A comma. Mid-clause; only worth cutting at when a phrase has grown too long to hold.
    Weak,
    /// A semicolon or colon. A real break in the thought, though not the end of one.
    Clause,
    /// A full stop, question mark or exclamation. The synthesiser's contour matches the text.
    Sentence,
}

/// How long to stay quiet after a phrase, given how that phrase ended.
///
/// Each chunk is synthesised as its own utterance and they are queued back to back, so without
/// this a reply arrives as one unbroken run of speech - the sentences are there but the spaces
/// between them are not, which is most of what makes a synthesiser sound like it is reading.
///
/// A single fixed gap does not fix it either, because the gaps people leave are not uniform: a
/// finished thought gets noticeably more silence than a comma inside one. Sizing the pause from
/// the punctuation the phrase ended on is what makes it read as phrasing rather than as buffering.
///
/// A phrase with no terminal punctuation was cut mid-sentence by the length limit, so it gets
/// almost nothing - the thought is still in progress and a pause there would be heard as a stall.
pub fn pause_after_ms(text: &str) -> u64 {
    // The synthesiser already renders the silence a full stop implies, so these are the
    // small amount *on top* of that, not the whole pause. Set as if from scratch they stack
    // with what was synthesised and the reply drags between sentences.
    /// Silence after a completed sentence.
    ///
    /// Measured, synthesis leaves 0-80 ms after the last sound and 70-130 ms before the first
    /// of the next phrase, so roughly 150 ms arrives for free. Conversational speech leaves
    /// something closer to half a second between sentences; this is the rest of it.
    const SENTENCE_MS: u64 = 280;
    /// After a semicolon or colon: a real break, but the thought continues.
    const CLAUSE_MS: u64 = 160;
    /// After a comma.
    const WEAK_MS: u64 = 90;
    /// A phrase the length limit cut mid-thought is not a pause at all. Inserting silence
    /// here puts a gap in the middle of a sentence, which is the one place it cannot belong.
    const UNFINISHED_MS: u64 = 0;

    match text.trim_end().chars().next_back() {
        Some('.') | Some('!') | Some('?') => SENTENCE_MS,
        Some(';') | Some(':') => CLAUSE_MS,
        Some(',') => WEAK_MS,
        _ => UNFINISHED_MS,
    }
}

/// Finds the last boundary of at least `minimum` strength.
///
/// Two guards on the full stop matter, both ported from the legacy scheduler: a stop between
/// digits is a decimal point, and a stop not followed by whitespace is an abbreviation or a
/// version number. Treating either as a sentence end cuts a phrase in half mid-thought.
pub fn last_boundary_of(text: &str, minimum: BoundaryStrength) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut fallback = None;
    for index in (0..bytes.len()).rev() {
        let strength = match bytes[index] {
            b'.' => {
                let between_digits = index > 0
                    && index + 1 < bytes.len()
                    && bytes[index - 1].is_ascii_digit()
                    && bytes[index + 1].is_ascii_digit();
                let mid_word = index + 1 < bytes.len() && !bytes[index + 1].is_ascii_whitespace();
                if between_digits || mid_word {
                    continue;
                }
                BoundaryStrength::Sentence
            }
            b'!' | b'?' => BoundaryStrength::Sentence,
            b';' | b':' => BoundaryStrength::Clause,
            b',' => BoundaryStrength::Weak,
            _ => continue,
        };
        if strength < minimum {
            continue;
        }
        // A stronger boundary is always preferred, even further back in the buffer: ending on a
        // finished sentence reads better than ending nearer the limit on a comma.
        if strength == BoundaryStrength::Sentence {
            return Some(index + 1);
        }
        fallback.get_or_insert(index + 1);
    }
    fallback
}

/// Snaps an index down to a valid UTF-8 boundary so slicing cannot panic.
fn nearest_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_translation_out_of_an_unspaced_script_is_not_read_as_invention() {
        // A Chinese question is one whitespace-delimited word however long it is, so the growth
        // ratio used to divide six English words by one and refuse every correct translation -
        // the raw characters then went to the reply slot and into the visible transcript.
        for (raw, cleaned) in [
            ("今天天气怎么样", "What is the weather like today?"),
            ("今日はいい天気ですね", "It is nice weather today."),
            ("อากาศวันนี้เป็นอย่างไร", "How is the weather today?"),
        ] {
            assert!(resembles_original(raw, cleaned), "{raw} -> {cleaned}");
        }
    }

    #[test]
    fn a_translation_that_balloons_into_a_paragraph_is_still_refused() {
        assert!(!resembles_original(
            "你好",
            "Hello there, I was wondering whether you could tell me about the weather forecast for the rest of this week please"
        ));
    }

    #[test]
    fn unicode_without_spaces_has_bounded_chunks_and_no_lost_characters() {
        let text = "你好世界".repeat(2000);
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let mut chunks = Vec::new();
        let mut next = chunker.push(&text);
        while let Some(chunk) = next {
            chunks.push(chunk);
            next = chunker.push("");
        }
        chunks.extend(chunker.flush());
        assert!(chunks.iter().all(|c| c.len() <= 1024));
        assert_eq!(chunks.concat(), text);
    }
    #[test]
    fn split_decimal_tokens_are_not_split_into_sentences() {
        // Enough words ahead of the decimal to pass the first-chunk gate, so the lookahead
        // rule is what is being tested rather than the gate.
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        // Past the gate and carrying no sentence end yet, so the decimal is the first candidate
        // boundary and the lookahead rule is what is being tested rather than the gate.
        for _ in 0..ChunkLimits::default().first_soft {
            assert!(chunker.push("checking ").is_none());
        }
        assert!(chunker
            .push("The value for this measurement is 3.")
            .is_none());
        let next = chunker.push("14 and it is stable. ").unwrap();
        assert!(next.contains("3.14"), "got {next:?}");
    }

    #[test]
    fn a_finished_sentence_earns_more_silence_than_a_comma() {
        // The point of sizing the pause rather than fixing it. A uniform gap reads as buffering;
        // an uneven one reads as phrasing.
        let sentence = pause_after_ms("That is done.");
        let clause = pause_after_ms("There is one thing:");
        let comma = pause_after_ms("First of all,");
        let unfinished = pause_after_ms("and then we went");
        assert!(sentence > clause && clause > comma && comma > unfinished);
    }

    #[test]
    fn every_sentence_ending_is_treated_alike() {
        let expected = pause_after_ms("Done.");
        assert_eq!(pause_after_ms("Really?"), expected);
        assert_eq!(pause_after_ms("Stop!"), expected);
    }

    #[test]
    fn a_phrase_cut_mid_thought_barely_pauses_at_all() {
        // The length limit cuts here, not the punctuation. The sentence is still running, and a
        // gap in the middle of one is heard as the assistant stalling.
        assert!(pause_after_ms("a lion once lived at the edge of a wide dry") < 100);
    }

    #[test]
    fn trailing_whitespace_does_not_hide_the_punctuation() {
        assert_eq!(
            pause_after_ms("That is done.   "),
            pause_after_ms("That is done.")
        );
    }

    #[test]
    fn an_empty_phrase_does_not_panic() {
        pause_after_ms("");
        pause_after_ms("   ");
    }

    #[test]
    fn the_longest_pause_stays_short_enough_not_to_read_as_a_fault() {
        // Silence between phrases is latency the listener is paying for. Past about a third of a
        // second it stops sounding like a breath and starts sounding like something broke.
        for text in ["Done.", "Well:", "First,", "unfinished"] {
            assert!(pause_after_ms(text) <= 300, "{text:?} pauses too long");
        }
    }

    #[test]
    fn a_translation_out_of_another_script_is_not_read_as_invention() {
        // Zen always answers in English, so the filter translates. Every word changes by design,
        // and spelling similarity across scripts measures nothing - the guard rejected every
        // correct translation and handed the talker text it was told never to reply in.
        assert!(resembles_original(
            "\u{915}\u{939}\u{93e}\u{901} \u{92e}\u{947}\u{902} \u{932}\u{921}\u{93c}\u{928} \u{938}\u{915}\u{924}\u{947} \u{939}\u{948}\u{902}",
            "Where can I fight?"
        ));
    }

    #[test]
    fn a_translation_is_still_refused_when_it_balloons() {
        // The length check is the one guard that still means something across scripts.
        assert!(!resembles_original(
            "\u{915}\u{939}\u{93e}\u{901} \u{92e}\u{947}\u{902}",
            "Where can I fight, and also please book me a table for two at eight o'clock tonight"
        ));
    }

    #[test]
    fn extended_latin_still_counts_as_latin() {
        // An accented repair is a repair, not a translation, and must stay under the guard.
        assert!(is_latin_script("caf\u{e9} na\u{ef}ve stra\u{df}e"));
        assert!(!is_latin_script("\u{915}\u{939}\u{93e}\u{901}"));
        assert!(!is_latin_script(
            "\u{43f}\u{440}\u{438}\u{432}\u{435}\u{442}"
        ));
        assert!(is_latin_script("plain ascii, 123!"));
    }

    /// Chunks a whole reply the way the runtime does: token by token, then a final flush.
    fn phrases(reply: &str) -> Vec<String> {
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let mut chunks = Vec::new();
        for word in reply.split_inclusive(' ') {
            if let Some(chunk) = chunker.push(word) {
                chunks.push(chunk);
            }
        }
        chunks.extend(chunker.flush());
        chunks
    }

    #[test]
    fn a_long_reply_is_cut_into_whole_sentences() {
        // The cadence problem. Each chunk is synthesised as its own utterance and given a falling
        // final contour, so cutting at commas delivers a row of identical fragments - the flat,
        // mechanical rhythm that stops an assistant sounding like a person.
        let reply = "I'd love to. A lion once lived at the edge of a wide, dry plain, \
where the grass came up to his shoulders. He was not the biggest lion, and he knew it, \
but he was patient, and patience out there counts for more than size. \
Every evening he walked the same path to the water. Would you like me to go on?";
        let chunks = phrases(reply);
        for chunk in &chunks {
            let last = chunk.trim_end().chars().last().unwrap();
            assert!(
                matches!(last, '.' | '!' | '?'),
                "chunk ended mid-clause on {last:?}: {chunk:?}"
            );
        }
    }

    #[test]
    fn a_sentence_running_long_is_given_room_to_finish() {
        // Steady state has slack that the first chunk does not: audio is already playing, and
        // generation runs several times faster than speech. Spending it on a finished sentence
        // beats cutting at the nearest comma.
        // Long enough to pass the first-chunk gate, so the choice between "cut at the comma
        // now" and "wait for the full stop" is actually exercised.
        let reply = "Yes, and I should say why. He was not the biggest lion out on that ridge, \
and he knew it perfectly well, but he was patient, and patience out there counts for a great \
deal more than size ever did.";
        let chunks = phrases(reply);
        assert!(
            chunks.iter().all(|chunk| !chunk.trim_end().ends_with(',')),
            "a comma was preferred over a sentence end: {chunks:?}"
        );
        assert!(chunks.last().unwrap().ends_with("did."));
    }

    #[test]
    fn no_phrase_is_too_short_to_be_worth_speaking() {
        // "Hello, how can I help?" must not arrive as a lone "Hello," and a pause. A two-word
        // fragment costs a whole utterance of startup and trailing silence to say nearly nothing,
        // and is heard as a stutter.
        //
        // The floor is on cuts, not on the reply: whatever is left when generation ends is
        // spoken however short it is, which is why the last piece is exempt.
        let spoken = phrases(
            "Hello, how can I help you with that today, Arya? There is a good deal to get              through, and I would rather take it in order than jump about. Tell me where you              would like to begin, and we can work forward from there together at your pace.",
        );
        for chunk in spoken.iter().take(spoken.len() - 1) {
            assert!(
                chunk.split_whitespace().count() >= ChunkLimits::default().minimum,
                "fragment reached synthesis: {chunk:?}"
            );
        }
    }

    #[test]
    fn a_sentence_end_is_preferred_over_a_later_comma() {
        // Even when the comma sits nearer the limit: ending on a finished thought reads better.
        let text = "That is done. And then we went on, and on, and";
        let cut = last_boundary_of(text, BoundaryStrength::Sentence).unwrap();
        assert_eq!(&text[..cut], "That is done.");
    }

    #[test]
    fn boundary_strength_ranks_punctuation_by_how_final_it_sounds() {
        assert!(BoundaryStrength::Sentence > BoundaryStrength::Clause);
        assert!(BoundaryStrength::Clause > BoundaryStrength::Weak);
        assert!(last_boundary_of("one, two, three", BoundaryStrength::Sentence).is_none());
        assert!(last_boundary_of("one, two, three", BoundaryStrength::Weak).is_some());
        assert!(last_boundary_of("one; two", BoundaryStrength::Clause).is_some());
    }

    #[test]
    fn a_stop_inside_a_word_is_not_a_sentence_end() {
        // The guard is positional, not lexical: a stop followed by a non-space is inside a token.
        // A stop that ends an abbreviation *and* is followed by a space is indistinguishable from
        // a sentence end without a dictionary, and is left to read as one - replies written for
        // the ear rarely contain any.
        assert!(last_boundary_of("see e.g this", BoundaryStrength::Sentence).is_none());
    }

    #[test]
    fn a_genuine_repair_of_misheard_words_is_accepted() {
        // The regression this guard's rewrite exists for. Every word the filter fixed used to
        // count against it, so a perfect repair scored two words out of five and was thrown away
        // in favour of the garbled recogniser output.
        assert!(resembles_original(
            "wut is the wether tooday",
            "What is the weather today?"
        ));
        assert!(resembles_original(
            "i want to no about the sollar system",
            "I want to know about the solar system."
        ));
        assert!(resembles_original(
            "turn on the kichen lite",
            "Turn on the kitchen light."
        ));
    }

    #[test]
    fn a_fluent_invention_is_still_rejected() {
        // The other half. Spelling similarity must not become a licence to replace the utterance:
        // these share no recognisable word with what was said.
        assert!(!resembles_original(
            "uh the the mm kitchen",
            "Please book me a table for two at eight o'clock"
        ));
        assert!(!resembles_original(
            "play some music please",
            "The capital of France is Paris"
        ));
    }

    #[test]
    fn short_words_are_not_allowed_to_ground_themselves_by_coincidence() {
        // At one or two characters nearly anything is within half of anything else, so fuzzy
        // matching is withheld there. Without that, "me" grounds on "mm" and "a" on "uh", and an
        // invented sentence borrows enough credit to pass.
        let raw = ["uh".to_string(), "mm".to_string(), "kitchen".to_string()];
        assert!(!is_grounded("me", &raw));
        assert!(!is_grounded("a", &raw));
        assert!(
            is_grounded("kitchen", &raw),
            "an exact word must still ground"
        );
    }

    #[test]
    fn spelling_similarity_separates_repairs_from_replacements() {
        assert_eq!(spelling_similarity("today", "today"), 1.0);
        assert!(spelling_similarity("wether", "weather") > 0.8);
        assert!(spelling_similarity("what", "wut") >= 0.5);
        assert!(spelling_similarity("book", "the") < 0.3);
        assert!(spelling_similarity("table", "kitchen") < 0.3);
    }

    #[test]
    fn similarity_is_symmetric_and_bounded() {
        for (a, b) in [
            ("weather", "wether"),
            ("", "abc"),
            ("same", "same"),
            ("x", ""),
        ] {
            let forward = spelling_similarity(a, b);
            assert!((0.0..=1.0).contains(&forward), "{a}/{b} scored {forward}");
            assert!(
                (forward - spelling_similarity(b, a)).abs() < 1e-6,
                "asymmetric"
            );
        }
    }

    #[test]
    fn a_written_pause_survives_as_a_spoken_one() {
        // These are how a writer marks a pause, and the reply slot uses them when it is asked to
        // write expressively. Dropping them flattened the delivery: the voice read straight
        // through every hesitation and aside.
        assert_eq!(
            to_speakable("Well \u{2014} let me think."),
            "Well, let me think."
        );
        assert_eq!(to_speakable("Wait\u{2026} really?"), "Wait, really?");
        assert_eq!(to_speakable("A \u{2013} B"), "A, B");
    }

    #[test]
    fn an_unspaced_dash_does_not_weld_two_words_together() {
        // The worst of it. Removing the character closed the gap, and the synthesiser read the
        // result as one invented word.
        assert_eq!(
            to_speakable("It was\u{2014}surprisingly\u{2014}fine."),
            "It was, surprisingly, fine."
        );
    }

    #[test]
    fn a_pause_mark_after_punctuation_does_not_stack_up() {
        // "Yes, \u{2014} well" would read as a stumble rather than a pause.
        assert_eq!(to_speakable("Yes, \u{2014} well."), "Yes, well.");
        assert_eq!(to_speakable("Stop. \u{2026} Then go."), "Stop. Then go.");
    }

    #[test]
    fn a_leading_pause_mark_does_not_open_with_a_comma() {
        assert_eq!(to_speakable("\u{2014}starting late"), "starting late");
        assert_eq!(to_speakable("\u{2026} well"), "well");
    }

    #[test]
    fn ordinary_punctuation_is_left_exactly_as_written() {
        // The synthesiser already reads these correctly; rewriting them would change phrasing the
        // model chose deliberately.
        assert_eq!(
            to_speakable("One thing: two things; three, four!"),
            "One thing: two things; three, four!"
        );
    }

    // --- filter protocol ---------------------------------------------------------------------

    #[test]
    fn a_marked_clean_verdict_is_read() {
        assert_eq!(
            parse_filter_verdict("CLEAN: turn on the kitchen light"),
            FilterVerdict::Clean("turn on the kitchen light".into())
        );
    }

    #[test]
    fn a_marked_ask_verdict_carries_the_models_own_words() {
        // The point of a dynamic clarification: it can name what it half-heard, which gives the
        // speaker something to correct instead of repeating themselves identically.
        assert_eq!(
            parse_filter_verdict("ASK: Did you say the kitchen light, or the kitchen fan?"),
            FilterVerdict::Ask("Did you say the kitchen light, or the kitchen fan?".into())
        );
    }

    #[test]
    fn surrounding_quotes_are_stripped() {
        assert_eq!(
            parse_filter_verdict("CLEAN: \"hello there\""),
            FilterVerdict::Clean("hello there".into())
        );
    }

    #[test]
    fn an_unmarked_response_is_treated_as_clean_rather_than_failing_the_turn() {
        // Small models drop the prefix sometimes. Asking for a repeat because of formatting
        // would make the assistant seem deaf when it understood perfectly.
        assert_eq!(
            parse_filter_verdict("turn on the light"),
            FilterVerdict::Clean("turn on the light".into())
        );
    }

    #[test]
    fn markers_are_matched_case_insensitively() {
        assert_eq!(
            parse_filter_verdict("clean: hello"),
            FilterVerdict::Clean("hello".into())
        );
    }

    #[test]
    fn a_genuine_repair_is_accepted() {
        let raw = "turn on the kitchin lite";
        let verdict = FilterVerdict::Clean("turn on the kitchen light".into());
        assert_eq!(
            accept_verdict(raw, verdict),
            FilterVerdict::Clean("turn on the kitchen light".into())
        );
    }

    #[test]
    fn an_invented_sentence_is_rejected_in_favour_of_the_raw_transcript() {
        // The failure this guard exists for. Given mangled input a model will produce a fluent
        // sentence sharing almost nothing with what was said, and that sentence then becomes the
        // user's words for the rest of the conversation.
        let raw = "uh the the mm kitchen";
        let verdict =
            FilterVerdict::Clean("Please book me a table for two at eight o'clock".into());
        assert_eq!(
            accept_verdict(raw, verdict),
            FilterVerdict::Clean(raw.to_string()),
            "an invented correction must lose to the imperfect truth"
        );
    }

    #[test]
    fn the_filter_elaborating_is_treated_as_invention() {
        let raw = "lights off";
        let verdict = FilterVerdict::Clean(
            "Please turn the lights off in the living room and the kitchen as well".into(),
        );
        assert!(matches!(
            accept_verdict(raw, verdict),
            FilterVerdict::Clean(text) if text == "lights off"
        ));
    }

    #[test]
    fn a_very_short_utterance_may_be_rewritten_freely() {
        // "hey zen" -> "Hey Zen" shares no normalised words with itself under a strict measure,
        // and the overlap test is too noisy to judge at that length.
        assert!(resembles_original("hey zen", "Hey Zen!"));
    }

    #[test]
    fn an_ask_verdict_stands_when_there_was_nothing_to_hear() {
        let verdict = FilterVerdict::Ask("Sorry, could you repeat that?".into());
        for noise in ["mmph", "uh um", "hmm", "  ", "shh"] {
            assert_eq!(
                accept_verdict(noise, verdict.clone()),
                verdict,
                "{noise:?} is a noise, not an utterance"
            );
        }
    }

    #[test]
    fn an_ask_verdict_cannot_discard_words_that_were_plainly_spoken() {
        // The failure this exists for. A small model reading one line out of context calls a
        // short or unfamiliar utterance garbled; spoken back, that is "could you say that
        // again?" to someone who was perfectly clear. Saying it again produces the same
        // verdict, so the conversation cannot move at all.
        let verdict = FilterVerdict::Ask("Could you say that again?".into());
        for spoken in ["हेलो", "ok", "Why.", "yes", "मुझे एक कहानी सुनाओ", "no thanks"]
        {
            assert_eq!(
                accept_verdict(spoken, verdict.clone()),
                FilterVerdict::Clean(spoken.trim().to_string()),
                "{spoken:?} was spoken clearly and must reach the reply"
            );
        }
    }

    #[test]
    fn a_marker_is_recognised_however_the_model_punctuated_it() {
        // Measured: the filter emits `ASK Could you say that again?` with no colon. Insisting
        // on the exact form made that answer parse as a transcript instead of a refusal.
        for line in [
            "ASK: Could you say that again?",
            "ASK Could you say that again?",
            "ask - Could you say that again?",
        ] {
            assert_eq!(
                parse_filter_verdict(line),
                FilterVerdict::Ask("Could you say that again?".into()),
                "{line:?}"
            );
        }
        assert_eq!(
            parse_filter_verdict("CLEAN turn on the light"),
            FilterVerdict::Clean("turn on the light".into())
        );
        // A word that merely starts with the marker is not the marker.
        assert_eq!(
            parse_filter_verdict("ASKING for directions"),
            FilterVerdict::Clean("ASKING for directions".into())
        );
    }

    // --- speakable text ----------------------------------------------------------------------

    #[test]
    fn digits_are_spelled_out() {
        assert_eq!(to_speakable("I have 3 apples"), "I have three apples");
        assert_eq!(to_speakable("in 2024"), "in two thousand twenty four");
    }

    #[test]
    fn decimals_are_read_as_point() {
        assert_eq!(
            to_speakable("it is 3.5 metres"),
            "it is three point five metres"
        );
    }

    #[test]
    fn a_long_number_is_read_digit_by_digit() {
        // A phone or account number read as one enormous quantity is unusable.
        assert_eq!(
            to_speakable("call 5551234"),
            "call five five five one two three four"
        );
    }

    #[test]
    fn symbols_become_words_rather_than_being_voiced_literally() {
        // A synthesiser handed "50%" may say "percent sign".
        assert_eq!(to_speakable("50% done"), "fifty percent done");
        assert_eq!(to_speakable("tea & coffee"), "tea and coffee");
    }

    #[test]
    fn markdown_is_removed_rather_than_read_aloud() {
        assert_eq!(to_speakable("that is **very** good"), "that is very good");
        assert_eq!(to_speakable("use `cargo build`"), "use cargo build");
    }

    #[test]
    fn urls_are_dropped_entirely() {
        // There is no useful way to speak one, and every attempt is long and wrong.
        assert_eq!(
            to_speakable("see https://example.com/page for details"),
            "see for details"
        );
    }

    #[test]
    fn sentence_punctuation_survives_because_the_synthesiser_needs_it() {
        assert_eq!(to_speakable("Really? Yes! Fine."), "Really? Yes! Fine.");
    }

    #[test]
    fn substitutions_do_not_leave_a_space_before_punctuation() {
        assert_eq!(to_speakable("it costs 5."), "it costs five.");
    }

    #[test]
    fn plain_prose_passes_through_untouched() {
        let text = "The weather today is mild and clear.";
        assert_eq!(to_speakable(text), text);
    }

    // --- boundaries --------------------------------------------------------------------------

    #[test]
    fn a_decimal_point_is_not_a_sentence_end() {
        // Cutting here would hand the synthesiser "the value is three." and then ".five".
        assert_eq!(
            last_boundary_of("the value is 3.5", BoundaryStrength::Sentence),
            None
        );
    }

    #[test]
    fn an_abbreviation_is_not_a_sentence_end() {
        assert_eq!(
            last_boundary_of("version 1.2.3 released", BoundaryStrength::Sentence),
            None
        );
    }

    #[test]
    fn strong_punctuation_beats_a_comma() {
        let text = "first, second. third";
        assert_eq!(
            &text[..last_boundary_of(text, BoundaryStrength::Weak).unwrap()],
            "first, second."
        );
    }

    #[test]
    fn a_comma_is_used_only_when_nothing_stronger_exists() {
        let text = "one, two three";
        assert_eq!(
            &text[..last_boundary_of(text, BoundaryStrength::Weak).unwrap()],
            "one,"
        );
    }

    // --- chunking ----------------------------------------------------------------------------

    #[test]
    fn a_short_reply_is_spoken_as_one_phrase_rather_than_split() {
        // The fluency case. Two sentences is what most answers are, and cutting between them
        // hands the synthesiser two separate calls: the contour restarts, the emphasis resets,
        // and the reply stops sounding like one thought. Nothing here should reach a boundary.
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let reply = "A lion is a king in the wild. He walks with a quiet strength, \
knowing his place under the sun.";
        for word in reply.split_inclusive(' ') {
            assert!(
                chunker.push(word).is_none(),
                "a {}-word reply must not be chunked",
                reply.split_whitespace().count()
            );
        }
        let whole = chunker.flush().expect("the reply is emitted at completion");
        assert_eq!(
            whole.split_whitespace().count(),
            reply.split_whitespace().count(),
            "every word must survive: {whole:?}"
        );
        assert!(whole.ends_with("sun."));
    }

    #[test]
    fn a_long_reply_still_starts_speaking_before_it_has_finished_generating() {
        // Waiting for the whole of a long answer would leave the listener in silence for as
        // long as it took to generate, so a boundary still has to arrive well before the end.
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let sentence = "The lion walks the ridge at dusk and the herd below him does not move. ";
        let mut first = None;
        for _ in 0..10 {
            for word in sentence.split_inclusive(' ') {
                if let Some(chunk) = chunker.push(word) {
                    first.get_or_insert(chunk);
                }
            }
            if first.is_some() {
                break;
            }
        }
        let first = first.expect("a long reply must start speaking before it ends");
        let words = first.split_whitespace().count();
        assert!(
            words <= ChunkLimits::default().first_hard,
            "first chunk was {words} words: {first:?}"
        );
    }

    #[test]
    fn later_chunks_are_allowed_to_be_longer_than_the_first() {
        let limits = ChunkLimits::default();
        assert!(limits.soft > limits.first_soft);
        assert!(limits.hard > limits.first_hard);
    }

    #[test]
    fn a_speaker_who_never_punctuates_is_still_cut_at_the_hard_limit() {
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let mut emitted = None;
        for _ in 0..ChunkLimits::default().first_hard + 10 {
            if let Some(chunk) = chunker.push("word ") {
                emitted = Some(chunk);
                break;
            }
        }
        assert!(
            emitted.is_some(),
            "an unpunctuated stream must still be cut"
        );
    }

    #[test]
    fn a_cut_never_lands_mid_word() {
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let mut chunk = None;
        for _ in 0..ChunkLimits::default().first_hard + 10 {
            if let Some(emitted) = chunker.push("alpha ") {
                chunk = Some(emitted);
                break;
            }
        }
        let chunk = chunk.expect("a chunk");
        for word in chunk.split_whitespace() {
            assert_eq!(
                word, "alpha",
                "a fragment would be voiced as a partial word"
            );
        }
    }

    #[test]
    fn flush_releases_the_tail_of_a_reply() {
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        chunker.push("and finally");
        assert_eq!(chunker.flush(), Some("and finally".into()));
        assert_eq!(chunker.flush(), None, "flush must be idempotent");
    }

    #[test]
    fn nothing_buffered_flushes_to_nothing() {
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        assert_eq!(chunker.flush(), None);
    }

    #[test]
    fn the_whole_reply_survives_chunking_without_loss() {
        // Chunks are spoken in sequence, so anything dropped here is simply never said.
        let reply = "The weather is mild. It will stay clear all day, and tomorrow looks similar. \
                     Expect a light breeze in the evening.";
        let mut chunker = ReplyChunker::new(ChunkLimits::default());
        let mut pieces = Vec::new();
        for word in reply.split_inclusive(' ') {
            if let Some(chunk) = chunker.push(word) {
                pieces.push(chunk);
            }
        }
        if let Some(tail) = chunker.flush() {
            pieces.push(tail);
        }
        let rejoined = pieces.join(" ");
        assert_eq!(
            rejoined.split_whitespace().collect::<Vec<_>>(),
            reply.split_whitespace().collect::<Vec<_>>()
        );
    }
}
