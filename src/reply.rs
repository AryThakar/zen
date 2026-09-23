// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Text between the recogniser and the speaker.
//!
//! Three jobs, all of them text-in text-out and none of them touching audio or the network, so
//! every rule below is reachable from a test:
//!
//! 1. **Filter protocol** - reading the filter slot's verdict on a raw transcript, and refusing
//!    to accept a "correction" that invented words the speaker never said.
//! 2. **Speakable text** - turning a reply into something a synthesiser reads aloud correctly.
//! 3. **Output chunking** - cutting a token stream into phrases, at a finished sentence wherever
//!    one is available, because each phrase is a separate call to the synthesiser.

use crate::tts::SLOWEST_MS_PER_CHAR;

// ============================================================================================
// Filter slot protocol
// ============================================================================================

/// What the filter slot decided about an utterance.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterVerdict {
    /// Usable. Carries the cleaned transcript.
    Clean(String),
    /// Not usable, in the filter's judgement. Carries its own words asking for a repeat.
    ///
    /// Parsed so it is never mistaken for a transcript, and never said: a transcript with nothing
    /// in it ends the turn before the filter sees it, so every request to repeat that comes back
    /// is about words that were said, and [`accept_verdict`] answers them instead.
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
    raw_words.iter().any(|raw| same_word(word, raw))
}

/// Whether two normalised words are the same word, allowing for how it was heard. See
/// [`resembles_original`].
fn same_word(word: &str, other: &str) -> bool {
    const MIN_FUZZY_LENGTH: usize = 3;
    const MIN_SIMILARITY: f32 = 0.5;
    word == other
        || (word.chars().count() >= MIN_FUZZY_LENGTH
            && spelling_similarity(word, other) >= MIN_SIMILARITY)
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
/// Measured as the share of cleaned words grounded in the raw transcript: the same word, or one
/// close enough in spelling to be a mishearing of it. Repairs keep each word recognisable;
/// invention replaces them with unrelated ones.
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

/// Words that carry nothing on their own. A transcript of only these was said to nobody.
const FILLER: [&str; 12] = [
    "uh", "um", "umm", "uhh", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "mhm",
];

/// Whether the speaker said anything worth answering. A transcript without substance ends the
/// turn in silence; one with it is answered.
///
/// Deliberately generous. The cost of getting this wrong in one direction is a reply to a
/// cough or a hum - measured on the real filter (2026-09-23), "Hmm." came back as a question
/// and was answered; in the other it is silence for someone who spoke, who has to say it again.
///
/// A word counts when it could be one: it carries a vowel, or it is not written in the Latin
/// alphabet at all, where that test means nothing. "mmph" is a noise, "ok" and "why" are
/// answers, and a greeting in Devanagari is a greeting.
pub(crate) fn has_substance(raw: &str) -> bool {
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

// ============================================================================================
// Meaning preservation
// ============================================================================================

/// Words that reverse what a sentence asks for.
///
/// Written without apostrophes because [`normalise_words`] strips them: "don't" arrives as
/// "dont", and "do not" arrives as two words one of which is "not". Either spelling carries one
/// negation, which is what gets counted.
const NEGATIONS: [&str; 28] = [
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "without",
    "cannot", "cant", "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt", "couldnt", "isnt",
    "arent", "wasnt", "werent", "havent", "hasnt", "hadnt", "aint", "except",
];

/// Words that place something in time. Changing or dropping one of these moves an appointment.
const CALENDAR: [&str; 31] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "midnight",
    "noon",
    "weekend",
    "week",
    "month",
];

/// The value of a single number word, or of a run of digits.
fn number_word(word: &str) -> Option<u64> {
    if !word.is_empty() && word.chars().all(|c| c.is_ascii_digit()) {
        return word.parse().ok();
    }
    Some(match word {
        "zero" | "oh" | "nought" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        "hundred" => 100,
        "thousand" => 1_000,
        "million" => 1_000_000,
        "billion" => 1_000_000_000,
        _ => return None,
    })
}

/// Keep digit sequences distinct from quantities: a PIN is not the sum of its digits.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NumberRun {
    value: Option<u64>,
    digits: String,
    sequence: bool,
}

fn numbers_in(words: &[String]) -> Vec<NumberRun> {
    let mut found = Vec::new();
    let numeric = |word: &str| !word.is_empty() && word.bytes().all(|c| c.is_ascii_digit());
    let is_number = |word: &String| numeric(word) || number_word(word).is_some();
    let mut rest = words;
    while let Some(start) = rest.iter().position(is_number) {
        rest = &rest[start..];
        let count = rest.iter().take_while(|word| is_number(word)).count();
        let run = &rest[..count];
        let mut digits = String::new();
        let (mut total, mut current) = (Some(0u64), Some(0u64));
        for word in run {
            let value = number_word(word);
            if numeric(word) {
                digits.push_str(word); // Leading zeros and oversized identifiers are significant.
            } else if let Some(value) = value {
                digits.push_str(&value.to_string());
            }
            match value {
                Some(100) => current = current.and_then(|n| n.max(1).checked_mul(100)),
                Some(scale @ (1_000 | 1_000_000 | 1_000_000_000)) => {
                    total = total.zip(current).and_then(|(t, n)| {
                        n.max(1).checked_mul(scale).and_then(|v| t.checked_add(v))
                    });
                    current = Some(0);
                }
                Some(value) => current = current.and_then(|n| n.checked_add(value)),
                None => current = None,
            }
        }
        found.push(NumberRun {
            value: total.zip(current).and_then(|(t, n)| t.checked_add(n)),
            digits,
            sequence: (count > 1
                && (run.iter().all(|word| numeric(word))
                    || run
                        .iter()
                        .all(|word| number_word(word).is_some_and(|n| n < 10))))
                || run
                    .iter()
                    .any(|word| numeric(word) && word.len() > 1 && word.starts_with('0')),
        });
        rest = &rest[count..];
    }
    found
}

/// Whether a repair says the same thing as what was said.
///
/// [`resembles_original`] asks whether the filter invented: every word it kept has to be
/// traceable to something the speaker said. That is a check in one direction only, and three
/// ways of changing a question survive it, because each one is built entirely from words that
/// really were said:
///
/// - dropping the negation: "do not book Tuesday" becomes "book Tuesday",
/// - swapping a number for a similarly spelled one: "fifteen minutes" becomes "fifty minutes",
/// - stopping early: "what is my account number" becomes "what is my".
///
/// All three were accepted. A repair is meant to correct how a word was *heard*, so none of
/// them is a repair at all - and unlike an obvious invention, each produces a fluent sentence
/// that the rest of the conversation then treats as what the speaker asked for.
///
/// When this refuses, the raw transcript is used instead. That is unpunctuated and sometimes
/// misspelled, and it is what the person actually said, which is the property worth keeping.
pub fn preserves_meaning(raw: &str, cleaned: &str) -> bool {
    let raw_words = normalise_words(raw);
    let cleaned_words = normalise_words(cleaned);
    if raw_words.is_empty() {
        return true;
    }
    // A translation replaces every word by design, so nothing below measures anything. The
    // length check in `resembles_original` is what guards that case.
    if !is_latin_script(raw) && is_latin_script(cleaned) {
        return true;
    }

    let negations = |words: &[String]| {
        words
            .iter()
            .filter(|word| NEGATIONS.contains(&word.as_str()))
            .count()
    };
    if negations(&cleaned_words) != negations(&raw_words) {
        return false;
    }

    let calendar = |words: &[String]| {
        words
            .iter()
            .filter(|word| CALENDAR.contains(&word.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    };
    let original_calendar = calendar(&raw_words);
    if !original_calendar.is_empty() && original_calendar != calendar(&cleaned_words) {
        return false;
    }

    let original = numbers_in(&raw_words);
    let repaired = numbers_in(&cleaned_words);
    if original.len() != repaired.len()
        || original.iter().zip(&repaired).any(|(a, b)| {
            a.digits != b.digits
                && (a.sequence || b.sequence || a.value.is_none() || a.value != b.value)
        })
    {
        return false;
    }

    // How much of what was said survives. The other direction of the resemblance check, and the
    // one that catches a repair stopping early. Below this length a short utterance is
    // legitimately rewritten whole - "ok" into "Okay" shares no spelling at all - and the
    // measure says nothing useful.
    if count_units(raw) <= 2 {
        return true;
    }
    let spoken: Vec<&String> = raw_words
        .iter()
        // Numbers are checked above, exactly, and "twenty five" written back as "25" shares no
        // spelling with what it came from. Counting them here as well would refuse the one
        // rewriting the filter is most obviously right to make.
        .filter(|word| !FILLER.contains(&word.as_str()) && number_word(word).is_none())
        .collect();
    if spoken.is_empty() {
        return true;
    }
    // Repetition and false starts are dropped legitimately, and a word the filter corrected is
    // still grounded in its correction, so this only has to tolerate a little.
    const MIN_SURVIVING: f32 = 0.7;
    let kept = spoken
        .iter()
        .filter(|word| is_grounded(word, &cleaned_words))
        .count();
    kept as f32 / spoken.len() as f32 >= MIN_SURVIVING
}

/// The question to answer: the filter's repair, or the raw transcript when the filter overreached.
pub fn accept_verdict(raw: &str, verdict: FilterVerdict) -> String {
    match verdict {
        FilterVerdict::Clean(cleaned)
            if resembles_original(raw, &cleaned) && preserves_meaning(raw, &cleaned) =>
        {
            cleaned
        }
        // The filter invented, or it changed what was asked. The raw transcript is imperfect
        // but it is what the speaker actually said, which is the property that matters.
        FilterVerdict::Clean(_) => raw.trim().to_string(),
        // The mirror of that, and the one that is actually maddening to sit through. The filter
        // is a small model reading one line out of context, and it will call a short or
        // unfamiliar utterance garbled - a greeting in another script, a bare "ok", a one-word
        // question. Spoken back, that is "could you say that again?" to someone who said it
        // perfectly clearly, and saying it again produces the same verdict, so the
        // conversation cannot move. A transcript with nothing in it never reaches the filter -
        // it ends the turn in silence first (`Session::on_transcript`) - so every request to
        // repeat is one of these, and is overruled.
        FilterVerdict::Ask(_) => raw.trim().to_string(),
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
            let (spoken, next) = read_number(&characters, index);
            push_spaced(&mut out, &spoken);
            index = next;
            continue;
        }

        // A currency sign is written before its amount and said after it: "$5" is "five dollars".
        if let Some((one, many)) = currency(character) {
            if characters.get(index + 1).is_some_and(char::is_ascii_digit) {
                let (spoken, next) = read_amount(&characters, index + 1, one, many);
                push_spaced(&mut out, &spoken);
                index = next;
                continue;
            }
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
            '₹' => Some("rupees"),
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

/// A number as it was written.
struct Written {
    /// The digits of the whole part, grouping commas removed.
    whole: String,
    fraction: Option<String>,
    /// Written with thousands separators, as a quantity is and an account number is not.
    grouped: bool,
    /// Index just past it.
    end: usize,
}

fn scan_number(characters: &[char], start: usize) -> Written {
    let digit_at = |at: usize| characters.get(at).is_some_and(char::is_ascii_digit);
    let mut index = start;
    let mut whole = String::new();
    let mut grouped = false;
    while index < characters.len() {
        if characters[index].is_ascii_digit() {
            whole.push(characters[index]);
            index += 1;
        } else if characters[index] == ','
            && (1..=3).all(|k| digit_at(index + k))
            && !digit_at(index + 4)
        {
            // "384,000": a comma followed by exactly three digits groups them. "1,2,3" is a list.
            grouped = true;
            index += 1;
        } else {
            break;
        }
    }
    let mut fraction = None;
    if characters.get(index) == Some(&'.') && digit_at(index + 1) {
        index += 1;
        let mut digits = String::new();
        while digit_at(index) {
            digits.push(characters[index]);
            index += 1;
        }
        fraction = Some(digits);
    }
    Written {
        whole,
        fraction,
        grouped,
        end: index,
    }
}

/// Reads the number starting at `start`, returning the words and the index just past it.
fn read_number(characters: &[char], start: usize) -> (String, usize) {
    let written = scan_number(characters, start);
    let end = written.end;
    let two_digits = |at: usize| {
        (0..2).all(|k| characters.get(at + k).is_some_and(char::is_ascii_digit))
            && !characters.get(at + 2).is_some_and(char::is_ascii_digit)
    };
    // "9:30" is a time, read "nine thirty" - not "nine: thirty".
    if written.fraction.is_none()
        && !written.grouped
        && written.whole.len() <= 2
        && characters.get(end) == Some(&':')
        && two_digits(end + 1)
    {
        let hour: u64 = written.whole.parse().unwrap_or(0);
        let minute: u64 = characters[end + 1..end + 3]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        if hour <= 24 && minute < 60 {
            let spoken = match minute {
                0 => format!("{} o'clock", spell(hour)),
                1..=9 => format!("{} oh {}", spell(hour), spell(minute)),
                _ => format!("{} {}", spell(hour), spell(minute)),
            };
            return (spoken, end + 3);
        }
    }
    // "1st", "22nd": the suffix is how the number is said, not letters to read after it.
    if written.fraction.is_none() {
        let suffix: String = characters
            .iter()
            .skip(end)
            .take(2)
            .collect::<String>()
            .to_ascii_lowercase();
        let word_ends = !characters.get(end + 2).is_some_and(|c| c.is_alphanumeric());
        if matches!(suffix.as_str(), "st" | "nd" | "rd" | "th") && word_ends {
            return (ordinal(&whole_words(&written)), end + 2);
        }
    }
    (cardinal(&written), end)
}

fn cardinal(written: &Written) -> String {
    let whole = whole_words(written);
    match &written.fraction {
        Some(fraction) => format!("{whole} point {}", digit_by_digit(fraction)),
        None => whole,
    }
}

fn digit_by_digit(digits: &str) -> String {
    digits
        .chars()
        .filter_map(|digit| digit.to_digit(10))
        .map(|digit| ONES[digit as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn whole_words(written: &Written) -> String {
    let trimmed = written.whole.trim_start_matches('0');
    if trimmed.is_empty() {
        return "zero".into();
    }
    let value = trimmed.parse::<u64>().ok();
    match value {
        // Written with separators, it is a quantity however long it is.
        Some(value) if written.grouped && value < 1_000_000_000_000_000 => spell(value),
        // Beyond six digits without them, reading digit by digit is what a listener expects:
        // an account or a phone number read as one enormous quantity is unusable.
        _ if written.grouped || trimmed.len() > 6 => digit_by_digit(trimmed),
        // Four digits in the range years fall in are read as a year: "nineteen ninety", "twenty
        // twenty four". For a quantity that reading is still ordinary English - "fifteen hundred".
        Some(value) if trimmed.len() == 4 && matches!(value, 1_100..=1_999 | 2_010..=2_099) => {
            let (high, low) = (value / 100, value % 100);
            match low {
                0 => format!("{} hundred", spell(high)),
                1..=9 => format!("{} oh {}", spell(high), spell(low)),
                _ => format!("{} {}", spell(high), spell(low)),
            }
        }
        Some(value) => spell(value),
        None => digit_by_digit(trimmed),
    }
}

/// The ordinal of a spelled number: "twenty two" becomes "twenty second".
fn ordinal(cardinal: &str) -> String {
    let (head, last) = match cardinal.rsplit_once(' ') {
        Some((head, last)) => (format!("{head} "), last),
        None => (String::new(), cardinal),
    };
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        word if word.ends_with('y') => format!("{}ieth", &word[..word.len() - 1]),
        word => format!("{word}th"),
    };
    format!("{head}{last}")
}

/// What a currency sign is called, singular and plural.
fn currency(character: char) -> Option<(&'static str, &'static str)> {
    match character {
        '$' => Some(("dollar", "dollars")),
        '£' => Some(("pound", "pounds")),
        '€' => Some(("euro", "euros")),
        '₹' => Some(("rupee", "rupees")),
        _ => None,
    }
}

/// An amount after a currency sign: "$1.50" is "one dollar fifty".
fn read_amount(characters: &[char], start: usize, one: &str, many: &str) -> (String, usize) {
    let written = scan_number(characters, start);
    let whole = whole_words(&written);
    let unit = if written.whole.trim_start_matches('0') == "1" {
        one
    } else {
        many
    };
    let spoken = match &written.fraction {
        Some(cents) if cents.len() == 2 => match cents.parse::<u64>().unwrap_or(0) {
            0 => format!("{whole} {unit}"),
            cents => format!("{whole} {unit} {}", spell(cents)),
        },
        Some(fraction) => format!("{whole} point {} {many}", digit_by_digit(fraction)),
        None => format!("{whole} {unit}"),
    };
    (spoken, written.end)
}

fn spell(value: u64) -> String {
    const SCALES: [(u64, &str); 4] = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
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
            let (size, name) = SCALES
                .into_iter()
                .find(|(size, _)| value >= *size)
                .unwrap_or(SCALES[3]);
            let head = format!("{} {name}", spell(value / size));
            match value % size {
                0 => head,
                rest => format!("{head} {}", spell(rest)),
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

    /// Where the `hard`-th word of the buffer ends, once a space has followed it: the furthest
    /// point one phrase may run to.
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

    /// Cuts at the last boundary of at least this strength, if doing so leaves a phrase worth
    /// speaking on its own.
    ///
    /// The minimum is what stops "Hello, how can I help?" from being delivered as a lone "Hello,"
    /// followed by a pause. A one- or two-word fragment costs a whole utterance's worth of
    /// startup and trailing silence to say almost nothing, and it is heard as a stutter.
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

/// How much later than its share of the text a clause can end in the audio. Of 43 clause and
/// sentence ends in the renders `examples/phrasepace.rs` makes, the latest came 495 ms after
/// that estimate.
const LATEST_CLAUSE_END_MS: usize = 500;

/// The part of a phrase that has certainly been heard, as a prefix of `text`.
///
/// A phrase can run for half a minute, so an interruption part-way through one used to leave no
/// trace of it: the model was told nothing of what it had just said, and started over. Speech
/// comes with no word timings, but where a clause ends is predictable from its share of the text
/// (see [`LATEST_CLAUSE_END_MS`]). Anything counted has to be certain - claiming words the listener
/// never heard is the error the reply ledger exists to prevent - so only whole clauses count,
/// and only once playback is past the latest point each could have ended. The last clause is
/// never counted here; the phrase's own completion credits it.
///
/// `heard_ms` is audio the page has confirmed playing. `total_ms` is the phrase's full length,
/// known once synthesis has finished. Until then, the phrase is assumed to be spoken at
/// [`SLOWEST_MS_PER_CHAR`], which places every clause end as late as it can be.
pub fn heard_part(text: &str, heard_ms: usize, total_ms: Option<usize>) -> &str {
    let chars = text.chars().count().max(1);
    let mut heard = 0;
    for (index, (byte, c)) in text.char_indices().enumerate() {
        let after = byte + c.len_utf8();
        // "384,000" and "9:30" are not clause breaks; a break is followed by a space.
        if !matches!(c, '.' | '!' | '?' | ',' | ';' | ':')
            || !text[after..].starts_with(char::is_whitespace)
        {
            continue;
        }
        let through = index + 1;
        let latest_end = match total_ms {
            Some(total) => through * total / chars,
            None => through * SLOWEST_MS_PER_CHAR,
        } + LATEST_CLAUSE_END_MS;
        if latest_end > heard_ms {
            break;
        }
        heard = after;
    }
    // A comma is where the cut came, not part of what was said.
    text[..heard].trim_end_matches([',', ';', ':'])
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
/// Two guards on the full stop matter: a stop between digits is a decimal point, and a stop not
/// followed by whitespace is an abbreviation or a version number. Treating either as a sentence
/// end cuts a phrase in half mid-thought.
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

    /// 64 characters. "Sure." ends at character 5 and "first," at 27.
    const PASTA: &str = "Sure. Boil the water first, then add the pasta and stir it once.";

    #[test]
    fn a_clause_counts_as_heard_only_once_it_has_certainly_ended() {
        // In a 3.2 s phrase, "Sure." takes up to 5/64 of it - 250 ms - and may end up to
        // 500 ms later than that.
        assert_eq!(heard_part(PASTA, 749, Some(3_200)), "");
        assert_eq!(heard_part(PASTA, 750, Some(3_200)), "Sure.");
        assert_eq!(heard_part(PASTA, 1_849, Some(3_200)), "Sure.");
        // The comma is where the cut came, so it is not kept.
        assert_eq!(
            heard_part(PASTA, 1_850, Some(3_200)),
            "Sure. Boil the water first"
        );
    }

    #[test]
    fn the_last_clause_is_left_to_the_phrase_finishing() {
        // Playback of every sample is not proof the listener heard the end: the phrase's own
        // acknowledgement is. Until then, the final clause stays unclaimed.
        assert_eq!(
            heard_part(PASTA, 3_200, Some(3_200)),
            "Sure. Boil the water first"
        );
    }

    #[test]
    fn a_phrase_still_being_synthesised_is_assumed_to_be_spoken_slowly() {
        // Its length is unknown, so every clause is placed as late as the slowest measured
        // pace puts it: "Sure." by 5 x 56 ms, plus the 500 ms it may run late.
        assert_eq!(heard_part(PASTA, 779, None), "");
        assert_eq!(heard_part(PASTA, 780, None), "Sure.");
        assert_eq!(heard_part(PASTA, 2_011, None), "Sure.");
        assert_eq!(heard_part(PASTA, 2_012, None), "Sure. Boil the water first");
    }

    #[test]
    fn separators_inside_numbers_are_not_clause_breaks() {
        let moon = "The Moon is 384,000 kilometres away, at 9:30 tonight. Look up.";
        // Past where "384," would end, but no clause has.
        assert_eq!(heard_part(moon, 1_800, None), "");
        assert_eq!(
            heard_part(moon, 100_000, None),
            "The Moon is 384,000 kilometres away, at 9:30 tonight."
        );
    }

    #[test]
    fn text_without_any_break_is_never_partly_claimed() {
        assert_eq!(
            heard_part("just one run of words", 100_000, Some(1_000)),
            ""
        );
        assert_eq!(heard_part("", 100_000, None), "");
    }

    /// What the filter handed back, after the guard has had its say.
    fn repaired(raw: &str, cleaned: &str) -> String {
        accept_verdict(raw, FilterVerdict::Clean(cleaned.to_string()))
    }

    #[test]
    fn a_repair_may_not_drop_the_negation() {
        // Every word of "book Tuesday" really was said, so the resemblance check passes it. It
        // is also the opposite of what was asked for.
        assert_eq!(
            repaired("do not book tuesday", "Book Tuesday."),
            "do not book tuesday",
        );
        assert_eq!(
            repaired("i cant make it on friday", "I can make it on Friday."),
            "i cant make it on friday",
        );
        // Contracting or expanding a negation is not dropping it.
        assert_eq!(
            repaired("i do not want that", "I don't want that."),
            "I don't want that.",
        );
    }

    #[test]
    fn a_repair_may_not_change_a_number() {
        assert_eq!(
            repaired("give me fifteen minutes", "Give me fifty minutes."),
            "give me fifteen minutes",
        );
        // Writing the same quantity a different way is a repair, not a change.
        assert_eq!(
            repaired("set it for twenty five past", "Set it for 25 past."),
            "Set it for 25 past.",
        );
        // Digits read out one at a time are the same string of digits.
        assert_eq!(
            repaired("my code is one one two", "My code is 112."),
            "My code is 112.",
        );
    }

    #[test]
    fn a_repair_preserves_ordered_identifiers_quantities_and_polarity() {
        for (raw, changed) in [
            (
                "my pin is one two three four",
                "My pin is four three two one.",
            ),
            ("my code is zero zero one", "My code is 1."),
            ("my code is 001", "My code is 1."),
            (
                "give Alice two and Bob three",
                "Give Alice three and Bob two.",
            ),
            ("do book Tuesday", "Do not book Tuesday."),
            (
                "move it from Tuesday to Friday",
                "Move it from Friday to Tuesday.",
            ),
            ("give me a minute", "Give me two minutes."),
        ] {
            assert_eq!(repaired(raw, changed), raw, "accepted {changed:?}");
        }
        assert_eq!(
            repaired("my code is zero zero one", "My code is 001."),
            "My code is 001."
        );
    }

    #[test]
    fn oversized_numbers_cannot_overflow_or_match_wrapped_quantities() {
        for raw in [
            "my number is 18446744073709551615 one",
            "my number is 99999999999999999999999999999999",
            "my number is billion billion billion billion",
        ] {
            assert!(preserves_meaning(raw, raw));
            assert!(!preserves_meaning(raw, "my number is zero"));
        }
    }

    #[test]
    fn a_repair_may_not_move_a_day_or_a_month() {
        assert_eq!(
            repaired("move it to thursday", "Move it to Tuesday."),
            "move it to thursday",
        );
        assert_eq!(
            repaired("book it for tomorrow morning", "Book it for tomorrow."),
            "book it for tomorrow morning",
        );
    }

    #[test]
    fn a_repair_may_not_stop_partway_through_the_question() {
        assert_eq!(
            repaired("what is my account number", "What is my"),
            "what is my account number",
        );
        assert_eq!(
            repaired(
                "book a table for four except friday",
                "Book a table for four.",
            ),
            "book a table for four except friday",
        );
    }

    #[test]
    fn ordinary_repairs_still_get_through() {
        // The guard must not cost the filter its actual job.
        assert_eq!(
            repaired("wut is the wether tooday", "What is the weather today?"),
            "What is the weather today?",
        );
        assert_eq!(
            repaired(
                "can you turn on the kitchen lights",
                "Can you turn on the kitchen lights?"
            ),
            "Can you turn on the kitchen lights?",
        );
        assert_eq!(repaired("ok", "Okay."), "Okay.");
        assert_eq!(repaired("hey zen", "Hey Zen."), "Hey Zen.");
        // Stutters and false starts are not the question getting shorter.
        assert_eq!(
            repaired("i i i want a a table for two", "I want a table for two."),
            "I want a table for two.",
        );
        assert_eq!(
            repaired(
                "um so i was thinking about the meeting",
                "So I was thinking about the meeting."
            ),
            "So I was thinking about the meeting.",
        );
    }

    #[test]
    fn a_translation_is_still_allowed_through() {
        // Nothing above can measure a rewrite that changes every word by design, and refusing
        // them would leave the speaker answered in a language they did not use.
        let hindi = "मुझे कल की मीटिंग के बारे में बताओ";
        assert_eq!(
            repaired(hindi, "Tell me about tomorrow's meeting."),
            "Tell me about tomorrow's meeting.",
        );
    }

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
        // Whatever the filter asks is carried whole, so the verdict is read as a request and never
        // as the words the speaker said.
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
        assert_eq!(accept_verdict(raw, verdict), "turn on the kitchen light");
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
            raw,
            "an invented correction must lose to the imperfect truth"
        );
    }

    #[test]
    fn the_filter_elaborating_is_treated_as_invention() {
        let raw = "lights off";
        let verdict = FilterVerdict::Clean(
            "Please turn the lights off in the living room and the kitchen as well".into(),
        );
        assert_eq!(accept_verdict(raw, verdict), "lights off");
    }

    #[test]
    fn a_very_short_utterance_may_be_rewritten_freely() {
        // "hey zen" -> "Hey Zen" shares no normalised words with itself under a strict measure,
        // and the overlap test is too noisy to judge at that length.
        assert!(resembles_original("hey zen", "Hey Zen!"));
    }

    #[test]
    fn hesitation_and_noise_have_no_substance() {
        for noise in [
            "mmph",
            "uh um",
            "hmm",
            "  ",
            "shh",
            "Hmm.",
            "Um... uhh.",
            "Hmmmm?",
            "Er, erm.",
        ] {
            assert!(
                !has_substance(noise),
                "{noise:?} is a noise, not an utterance"
            );
        }
        for spoken in [
            "ok",
            "Why.",
            "Hmm, what about tomorrow?",
            "Oh.",
            "Uh-huh.",
            "हेलो",
            "Zen.",
        ] {
            assert!(has_substance(spoken), "{spoken:?} was said");
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
                spoken.trim(),
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
        assert_eq!(
            to_speakable("about 250 of them"),
            "about two hundred fifty of them"
        );
    }

    #[test]
    fn a_grouped_number_is_one_quantity() {
        // Read one group at a time, this came out as "three hundred eighty four, zero".
        assert_eq!(
            to_speakable("The Moon is 384,000 km away."),
            "The Moon is three hundred eighty four thousand km away."
        );
        assert_eq!(to_speakable("1,000,000 people"), "one million people");
        assert_eq!(
            to_speakable("7,900,000,000 of us"),
            "seven billion nine hundred million of us"
        );
        // A comma that does not group three digits is still a comma.
        assert_eq!(to_speakable("pick 1,2 or 3"), "pick one, two or three");
    }

    #[test]
    fn a_time_is_read_as_a_time() {
        assert_eq!(
            to_speakable("the dentist at 9:30, then lunch"),
            "the dentist at nine thirty, then lunch"
        );
        assert_eq!(to_speakable("at 10:05"), "at ten oh five");
        assert_eq!(to_speakable("by 12:00."), "by twelve o'clock.");
        // Not a time: the minutes do not exist.
        assert_eq!(to_speakable("ratio 3:75"), "ratio three: seventy five");
    }

    #[test]
    fn an_ordinal_is_said_as_one() {
        assert_eq!(
            to_speakable("the 1st, 2nd, 3rd and 4th"),
            "the first, second, third and fourth"
        );
        assert_eq!(to_speakable("on the 22nd"), "on the twenty second");
        assert_eq!(
            to_speakable("her 11th and 20th"),
            "her eleventh and twentieth"
        );
        assert_eq!(to_speakable("the 105th time"), "the one hundred fifth time");
    }

    #[test]
    fn a_year_is_read_as_a_year() {
        assert_eq!(to_speakable("in 1990"), "in nineteen ninety");
        assert_eq!(to_speakable("in 2024"), "in twenty twenty four");
        assert_eq!(to_speakable("in 1905"), "in nineteen oh five");
        assert_eq!(to_speakable("by 1900"), "by nineteen hundred");
        // The first decade of the century is said the other way.
        assert_eq!(to_speakable("in 2005"), "in two thousand five");
    }

    #[test]
    fn money_is_said_amount_first() {
        assert_eq!(to_speakable("it costs $5"), "it costs five dollars");
        assert_eq!(to_speakable("only $1.50"), "only one dollar fifty");
        assert_eq!(to_speakable("₹500 each"), "five hundred rupees each");
        assert_eq!(to_speakable("£2.00 flat"), "two pounds flat");
        assert_eq!(to_speakable("€3.5"), "three point five euros");
        // A sign on its own is still a word.
        assert_eq!(to_speakable("in $ or ₹"), "in dollars or rupees");
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
