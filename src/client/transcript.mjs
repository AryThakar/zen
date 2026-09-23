/// Sentences as a reader would break them: ICU's rules.
const SENTENCES = new Intl.Segmenter("en", { granularity: "sentence" });
/// Short forms a sentence runs on through. ICU knows them only through its list of exceptions,
/// which the engine this page runs in does not ship, so without this "Dr. Rao" went up as "Dr."
/// and then "Rao", a sentence each.
const RUNS_ON = /(?:^|\s)(?:mr|mrs|ms|dr|prof|st|mt|vs|e\.g|i\.e)\.\s*$/i;

/// `text` in sentences, each with where it starts in the text.
function sentences(text) {
  const found = [];
  for (const { segment, index } of SENTENCES.segment(text)) {
    const last = found.at(-1);
    if (last && RUNS_ON.test(last.text)) last.text += segment;
    else found.push({ text: segment, at: index });
  }
  return found.filter(({ text }) => text.trim()).map(({ text, at }) => ({ text: text.trim(), at }));
}

/// How early a sentence goes up, in milliseconds. Where a sentence begins is an estimate from its
/// share of the phrase, so it is put up a moment ahead rather than a moment late.
const LEAD_MS = 150;

// Display hypotheses separately from committed, audibly completed conversation entries.
export class Transcript {
  /// How far from the bottom still counts as following the conversation, in pixels. Generous,
  /// because a view that is one line short of the end is being read at the end.
  static FOLLOW_MARGIN = 48;

  /// Zen's pace before any of its speech has been timed, in characters a millisecond.
  /// Conversational English runs at about 150 words a minute, and a word with its space averages
  /// about six characters: some 15 characters a second. Replaced by the measured pace as soon as a
  /// phrase has played.
  static PACE = 15 / 1000;

  /// `list` holds the history and `scroller` is what scrolls it; the rest is the board: the box,
  /// its speaker line, the words, and the frame they scroll in.
  constructor({ list, scroller, caption, speaker, text, frame }) {
    Object.assign(this, { list, scroller, caption, speaker, text, frame });
    this.entries = [];
    this.pace = Transcript.PACE;
    /// The reply on the board: its generation, and each phrase's sentences still to go up.
    this.reply = null;
    this.reset();
  }

  /// Whether the board is where it was last put, or on its way there, so what is said next should
  /// be followed. Scrolled up above that, it is being read, and it is left alone.
  following() {
    return this.frame.scrollTop >= Math.min(this.from, this.rest) - Transcript.FOLLOW_MARGIN;
  }

  /// The board shows one thing from one speaker: what you said, a note, or a fresh page.
  show(who, text, partial = false) {
    this.reply = null;
    this.caption.dataset.speaker = who;
    this.caption.classList.toggle("partial", partial);
    this.speaker.textContent =
      who === "You" ? (partial ? "You · listening" : "You") : who;
    if (this.text.textContent !== text) {
      this.text.textContent = text;
      this.arrive(this.text);
    }
    this.top();
  }

  /// A phrase of Zen's has become audible. Its sentences go up one at a time as they are said,
  /// under the rest of the same reply.
  say(generation, phrase, text) {
    if (this.reply?.generation !== generation) {
      this.reply = { generation, phrases: new Map() };
      this.caption.dataset.speaker = "Zen";
      this.caption.classList.remove("partial");
      this.speaker.textContent = "Zen";
      this.text.replaceChildren();
      this.top();
    }
    this.reply.phrases.set(phrase, { sentences: sentences(text), length: Math.max(1, text.length) });
    this.reach(phrase, 0, null);
  }

  /// Playback is `heard` milliseconds into `phrase`, of `length` once synthesis has finished it.
  /// Every sentence that has begun by now goes up. Until the length is known, it is estimated
  /// from Zen's measured pace.
  reach(phrase, heard, length) {
    const entry = this.reply?.phrases.get(phrase);
    if (!entry) return;
    const total = length ?? entry.length / this.pace;
    const reached = (heard + LEAD_MS) / total;
    while (entry.sentences.length && entry.sentences[0].at / entry.length <= reached)
      this.put(entry.sentences.shift(), phrase);
    this.follow(phrase, reached * entry.length);
  }

  /// The phrase has been said: whatever of it is not up yet goes up.
  said(phrase) {
    const entry = this.reply?.phrases.get(phrase);
    if (!entry) return;
    while (entry.sentences.length) this.put(entry.sentences.shift(), phrase);
    this.follow(phrase, entry.length);
    this.reply.phrases.delete(phrase);
  }

  /// The reply was cut off. What was said stays; what never was does not go up.
  stop() {
    if (this.reply) this.reply.phrases.clear();
  }

  /// One sentence of `phrase` onto the board. The one being said is at full strength and those
  /// before it step back.
  put(sentence, phrase) {
    this.text.lastElementChild?.classList.add("said");
    if (this.text.childElementCount) this.text.append(" ");
    const line = document.createElement("span");
    line.className = "sentence";
    line.textContent = sentence.text;
    this.text.append(line);
    this.arrive(line);
    this.reading = { phrase, line, at: sentence.at, length: sentence.text.length };
  }

  /// Reads down the board as Zen speaks, the way a prompter does. A reply starts at the top and
  /// nothing moves while the words being said sit above the last line; when they reach it, the
  /// board glides up until they are on the second, with one line of what was just said above
  /// and room below. It used to jump to the end on every sentence, so past four lines each new
  /// one appeared pinned to the bottom edge. `spoken` is how far into the phrase Zen has got, in
  /// characters.
  follow(phrase, spoken) {
    const reading = this.reading;
    if (reading?.phrase !== phrase || !reading.line.isConnected || !this.following()) return;
    const at = Math.min(reading.length - 1, Math.max(0, Math.round(spoken - reading.at)));
    const range = document.createRange();
    range.setStart(reading.line.firstChild, at);
    range.setEnd(reading.line.firstChild, at + 1);
    const glyph = range.getClientRects()[0];
    if (!glyph) return;
    const frame = this.frame;
    const height = parseFloat(getComputedStyle(this.text).lineHeight) || glyph.height;
    // The top of the line the word is on, in the board's own coordinates - counted in whole lines
    // from the top of the text rather than taken from the glyph, whose box sits a pixel or two off
    // its line. Off by that much, the tail of a letter on the line above showed at the top edge.
    const view = frame.getBoundingClientRect().top - frame.scrollTop;
    const first = this.text.getBoundingClientRect().top - view;
    const line = first + Math.round((glyph.top + glyph.height / 2 - view - first - height / 2) / height) * height;
    // Judged against where the board is going, not where a glide has got to: sentences can go up
    // faster than it moves.
    if (line + height < this.rest + frame.clientHeight - height / 2) return;
    const top = Math.max(0, line - height);
    if (top <= this.rest) return;
    // Room to glide that far, though nothing is written below the words yet.
    const padding = parseFloat(this.text.style.paddingBottom) || 0;
    const room = top + frame.clientHeight - (frame.scrollHeight - padding);
    this.text.style.paddingBottom = room > 0 ? `${Math.ceil(room)}px` : "";
    this.from = frame.scrollTop;
    this.rest = top;
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    frame.scrollTo({ top, behavior: still ? "auto" : "smooth" });
  }

  /// A fresh board: back to the top, with nothing held below.
  top() {
    this.from = 0;
    this.rest = 0;
    this.reading = null;
    this.text.style.paddingBottom = "";
    this.frame.scrollTop = 0;
  }

  /// Back to the last thing said, when what was being heard came to nothing.
  recall() {
    const last = this.entries.at(-1);
    if (last) this.show(last.who, last.body.textContent);
    else this.show("All the time you need", "Go ahead. I’m listening.");
  }

  /// Each new line settles in rather than appearing in a blink.
  arrive(element) {
    element.classList.remove("arrive");
    void element.offsetWidth;
    element.classList.add("arrive");
  }

  /// A phrase of `characters` took `ms` to play. Zen's pace is learned from its own speech.
  measured(characters, ms) {
    if (characters < 20 || !(ms > 0)) return;
    const pace = characters / ms;
    this.pace += (pace - this.pace) * 0.3;
  }

  append(who, text, generation) {
    if (!text.trim()) return;
    // Whether to follow the conversation down is decided before anything is added, because
    // afterwards the answer is always "no, the view is not at the bottom any more". Someone who
    // has scrolled up is reading something; a reply arriving every few seconds used to drag
    // them back to the end each time, which makes the history unreadable while Zen is talking.
    const follow =
      this.scroller.scrollHeight -
        this.scroller.scrollTop -
        this.scroller.clientHeight <
      Transcript.FOLLOW_MARGIN;
    this.list.querySelector(".history-empty")?.remove();
    const last = this.entries.at(-1);
    if (
      who === "Zen" &&
      last?.who === who &&
      last.generation === generation &&
      last.body.textContent.length + text.length < 32768
    ) {
      last.body.textContent += ` ${text}`;
    } else {
      const row = document.createElement("div");
      row.className = `history-entry ${who === "You" ? "msg-you" : "msg-zen"}`;
      row.dataset.speaker = who;
      if (who !== "You") {
        const label = document.createElement("div");
        label.className = "who";
        label.textContent = who;
        row.append(label);
      }
      const body = document.createElement("p");
      body.textContent = text;
      row.append(body);
      this.list.append(row);
      this.entries.push({ row, body, who, generation });
      while (this.entries.length > 40) this.entries.shift().row.remove();
    }
    if (follow) this.scroller.scrollTop = this.scroller.scrollHeight;
  }

  /// Ends a reply that was cut off: what was certainly heard of the phrase playing at the cut,
  /// then the dash that marks it. A reply nobody heard any of adds nothing.
  cut(who, heard, generation) {
    const last = this.entries.at(-1);
    const continuing = last?.who === who && last.generation === generation;
    if (!heard.trim() && !continuing) return;
    this.append(who, `${heard.trim()} —`.trim(), generation);
  }

  reset() {
    this.entries = [];
    this.list.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "A fresh page. Say hello.";
    this.list.append(empty);
    this.reply = null;
    this.caption.classList.remove("partial");
    delete this.caption.dataset.speaker;
    this.speaker.textContent = "Your conversation starts here";
    this.text.textContent = "What’s on your mind?";
    this.top();
  }
}
