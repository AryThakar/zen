// Chrome AudioWorklet. Plays Zen's voice as one continuous stream from a bounded queue.
//
// Replaces a scheduler that gave every 0.25 s codec block its own AudioBufferSourceNode and
// worked out by hand where each one should start. That design had to be right about three
// separate things forever - the arithmetic of each start time, the rate conversion of each
// block, and the page's main thread never being late - and it is the main thread that also
// renders the orb and the transcript. A block arriving after the previous one finished leaves
// a hole in the middle of a word that scheduling it "immediately" cannot fill, because the
// moment it belonged in has already gone.
//
// Here there are no start times at all. The page pushes samples in, this pulls 128 out per
// render quantum on the audio thread, and the join between one block and the next is nothing
// more than the next read. A stall on the main thread costs buffer depth rather than audio,
// and when depth does run out it is counted and reported rather than silently heard.
class Render extends AudioWorkletProcessor {
  /// Ramp applied to what is already in flight when a turn is abandoned. Long enough not to
  /// click, short enough that barge-in still feels immediate.
  static CUT = 0.004;
  /// How often the page is told where playback has reached, in render quanta. Roughly every
  /// 20 ms, which is the rate the page already works at.
  static REPORT = 8;

  constructor() {
    super();
    /// Blocks of the current turn, oldest first, at the context's own rate.
    this.blocks = [];
    /// Read position inside `blocks[0]`.
    this.offset = 0;
    /// Frames waiting across every block.
    this.queued = 0;
    /// What is left of an abandoned turn: its own queue, ramping to nothing.
    ///
    /// Kept apart from `blocks` rather than faded in place. The page sends the next turn's
    /// audio as soon as it has any, which can be while this is still ramping; sharing one
    /// queue meant the discard at the end of the ramp threw away the new turn's first blocks
    /// along with the old turn's tail, and the reply lost its opening words.
    this.tail = [];
    this.tailOffset = 0;
    this.cut = 0;
    this.cutLength = 0;
    /// Frames of real audio handed to the device since the current turn began. The page maps
    /// this back to what it sent, so it must count only audio - never the silence below.
    this.consumed = 0;
    /// Frames to gather before starting, and again after running dry. Buffering is a state
    /// rather than a guess: while this is unmet the output is silence and nothing is lost.
    this.target = 0;
    this.filling = true;
    /// Times the queue ran dry with the stream still open. The number that says whether
    /// delivery is keeping up; it is reported, never hidden.
    this.underruns = 0;
    this.epoch = 0;
    this.ticks = 0;
    /// Frames of fade-in still to apply. Set wherever playback restarts from a silence it did
    /// not choose - after running dry mid-phrase, and at the start of a new turn - so that the
    /// restart is not heard as a click.
    this.resumeFade = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "push") {
        const block = new Float32Array(data.pcm);
        if (block.length) {
          this.blocks.push(block);
          this.queued += block.length;
        }
      } else if (data.type === "target") {
        this.target = Math.max(0, data.frames | 0);
      } else if (data.type === "clear") {
        // Everything queued becomes tail and is ramped away; the turn replacing it starts from
        // an empty queue, so anything it pushes from here on cannot be caught by the discard.
        this.tail = this.blocks;
        this.tailOffset = this.offset;
        this.cutLength = Math.min(
          Math.round(Render.CUT * sampleRate),
          this.queued,
        );
        this.cut = this.cutLength;
        this.blocks = [];
        this.offset = 0;
        this.queued = 0;
        this.filling = true;
        this.consumed = 0;
        this.resumeFade = Math.round(Render.CUT * sampleRate);
        this.epoch = data.epoch | 0;
        this.report();
      }
    };
  }

  /// Copies up to `count` frames out of a queue into `out` at `from`, advancing that queue.
  /// Returns how many it actually had.
  drain(cursor, out, from, count) {
    let written = 0;
    while (written < count && cursor.blocks.length) {
      const block = cursor.blocks[0];
      const run = Math.min(count - written, block.length - cursor.offset);
      out.set(
        block.subarray(cursor.offset, cursor.offset + run),
        from + written,
      );
      cursor.offset += run;
      written += run;
      if (cursor.offset === block.length) {
        cursor.blocks.shift();
        cursor.offset = 0;
      }
    }
    return written;
  }

  take(out, from, count) {
    const cursor = { blocks: this.blocks, offset: this.offset };
    const written = this.drain(cursor, out, from, count);
    this.offset = cursor.offset;
    this.queued -= written;
    return written;
  }

  report() {
    this.port.postMessage({
      type: "progress",
      epoch: this.epoch,
      consumed: this.consumed,
      queued: this.queued,
      underruns: this.underruns,
      filling: this.filling,
    });
  }

  /// Reports at most once every `REPORT` quanta, so an idle renderer does not spend the page's
  /// main thread on several hundred messages a second saying that nothing has changed.
  maybeReport() {
    if (++this.ticks % Render.REPORT === 0) this.report();
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;

    if (this.cut > 0) {
      // Never read past the end of the ramp. A frame taken after the ramp has finished would
      // go out at its original amplitude - a burst of the abandoned turn at full volume, which
      // is the one thing the ramp exists to prevent.
      const cursor = { blocks: this.tail, offset: this.tailOffset };
      const had = this.drain(cursor, out, 0, Math.min(out.length, this.cut));
      this.tail = cursor.blocks;
      this.tailOffset = cursor.offset;
      for (let i = 0; i < had; i++) {
        this.cut--;
        out[i] *= this.cut / this.cutLength;
      }
      // Everything after the ramp is silence until the next turn has buffered.
      out.fill(0, had);
      if (this.cut <= 0 || had < Math.min(out.length, this.cutLength)) {
        this.cut = 0;
        this.tail = [];
        this.tailOffset = 0;
      }
      this.maybeReport();
      return true;
    }

    if (this.filling && this.queued < this.target) {
      out.fill(0);
      this.maybeReport();
      return true;
    }
    this.filling = false;

    const had = this.take(out, 0, out.length);
    this.consumed += had;
    const fadeLength = Math.max(1, Math.round(Render.CUT * sampleRate));
    for (let i = 0; i < had && this.resumeFade > 0; i++) {
      out[i] *= 1 - this.resumeFade / fadeLength;
      this.resumeFade--;
    }
    // An exhausted open stream needs a soft edge even when it ends exactly on a
    // render quantum. Fade only the available tail; never repeat or invent speech.
    const shortfall = this.target > 0 && this.queued === 0;
    if (shortfall && had) {
      const fade = Math.min(had, fadeLength);
      for (let i = had - fade; i < had; i++) out[i] *= (had - 1 - i) / fade;
    }
    if (had < out.length) {
      out.fill(0, had);
      this.filling = true;
    }
    if (shortfall) {
      this.filling = true;
      this.resumeFade = fadeLength;
      this.underruns++;
      this.report();
      return true;
    }
    this.maybeReport();
    return true;
  }
}
registerProcessor("zen-render", Render);
