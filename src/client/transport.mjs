// A session owns one ordered IPC pump. Retiring it cannot drain a newer session.

/// Frame kind byte for capture audio, the only kind that may ever be shed.
const AUDIO_KIND = 0;
/// Queued capture frames tolerated before the oldest are dropped: one second at 20 ms each.
const MAX_QUEUED_AUDIO = 50;

/// Controls that describe the *output* stream rather than the input one, and so have no
/// ordering relationship with captured speech.
///
/// These are what the engine waits on before it will send more audio: it stops at about two
/// seconds unacknowledged. Behind a second of queued capture they arrive late, the engine
/// stops feeding the player, and the player runs dry mid-sentence - a queue on the way in
/// starving the way out. Nothing about them is sequenced against speech, so they go first.
///
/// Deliberately not included: end_audio, which marks where an utterance stopped and means
/// nothing if it overtakes the frames it refers to; text, for the same reason; and interrupt,
/// which the page already acts on locally and whose arrival before queued speech would change
/// which turn that speech belongs to.
export const PRIORITY_CONTROLS = new Set(["audio_played", "playback_started", "played"]);

export class OrderedInput {
  constructor(invoke, revision, onError, { timeout = 5000, onShed = () => {} } = {}) {
    this.invoke = invoke;
    this.revision = revision;
    this.onError = onError;
    this.onShed = onShed;
    this.timeout = timeout;
    this.queue = [];
    this.priority = [];
    this.running = false;
    this.closed = false;
    this.dropped = 0;
  }

  send(kind, payload, priority = false) {
    if (this.closed) return false;
    const bytes = new Uint8Array(payload.buffer ?? payload,
      payload.byteOffset ?? 0, payload.byteLength);
    const frame = new Uint8Array(9 + bytes.byteLength);
    new DataView(frame.buffer).setBigUint64(0, BigInt(this.revision), true);
    frame[8] = kind;
    frame.set(bytes, 9);
    if (priority) this.priority.push(frame);
    else this.queue.push(frame);
    // Never remove the in-flight frame or a control. Only queued capture can expire.
    // One pass: this runs on the 20 ms capture path, and it runs precisely when the machine
    // is already too busy to keep up, so re-scanning the queue once per dropped frame spends
    // the most time exactly where there is least to spare.
    const before = this.dropped;
    let audio = 0;
    for (const item of this.queue) if (item[8] === AUDIO_KIND) audio++;
    if (audio > MAX_QUEUED_AUDIO) {
      let excess = audio - MAX_QUEUED_AUDIO;
      this.queue = this.queue.filter((item) => {
        if (excess && item[8] === AUDIO_KIND) {
          excess--;
          return false;
        }
        return true;
      });
      this.dropped += audio - MAX_QUEUED_AUDIO;
    }
    // Shedding is speech being thrown away - 20 ms of it per frame. Recognition gets worse
    // and nothing else in the pipeline can tell that it did, so it must not be silent.
    if (this.dropped !== before) this.onShed(this.dropped);
    // Both lanes count. Bounding only the ordered one would let a stalled connection collect
    // acknowledgements without limit, which is the lane that exists to stay short.
    if (this.queue.length + this.priority.length > 256) {
      this.fail(new Error("Engine input queue is full."));
      return false;
    }
    void this.pump();
    return true;
  }

  async pump() {
    if (this.running || this.closed) return;
    this.running = true;
    try {
      while ((this.priority.length || this.queue.length) && !this.closed) {
        // Acknowledgements first, always. They are a handful a second against fifty frames of
        // capture, so nothing behind them can be starved by letting them past.
        const frame = this.priority.length
          ? this.priority.shift()
          : this.queue.shift();
        let timer;
        try {
          await Promise.race([
            this.invoke("zen_input", frame),
            new Promise((_, reject) => {
              timer = setTimeout(() => reject(new Error("Engine input stalled.")), this.timeout);
            }),
          ]);
        } finally { clearTimeout(timer); }
      }
    } catch (error) {
      this.fail(error);
    } finally { this.running = false; }
  }

  fail(error) {
    if (this.closed) return;
    this.close();
    this.onError(error);
  }

  close() {
    this.closed = true;
    this.queue = [];
    this.priority = [];
  }
}
