// A session owns one ordered IPC pump. Retiring it cannot drain a newer session.

/// Frame kind byte for capture audio, the only kind that may ever be shed.
const AUDIO_KIND = 0;
/// Queued capture frames tolerated before the oldest are dropped: one second at 20 ms each.
const MAX_QUEUED_AUDIO = 50;

export class OrderedInput {
  constructor(invoke, revision, onError, { timeout = 5000, onShed = () => {} } = {}) {
    this.invoke = invoke;
    this.revision = revision;
    this.onError = onError;
    this.onShed = onShed;
    this.timeout = timeout;
    this.queue = [];
    this.running = false;
    this.closed = false;
    this.dropped = 0;
  }

  send(kind, payload) {
    if (this.closed) return false;
    const bytes = new Uint8Array(payload.buffer ?? payload,
      payload.byteOffset ?? 0, payload.byteLength);
    const frame = new Uint8Array(9 + bytes.byteLength);
    new DataView(frame.buffer).setBigUint64(0, BigInt(this.revision), true);
    frame[8] = kind;
    frame.set(bytes, 9);
    this.queue.push(frame);
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
    if (this.queue.length > 256) {
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
      while (this.queue.length && !this.closed) {
        const frame = this.queue.shift();
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
  }
}
