import { kaiserSinc } from "./sinc.mjs";

/// Band-limited conversion from the synthesiser's rate to the device's, continuous across
/// blocks.
///
/// The renderer plays at the context's rate, so something has to convert. Left to WebAudio, a
/// buffer at another rate is interpolated one buffer at a time. Measured in the engine this app
/// runs on: a 6 kHz tone in a 24 kHz buffer played at 48 kHz came back with an image at 18 kHz
/// only 15.3 dB below the tone, against 156 dB down for a buffer already at the context rate;
/// and each buffer's last frame, interpolated against nothing beyond it, overshot - a steady
/// tone in 0.25 s blocks read -1.2 where the waveform was -0.4, once per block. Both were well
/// above the range of speech rather than the cause of anything heard in it, but neither belongs
/// in the output.
///
/// Converting here, with the filter's history carried from one block to the next, joins each
/// block to its neighbour exactly: the same test reproduces the ideal waveform to the last bit.
export class Resampler {
  /// Kernel width in source samples. Sixteen either side of the output instant puts the
  /// stopband far enough down that the conversion is inaudible.
  static TAPS = 32;
  /// Fractional positions the kernel is tabulated at. Positions in between are interpolated
  /// from the two neighbouring rows, which costs one extra multiply and removes the stepping a
  /// table alone would leave.
  static PHASES = 512;
  /// Kaiser shape, the same family and the same reasoning as the true-peak detector in the
  /// engine: chosen for stopband depth rather than for the narrowest transition.
  static BETA = 9;

  constructor(from, to) {
    this.from = from;
    this.to = to;
    this.step = from / to;
    this.identity = from === to;
    // Downsampling has to filter to the *destination* Nyquist or it aliases; upsampling only
    // has to reconstruct up to the source Nyquist.
    if (!this.identity) this.table = Resampler.kernel(Math.min(1, to / from));
    this.reset();
  }

  static kernel(cutoff) {
    const { TAPS, PHASES, BETA } = Resampler;
    return kaiserSinc({ taps: TAPS, phases: PHASES, cutoff, beta: BETA });
  }

  /// Starts a new stream. Anything the filter was still holding is dropped, so this belongs at
  /// a phrase boundary and nowhere inside one.
  reset() {
    this.held = new Float32Array(0);
    /// Absolute source index of `held[0]`.
    this.base = 0;
    /// Absolute source position the next output sample is taken at.
    this.position = 0;
  }

  /// Pushes the kernel through to the end of the stream and hands back what it was still
  /// holding - about 0.7 ms. Without this the end of every phrase stays inside the filter.
  finish() {
    if (this.identity) return new Float32Array(0);
    const out = this.process(new Float32Array(Resampler.TAPS / 2));
    this.reset();
    return out;
  }

  /// Converts one block, keeping whatever the kernel still needs for the next.
  process(input) {
    if (this.identity) return input;
    const { TAPS, PHASES } = Resampler;
    const half = TAPS / 2;
    const held = new Float32Array(this.held.length + input.length);
    held.set(this.held);
    held.set(input, this.held.length);
    this.held = held;

    // An output at `position` reads source samples up to floor(position) + half, so the last
    // few source samples of every block wait for the block after it. In the steady state that
    // costs nothing; it means only that a block's output is not a fixed length.
    const newest = this.base + held.length - 1;
    const count = Math.max(
      0,
      Math.floor((newest - half - this.position) / this.step) + 1,
    );
    const out = new Float32Array(count);
    for (let k = 0; k < count; k++) {
      const index = Math.floor(this.position);
      const phase = (this.position - index) * PHASES;
      const row = Math.min(PHASES - 1, Math.floor(phase));
      const blend = phase - row;
      const near = row * TAPS,
        far = near + TAPS;
      const start = index - half + 1 - this.base;
      let sum = 0;
      for (let tap = 0; tap < TAPS; tap++) {
        const at = start + tap;
        // Before the stream began, and past what has arrived, there is silence.
        const sample = at < 0 || at >= held.length ? 0 : held[at];
        sum +=
          sample *
          (this.table[near + tap] * (1 - blend) + this.table[far + tap] * blend);
      }
      out[k] = sum;
      this.position += this.step;
    }

    const drop = Math.floor(this.position) - half + 1 - this.base;
    if (drop > 0) {
      this.held = this.held.subarray(drop);
      this.base += drop;
    }
    return out;
  }
}

// Audio and device identifiers are held only for the lifetime of a connection.
//
// Zen's voice is one stream, not a series of clips. The page converts each block that arrives
// to the device's rate and hands it to `render.js`, which plays it from a queue on the audio
// thread; nothing here decides when a block starts, because within a phrase there is no such
// moment to decide - the next block simply continues the last. Everything this class still
// does is bookkeeping about that stream: which part of it has been heard, what to credit the
// engine for, and when a phrase began and ended.
//
// Positions are measured in frames of the stream rather than in seconds on the clock, so a
// late message cannot move an event that has already been decided.
export class AudioPlayback {
  /// Audio gathered before playback starts, and again after the queue has run dry.
  ///
  /// Buffering is a state rather than a hope: below this the renderer emits silence and keeps
  /// what it has. Starting it low keeps the reply prompt; it grows when delivery proves it
  /// needs to, which is the only evidence worth sizing it from. When a stream starts, the
  /// playout delay below decides first.
  static STARTUP_BUFFER = 0.1;
  /// Added each time the queue runs dry, up to the cap. Recovering to the same depth that has
  /// just been shown to be too shallow only buys another underrun.
  static BUFFER_STEP = 0.04;
  static MAX_BUFFER = 0.32;
  /// The rate synthesis produces.
  static RATE = 24000;
  /// Ramped off the end of a phrase. Synthesis stops when it runs out of text, sometimes with
  /// the waveform still at a fifth of full scale, which is heard as the last sound being cut.
  ///
  /// This much of the stream is held back rather than pushed, so that when the phrase does end
  /// there is always something left to apply the fade to. Holding it costs six milliseconds of
  /// delay; not holding it made the fade conditional on a message arriving before the audio it
  /// was meant to shape had already gone out, which is not a guarantee at all.
  static PHRASE_FADE = 0.006;
  /// How far delivery ran behind real time at the start of three replies, in milliseconds,
  /// measured in the running app on 2026-09-17.
  ///
  /// Synthesis hands over the start of a reply in growing lumps - 80, 160 and 320 ms of audio -
  /// and the next, 640 ms, comes about 370 ms after the last. A depth target alone cannot see
  /// that coming: 100 ms was already queued when the second lump landed, so playback began
  /// there, used up all 560 ms, and ran dry a few milliseconds before the fourth arrived. That
  /// was a gap in a word half a second into most replies, and deepening the target did not move
  /// it, because the second lump still cleared the deeper target. So the start of a stream is
  /// held by time instead. These seed that until the session has measured its own.
  static SEED_LAGS_MS = [149, 118, 89];
  /// Streams remembered when sizing the playout delay.
  static PLAYOUT_HISTORY = 8;
  /// The longest the start of a stream is held, however late delivery has been.
  static MAX_PLAYOUT_MS = 1000;

  constructor(context, send, onPhrase = () => {}, onError = () => {}) {
    this.context = context;
    this.send = send;
    this.onPhrase = onPhrase;
    this.onError = onError;
    this.analyser = context.createAnalyser();
    this.analyser.fftSize = 256;
    /// Muting turns this down, after the meter: the orb still moves with a voice nobody hears,
    /// which is how the page shows that Zen is talking while muted.
    this.volume = context.createGain();
    this.analyser.connect(this.volume).connect(context.destination);
    this.meter = new Float32Array(256);
    this.phrases = new Map();
    this.markers = [];
    this.generation = 0;
    this.epoch = 0;
    // Set here as well as in `reset`, so the ordering check below is a comparison against a
    // number from the very first block rather than against `undefined`, which nothing is
    // less than or equal to.
    this.lastSequence = 0;
    this.accepting = true;
    /// Frames of the stream handed to the renderer, and the position it reports having played.
    this.pushed = 0;
    this.played = 0;
    /// When each reported position will actually have been heard, oldest first.
    this.arrivals = [];
    this.heardFrames = 0;
    this.queuedFrames = 0;
    this.underruns = 0;
    this.buffer = AudioPlayback.STARTUP_BUFFER;
    /// How far behind real time delivery ran in recent streams. See `playoutDelay`.
    this.lags = [...AudioPlayback.SEED_LAGS_MS];
    /// The stream now arriving, if it started from silence: when its first audio came, how much
    /// audio has come since, and the furthest delivery has fallen behind.
    this.stream = null;
    this.holdTimer = null;
    /// The end of the stream, withheld so a phrase can always be faded out properly.
    this.hold = new Float32Array(0);
    /// Converted blocks waiting for the renderer, or for the start of the stream to be released.
    this.waiting = null;
    /// The phrase the converter's history belongs to.
    this.lastPhrase = 0;
    this.resampler = new Resampler(AudioPlayback.RATE, context.sampleRate);
    this.node = null;
    this.loading = null;
    this.timer = setInterval(() => this.tick(), 20);
    this.open();
  }

  /// Brings the renderer up. Blocks that arrive first are converted and held until it exists,
  /// so a session does not have to wait on a module load before it can be spoken to.
  open() {
    if (this.node || this.loading) return this.loading;
    this.loading = this.context.audioWorklet
      .addModule("/render.js")
      .then(() => {
        if (!this.timer) return;
        const node = new AudioWorkletNode(this.context, "zen-render", {
          numberOfInputs: 0,
          outputChannelCount: [1],
        });
        node.port.onmessage = ({ data }) => this.progress(data);
        node.onprocessorerror = () => this.fail(new Error("Voice playback stopped. Please try again."));
        node.connect(this.analyser);
        this.node = node;
        // A session can be reset before the module has finished loading, so the renderer has
        // to be told which turn it is serving before it is given anything to play. Without
        // this it reports progress under an epoch the page has already moved past, and every
        // report is discarded as stale - which is to say no audio is ever credited.
        node.port.postMessage({ type: "clear", epoch: this.epoch });
        this.aim();
        this.flush();
      })
      .catch((error) => { if (this.timer) this.fail(error); });
    return this.loading;
  }

  fail(error) {
    const node = this.node;
    this.node = null;
    this.loading = null;
    if (node) {
      node.onprocessorerror = null;
      node.port.onmessage = null;
      node.disconnect();
    }
    this.clear(false);
    this.onError(error);
  }

  progress(data) {
    if (data.type !== "progress" || data.epoch !== this.epoch) return;
    this.played = data.consumed;
    // Frames pulled now reach the speaker one output latency from now, so what matters is
    // recorded as a time rather than as a distance behind the head of the stream. Measuring it
    // as "consumed minus latency" cannot finish a reply: consumed stops rising the moment the
    // queue empties, so the last latency-worth of frames is never reached, the final phrase
    // never completes, and Zen is left in its speaking state with the answer missing from
    // history. The device keeps emptying on the clock whether or not anything else is sent.
    const due = this.context.currentTime + (this.context.outputLatency ?? 0.04);
    const last = this.arrivals[this.arrivals.length - 1];
    if (this.arrivals.length >= 512 && last) {
      last.at = Math.max(last.at, due);
      last.frames = data.consumed;
    } else this.arrivals.push({ at: due, frames: data.consumed });
    this.queuedFrames = data.queued;
    if (data.underruns > this.underruns) {
      this.underruns = data.underruns;
      // Delivery has proved the buffer too shallow. Deepen it rather than rebuffering to the
      // same depth and waiting for the same thing to happen again.
      this.buffer = Math.min(
        AudioPlayback.MAX_BUFFER,
        this.buffer + AudioPlayback.BUFFER_STEP,
      );
      this.aim();
    }
  }

  /// Tells the renderer how much to gather before it plays, and whether more is coming at all.
  /// With nothing outstanding the target is zero, so a reply simply ending is not mistaken for
  /// the queue running dry.
  aim() {
    if (!this.node) return;
    const open = [...this.phrases.values()].some((phrase) => !phrase.ended);
    this.node.port.postMessage({
      type: "target",
      frames: open ? Math.round(this.buffer * this.context.sampleRate) : 0,
    });
  }

  /// Silences Zen's voice, or brings it back. Ramped over a few milliseconds, because a level
  /// that jumps is heard as a click; the reply itself carries on, heard or not.
  mute(muted) {
    this.volume.gain.setTargetAtTime(muted ? 0 : 1, this.context.currentTime, 0.012);
  }

  /// How far into `phrase` playback has reached, and how long the phrase is once synthesis has
  /// finished it, both in milliseconds. Null before its first sound and after its last.
  position(phrase) {
    const entry = this.phrases.get(phrase);
    if (!entry?.started) return null;
    const rate = this.context.sampleRate;
    return {
      heard: (Math.max(0, this.heard() - entry.from) / rate) * 1000,
      length: entry.to === undefined ? null : ((entry.to - entry.from) / rate) * 1000,
    };
  }

  hand(block) {
    this.node.port.postMessage({ type: "push", pcm: block.buffer }, [
      block.buffer,
    ]);
  }

  /// Adds samples to the stream and returns the frame position of its new end.
  push(samples) {
    if (!samples.length) return this.pushed;
    this.pushed += samples.length;
    (this.waiting ||= []).push(new Float32Array(samples));
    this.flush();
    return this.pushed;
  }

  /// Hands over what is waiting, once there is a renderer and the start of the stream is no
  /// longer being held.
  flush() {
    if (!this.node || this.holdTimer || !this.waiting?.length) return;
    const blocks = this.waiting;
    this.waiting = null;
    for (const block of blocks) this.hand(block);
  }

  /// How long to hold the start of a stream, in milliseconds: the furthest delivery has run
  /// behind real time in recent streams, plus how much that has varied between them. Starting
  /// that late means the audio already delivered lasts until the rest arrives.
  playoutDelay() {
    const worst = Math.max(...this.lags);
    const best = Math.min(...this.lags);
    return Math.min(AudioPlayback.MAX_PLAYOUT_MS, worst + (worst - best));
  }

  /// Notes the arrival of `ms` of audio. A stream that starts from silence is held for the
  /// playout delay, and how far its delivery falls behind is measured as it goes.
  arrive(ms) {
    const now = performance.now();
    if (!this.stream && this.pending() === 0) {
      this.stream = { start: now, audio: 0, lag: 0 };
      const delay = this.playoutDelay();
      if (delay > 0)
        this.holdTimer = setTimeout(() => {
          this.holdTimer = null;
          this.flush();
        }, delay);
    }
    if (!this.stream) return;
    this.stream.lag = Math.max(this.stream.lag, now - this.stream.start - this.stream.audio);
    this.stream.audio += ms;
  }

  /// A stream that has played out completely teaches the playout delay how late its delivery
  /// ran. One cut short says nothing reliable and is dropped.
  settle() {
    if (!this.stream || this.holdTimer || this.waiting?.length || this.pending() > 0) return;
    if ([...this.phrases.values()].some((phrase) => !phrase.ended)) return;
    this.lags = [...this.lags, this.stream.lag].slice(-AudioPlayback.PLAYOUT_HISTORY);
    this.stream = null;
  }

  /// Where playback has actually reached, in frames.
  ///
  /// What the renderer has pulled is not yet what anyone has heard: the device holds its own
  /// buffer beyond it. Credit and phrase events are both about what was heard.
  heard() {
    const now = this.context.currentTime;
    while (this.arrivals.length && this.arrivals[0].at <= now)
      this.heardFrames = this.arrivals.shift().frames;
    return this.heardFrames;
  }

  /// Frames of this turn still to come out of the speakers.
  pending() {
    return Math.max(0, this.pushed - this.heard());
  }

  reset(generation) {
    this.clear();
    this.generation = generation;
    this.lastSequence = 0;
    this.accepting = true;
  }

  clear(credit = true) {
    // Anything already heard is owed its credit before the turn is torn down. A phrase can
    // finish playing in the twenty milliseconds between one tick and the next, and dropping its
    // marker here meant a whole phrase the listener had heard appeared nowhere afterwards:
    // not in the transcript on screen, and not in the conversation the model is given, because
    // the engine learns that a phrase was spoken only from this acknowledgement.
    if (credit) this.credit();
    this.epoch++;
    this.accepting = false;
    this.markers = [];
    this.phrases.clear();
    this.hold = new Float32Array(0);
    this.waiting = null;
    this.lastPhrase = 0;
    clearTimeout(this.holdTimer);
    this.holdTimer = null;
    this.stream = null;
    this.resampler.reset();
    this.pushed = 0;
    this.played = 0;
    this.arrivals = [];
    this.heardFrames = 0;
    this.queuedFrames = 0;
    this.node?.port.postMessage({ type: "clear", epoch: this.epoch });
  }

  begin(event) {
    if (!this.accepting || event.generation !== this.generation) return;
    if (!Number.isSafeInteger(event.phrase) || event.phrase < 1 || this.phrases.size >= 128 || this.phrases.has(event.phrase))
      throw new Error("Invalid voice phrase.");
    this.phrases.set(event.phrase, { text: event.text || "", started: false });
    this.aim();
  }

  queue(event) {
    if (!this.accepting || event.generation !== this.generation) return;
    const phrase = this.phrases.get(event.phrase);
    const bytes = event.pcm;
    // Reply audio arrives as raw little-endian PCM on its own channel rather than
    // as base64 inside JSON, so there is nothing to decode before scheduling it.
    if (!phrase || phrase.ended || !Number.isSafeInteger(event.sequence)
      || event.sequence <= this.lastSequence || !(bytes instanceof Uint8Array) || bytes.length > 192000)
      throw new Error("Invalid voice audio.");
    if (!bytes.length || bytes.length % 2 || this.markers.length > 1024)
      throw new Error("Voice playback exceeded its buffer.");
    if (!this.node) this.open();
    // A new phrase is a new stream: the converter's history belongs to the old one, and
    // anything the old one still had held back goes out before this starts.
    if (this.lastPhrase !== event.phrase) {
      this.release(false);
      this.resampler.reset();
      this.lastPhrase = event.phrase;
    }
    this.arrive((bytes.length / 2 / AudioPlayback.RATE) * 1000);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const pcm = new Float32Array(bytes.length / 2);
    for (let i = 0; i < pcm.length; i++)
      pcm[i] = view.getInt16(i * 2, true) / 32768;
    // Converted here so that what the renderer holds is already at the device's rate. See
    // `Resampler` for what WebAudio does when it is left to convert this itself.
    const samples = this.resampler.process(pcm);

    if (!phrase.started) {
      phrase.started = true;
      phrase.from = this.pushed;
      this.markers.push({
        // One frame past where the phrase begins. A marker fires once playback has reached its
        // position, and the phrase has begun only when its first sample has actually been
        // heard - not when the stream has arrived at the point where that sample sits.
        at: this.pushed + 1,
        type: "playback_started",
        phrase: event.phrase,
        text: phrase.text,
        epoch: this.epoch,
      });
    }
    // Everything but the last few milliseconds. What is held is what the phrase will be faded
    // out with, whenever the message saying it ended happens to arrive.
    const carried = new Float32Array(this.hold.length + samples.length);
    carried.set(this.hold);
    carried.set(samples, this.hold.length);
    const keep = Math.min(
      carried.length,
      Math.round(AudioPlayback.PHRASE_FADE * this.context.sampleRate),
    );
    this.hold = carried.slice(carried.length - keep);
    this.push(carried.subarray(0, carried.length - keep));

    this.markers.push({
      at: this.pushed,
      type: "audio_played",
      sequence: event.sequence,
      epoch: this.epoch,
    });
    this.lastSequence = event.sequence;
    this.markers.sort((a, b) => a.at - b.at);
  }

  /// Pushes whatever is being held, faded to silence or not.
  release(fade) {
    if (!this.hold.length) return;
    const tail = this.hold;
    this.hold = new Float32Array(0);
    if (fade)
      for (let i = 0; i < tail.length; i++)
        tail[i] *= 1 - (i + 1) / tail.length;
    this.push(tail);
  }

  end(event) {
    if (!this.accepting || event.generation !== this.generation) return;
    const phrase = this.phrases.get(event.phrase);
    if (!phrase?.started || phrase.ended) throw new Error("Invalid or duplicate voice phrase ending.");
    phrase.ended = true;
    // Synthesis for this phrase is over, so the converter has nothing left to hold its own
    // tail against either. Its last fraction of a millisecond belongs before the fade.
    const drained = this.resampler.finish();
    if (drained.length) {
      const merged = new Float32Array(this.hold.length + drained.length);
      merged.set(this.hold);
      merged.set(drained, this.hold.length);
      this.hold = merged;
    }
    // End on silence. Without this the phrase stops on an edge and the last sound is heard
    // clipped - measured at up to a fifth of full scale on the final sample.
    this.release(true);
    phrase.to = this.pushed;
    // The pause belongs to the stream, so it happens where the audio put it rather than
    // wherever the clock was when this message arrived.
    const pause = Math.max(0, Math.min(500, event.pause_ms || 0)) / 1000;
    this.push(new Float32Array(Math.round(pause * this.context.sampleRate)));
    this.markers.push({
      at: this.pushed,
      type: "played",
      phrase: event.phrase,
      text: event.text || phrase.text,
      epoch: this.epoch,
    });
    this.markers.sort((a, b) => a.at - b.at);
    this.aim();
  }

  /// Reports everything the listener has now heard.
  ///
  /// Returns false if the transport refused, in which case the turn has already been torn down
  /// and nothing after the call may run.
  credit() {
    if (this.context.state !== "running") return true;
    const heard = this.heard();
    while (this.markers.length && this.markers[0].at <= heard) {
      const marker = this.markers.shift();
      if (marker.epoch !== this.epoch) continue;
      const event = { type: marker.type, generation: this.generation };
      if (marker.sequence !== undefined) event.sequence = marker.sequence;
      if (marker.phrase !== undefined) event.phrase = marker.phrase;
      if (this.send(event) === false) {
        this.clear(false);
        return false;
      }
      if (marker.type === "playback_started")
        this.onPhrase("start", marker.text, marker.phrase, this.generation);
      if (marker.type === "played") {
        this.phrases.delete(marker.phrase);
        this.onPhrase("end", marker.text, marker.phrase, this.generation);
        this.aim();
      }
    }
    return true;
  }

  tick() {
    if (this.credit() === false) return;
    this.settle();
  }

  level() {
    if (this.context.state !== "running" || !this.accepting) return 0;
    this.analyser.getFloatTimeDomainData(this.meter);
    return Math.min(
      1,
      Math.sqrt(
        this.meter.reduce((sum, x) => sum + x * x, 0) / this.meter.length,
      ) * 5,
    );
  }

  dispose() {
    clearInterval(this.timer);
    this.timer = null;
    this.clear();
    this.node?.disconnect();
    this.node = null;
    this.analyser.disconnect();
    this.volume.disconnect();
  }
}

export class Microphone {
  constructor(context, { onFrame, onLevel, onState, onError }) {
    this.context = context;
    this.callbacks = { onFrame, onLevel, onState, onError };
    this.revision = 0;
    this.active = false;
    this.pending = false;
    this.loaded = null;
    this.nodes = [];
  }

  async start(deviceId = "") {
    if (this.active || this.pending) return;
    if (
      !globalThis.isSecureContext ||
      !navigator.mediaDevices?.getUserMedia ||
      !this.context.audioWorklet
    ) {
      throw new Error(
        "Microphone capture is unavailable. Open the Zen desktop app with an up-to-date WebView2 runtime.",
      );
    }
    const revision = ++this.revision;
    this.pending = true;
    this.callbacks.onState();
    let stream;
    try {
      await this.context.resume();
      this.loaded ||= this.context.audioWorklet
        .addModule("/capture.mjs")
        .catch((error) => {
          this.loaded = null;
          throw error;
        });
      await this.loaded;
      if (revision !== this.revision) return;
      const supported = navigator.mediaDevices.getSupportedConstraints();
      const audio = { channelCount: { ideal: 1 } };
      for (const key of [
        "echoCancellation",
        "noiseSuppression",
        "autoGainControl",
      ]) {
        if (supported[key]) audio[key] = { ideal: true };
      }
      if (deviceId) audio.deviceId = { exact: deviceId };
      const permission = navigator.mediaDevices.getUserMedia({
        audio,
        video: false,
      }).then((granted) => {
        if (revision !== this.revision) granted.getTracks().forEach((track) => track.stop());
        return granted;
      });
      let timer;
      try {
        stream = await Promise.race([permission, new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error("Microphone access timed out. Check Windows microphone permissions, then try again.")), 30000);
        })]);
      } finally { clearTimeout(timer); }
      // A delayed permission grant must not revive a disconnected or muted session.
      if (revision !== this.revision) {
        return;
      }
      this.stream = stream;
      const track = stream.getAudioTracks()[0];
      if (!track) throw new Error("No microphone track is available.");
      track.contentHint = "speech";
      this.settings = track.getSettings();
      /// The device's own name, which is the only place a Bluetooth hands-free profile shows.
      this.label = track.label || "";
      const source = this.context.createMediaStreamSource(stream);
      this.nodes.push(source);
      // Rumble and the DC offset some microphones carry lie below any voice. Everything above the
      // speech band is the capture worklet's to remove, as part of converting to 16 kHz.
      const high = this.context.createBiquadFilter();
      this.nodes.push(high);
      high.type = "highpass";
      high.frequency.value = 70;
      high.Q.value = 0.707;
      const capture = new AudioWorkletNode(this.context, "zen-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.nodes.push(capture);
      this.capture = capture;
      capture.port.onmessage = ({ data }) => {
        if (revision !== this.revision) return;
        if (data instanceof ArrayBuffer) {
          if (this.active && !this.callbacks.onFrame(data))
            this.fail(
              "The engine is not accepting audio. Your microphone has been paused.",
            );
        } else if (data.type === "level")
          this.callbacks.onLevel(Math.min(1, data.rms * 8));
        else if (data.type === "flushed") this.flushed?.();
      };
      capture.onprocessorerror = () =>
        this.fail(
          "Microphone processing stopped. Start talking again to reconnect it.",
        );
      source.connect(high).connect(capture).connect(this.context.destination);
      this.active = true;
      track.onended = () =>
        this.fail(
          "Your microphone was disconnected. Choose an available microphone.",
        );
      track.onmute = () =>
        this.fail(
          "Your microphone was paused by the system. Start talking again when you are ready.",
        );
    } catch (error) {
      stream?.getTracks().forEach((t) => t.stop());
      if (revision === this.revision) {
        this.stop(false);
        throw error;
      }
    } finally {
      if (revision === this.revision) {
        this.pending = false;
        this.callbacks.onState();
      }
    }
  }

  async stop(flush = true) {
    const revision = this.revision;
    if (flush && this.capture && this.active) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 250);
        this.flushed = () => {
          clearTimeout(timer);
          resolve();
        };
        this.capture.port.postMessage({ type: "flush" });
      });
    }
    if (revision !== this.revision) return;
    this.revision++;
    this.flushed?.();
    this.flushed = null;
    this.active = false;
    this.pending = false;
    if (this.capture) {
      this.capture.port.onmessage = null;
      this.capture.port.close();
      this.capture.onprocessorerror = null;
    }
    this.capture = null;
    for (const node of this.nodes) node.disconnect();
    this.nodes = [];
    for (const track of this.stream?.getTracks() || []) {
      track.onended = null;
      track.onmute = null;
      track.stop();
    }
    this.stream = null;
    this.settings = null;
    this.callbacks.onLevel(0);
    this.callbacks.onState();
  }

  fail(message) {
    this.stop(false);
    this.callbacks.onError(message);
  }
}

/// Soft chimes for the moments a person cannot otherwise see: the microphone opening, the end of
/// the greeting, the microphone going off after quiet, a message sent, a reply stopped.
///
/// Each note is glass-like: a few sine partials, the upper ones fading first, through a gentle
/// low-pass and a short synthetic room. They play through the same context as Zen's voice, so they
/// reach whichever speaker was chosen.
const CHIMES = {
  // [frequency, start offset s, length s, level]
  ready: [[659.25, 0, 1.0, 0.8], [987.77, 0.11, 1.4, 0.7]],
  mic: [[659.25, 0, 0.55, 0.5], [987.77, 0.07, 0.7, 0.45]],
  connect: [[440, 0, 0.6, 0.4]],
  mute: [[587.33, 0, 0.5, 0.5], [440, 0.08, 0.7, 0.45]],
  quiet: [[987.77, 0, 0.9, 0.5], [739.99, 0.14, 1.0, 0.55], [493.88, 0.28, 1.6, 0.6]],
  stop: [[392, 0, 0.45, 0.6]],
  send: [[1174.66, 0, 0.35, 0.35]],
  error: [[349.23, 0, 0.5, 0.5], [329.63, 0.12, 0.7, 0.5]],
};
/// The partials of one note: [multiple of the fundamental, level, share of the note's length].
const PARTIALS = [[1, 1, 1], [2, 0.32, 0.55], [3, 0.1, 0.35], [4.16, 0.04, 0.22]];

export class Soundscape {
  constructor(context) {
    this.context = context;
    this.enabled = true;
    this.nodes = new Set();
    this.out = null;
  }

  /// The shared tail every note runs through, built on first use: the level, a low-pass that takes
  /// the edge off the upper partials, and a short room.
  chain() {
    if (this.out) return this.out;
    const context = this.context;
    const out = context.createGain();
    // About 3 dB above where it was. The loudest cue still peaks well under Zen's voice, and
    // while Zen is speaking every cue is played lower again - see `play`.
    out.gain.value = 0.31;
    const tone = context.createBiquadFilter();
    tone.type = "lowpass";
    tone.frequency.value = 5200;
    const room = context.createConvolver();
    const length = Math.round(context.sampleRate * 1.3);
    const impulse = context.createBuffer(2, length, context.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const data = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 3.4);
    }
    room.buffer = impulse;
    const wet = context.createGain();
    wet.gain.value = 0.2;
    out.connect(tone);
    tone.connect(context.destination);
    tone.connect(room);
    room.connect(wet);
    wet.connect(context.destination);
    return (this.out = out);
  }

  /// `scale` lowers a cue that lands on top of Zen's voice. The two share one output, and a cue at
  /// full level on the loudest syllable of a reply sums past full scale, which is a clip.
  play(kind, scale = 1) {
    if (!this.enabled || this.context.state !== "running" || !CHIMES[kind]) return;
    const out = this.chain();
    const start = this.context.currentTime + 0.02;
    for (const [frequency, offset, length, level] of CHIMES[kind]) {
      const at = start + offset;
      const peak = level * scale;
      for (const [ratio, share, fade] of PARTIALS) {
        const oscillator = this.context.createOscillator(),
          gain = this.context.createGain();
        oscillator.type = "sine";
        oscillator.frequency.value = frequency * ratio;
        gain.gain.setValueAtTime(0, at);
        gain.gain.linearRampToValueAtTime(peak * share, at + 0.006);
        gain.gain.exponentialRampToValueAtTime(0.0001, at + length * fade);
        oscillator.connect(gain).connect(out);
        const item = { oscillator, gain };
        this.nodes.add(item);
        oscillator.onended = () => {
          oscillator.disconnect();
          gain.disconnect();
          this.nodes.delete(item);
        };
        oscillator.start(at);
        oscillator.stop(at + length + 0.05);
      }
    }
  }

  /// Silences whatever is sounding. The tail stays, so turning the cues back on needs nothing new.
  dispose() {
    for (const { oscillator } of this.nodes) {
      try {
        oscillator.stop();
      } catch {}
    }
    this.nodes.clear();
  }
}
