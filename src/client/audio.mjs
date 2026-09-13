// Audio and device identifiers are held only for the lifetime of a connection.
export class AudioPlayback {
  // The codec's first block can be only 80 ms, with the next arriving 125 ms later.
  // Cover that initial deficit plus a few IPC/render ticks before starting the device.
  static STARTUP_BUFFER = 0.1;
  /// The rate synthesis produces.
  static RATE = 24000;
  /// Ramped off the end of a phrase. Synthesis stops when it runs out of text, sometimes with
  /// the waveform still at a fifth of full scale, which is heard as the last sound being cut.
  static PHRASE_FADE = 0.006;

  constructor(context, send, onPhrase = () => {}, onSounding = () => {}) {
    this.context = context;
    this.send = send;
    this.onPhrase = onPhrase;
    // Raised while Zen's voice is actually leaving the speakers, which is not the same as the
    // session being in a phase that implies it: the opening greeting is dispatched straight to
    // the synthesiser without the turn machine, so the phase stays idle throughout it.
    this.onSounding = onSounding;
    this.sounding = false;
    this.analyser = context.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.connect(context.destination);
    this.meter = new Float32Array(256);
    this.sources = new Set();
    this.phrases = new Map();
    this.markers = [];
    this.generation = 0;
    this.epoch = 0;
    // Set here as well as in `reset`, so the ordering check below is a comparison against a
    // number from the very first block rather than against `undefined`, which nothing is
    // less than or equal to.
    this.lastSequence = 0;
    this.next = 0;
    this.lastItem = null;
    this.lastPhrase = null;
    this.accepting = true;
    this.jitter = AudioPlayback.STARTUP_BUFFER;
    this.timer = setInterval(() => this.tick(), 20);
  }

  reset(generation) {
    this.clear();
    this.generation = generation;
    this.lastSequence = 0;
    this.accepting = true;
  }

  clear() {
    this.epoch++;
    this.accepting = false;
    if (this.sounding) {
      this.sounding = false;
      this.onSounding?.(false);
    }
    this.markers = [];
    this.phrases.clear();
    const now = this.context.currentTime;
    for (const { source, gain } of this.sources) {
      source.onended = () => {
        source.disconnect();
        gain.disconnect();
      };
      gain.gain.cancelScheduledValues(now);
      gain.gain.setTargetAtTime(0, now, 0.002);
      try {
        source.stop(now + 0.01);
      } catch {
        source.disconnect();
        gain.disconnect();
      }
    }
    this.sources.clear();
    this.next = now;
    this.lastItem = null;
    this.lastPhrase = null;
  }

  begin(event) {
    if (!this.accepting || event.generation !== this.generation) return;
    if (!Number.isSafeInteger(event.phrase) || event.phrase < 1 || this.phrases.size >= 128 || this.phrases.has(event.phrase))
      throw new Error("Invalid voice phrase.");
    this.phrases.set(event.phrase, { text: event.text || "", started: false });
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
    if (
      !bytes.length ||
      bytes.length % 2 ||
      this.next - this.context.currentTime > 8 ||
      this.markers.length > 1024
    )
      throw new Error("Voice playback exceeded its buffer.");
    const buffer = this.context.createBuffer(1, bytes.length / 2, AudioPlayback.RATE);
    const data = buffer.getChannelData(0),
      view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < data.length; i++)
      data[i] = view.getInt16(i * 2, true) / 32768;
    const source = this.context.createBufferSource(),
      gain = this.context.createGain();
    source.buffer = buffer;
    source.connect(gain).connect(this.analyser);
    const now = this.context.currentTime;
    // Buffer only at startup or after a real underrun. Rebuffering while even a few
    // milliseconds remain inserts silence into an otherwise contiguous waveform.
    if (phrase.started && this.next < now)
      this.jitter = Math.min(0.16, this.jitter + 0.015);
    // Blocks of one phrase meet at the sample, so they are joined and not crossfaded.
    //
    // A crossfade was here to hide codec seams, and a few are genuinely discontinuous - the
    // worst measured six times the sample-to-sample movement either side of it. But only three
    // of twenty-five seams in a rendered passage were like that, and overlapping all
    // twenty-five to cover those three costs six milliseconds of speech at every join and
    // smears each one. Rendered both ways through the same audio graph and compared against
    // the identical audio played as a single buffer, joining is indistinguishable from the
    // single buffer and any crossfade is not.
    const contiguous =
      phrase.started && this.lastPhrase === event.phrase && this.next > now;
    const at = contiguous
      ? this.next
      : Math.max(now + (this.next > now ? 0 : this.jitter), this.next);
    const end = at + buffer.duration;
    this.next = end;
    if (contiguous) {
      gain.gain.setValueAtTime(1, at);
    } else {
      if (!phrase.started) {
        phrase.started = true;
        this.markers.push({
          at,
          type: "playback_started",
          phrase: event.phrase,
          text: phrase.text,
          epoch: this.epoch,
        });
      }
      gain.gain.setValueAtTime(0, at);
      gain.gain.linearRampToValueAtTime(
        1,
        at + Math.min(0.004, buffer.duration / 2),
      );
    }
    const item = { source, gain, at, end, ended: false };
    const epoch = this.epoch;
    this.sources.add(item);
    this.lastItem = item;
    this.lastPhrase = event.phrase;
    source.onended = () => {
      item.ended = true;
      this.sources.delete(item);
      source.disconnect();
      gain.disconnect();
      if (epoch === this.epoch) this.tick();
    };
    this.markers.push({
      at: end,
      type: "audio_played",
      sequence: event.sequence,
      item,
      epoch,
    });
    source.start(at);
    this.lastSequence = event.sequence;
    this.markers.sort((a, b) => a.at - b.at);
    this.report();
  }

  end(event) {
    if (!this.accepting || event.generation !== this.generation) return;
    const phrase = this.phrases.get(event.phrase);
    if (!phrase?.started || phrase.ended) throw new Error("Invalid or duplicate voice phrase ending.");
    phrase.ended = true;
    // End on silence. Without this the phrase stops on an edge and the last sound is heard
    // clipped - measured at up to a fifth of full scale on the final sample.
    const item = this.lastItem;
    if (item && !item.ended && this.next > this.context.currentTime) {
      const from = Math.max(item.at, this.next - AudioPlayback.PHRASE_FADE);
      if (from > this.context.currentTime) {
        item.gain.gain.cancelScheduledValues(from);
        item.gain.gain.setValueAtTime(1, from);
        item.gain.gain.linearRampToValueAtTime(0, this.next);
      }
    }
    // The pause starts when audio ends, even if this message reaches the page later.
    this.next =
      this.next +
      Math.max(0, Math.min(500, event.pause_ms || 0)) / 1000;
    this.markers.push({
      at: this.next,
      type: "played",
      phrase: event.phrase,
      text: event.text || phrase.text,
      epoch: this.epoch,
    });
    this.markers.sort((a, b) => a.at - b.at);
  }

  outputTime() {
    // Chrome exposes the device's playout position. Rendering a buffer is not proof
    // it was heard; suspended contexts and Bluetooth output must not earn early credit.
    const stamp = this.context.getOutputTimestamp?.();
    if (stamp?.contextTime > 0)
      return Math.min(this.context.currentTime, stamp.contextTime);
    return Math.max(
      0,
      this.context.currentTime - (this.context.outputLatency || 0.04),
    );
  }

  tick() {
    if (this.context.state !== "running") return;
    const now = this.outputTime();
    while (this.markers.length && this.markers[0].at <= now) {
      const marker = this.markers[0];
      // Normally a block is credited once the device clock has passed it *and* its source
      // reported it ended. `onended` is a main thread callback, though, and the main thread
      // also renders the orb: a stall there would otherwise hold up the whole ledger behind
      // one marker, stop the acknowledgements, and starve the engine into a real gap in
      // playback. Past a quarter second the device clock is proof enough on its own.
      if (
        marker.type === "audio_played" &&
        !marker.item.ended &&
        now - marker.at < 0.25
      )
        break;
      this.markers.shift();
      if (marker.epoch !== this.epoch) continue;
      const event = { type: marker.type, generation: this.generation };
      if (marker.sequence !== undefined) event.sequence = marker.sequence;
      if (marker.phrase !== undefined) event.phrase = marker.phrase;
      if (this.send(event) === false) { this.clear(); return; }
      if (marker.type === "playback_started")
        this.onPhrase("start", marker.text, marker.phrase, this.generation);
      if (marker.type === "played") {
        this.phrases.delete(marker.phrase);
        this.jitter = Math.max(AudioPlayback.STARTUP_BUFFER, this.jitter * 0.98);
        this.onPhrase("end", marker.text, marker.phrase, this.generation);
      }
    }
    this.report();
  }

  /// Tell the caller whether audio is on its way out of the speakers right now.
  report() {
    const sounding = this.accepting && this.next > this.outputTime();
    if (sounding === this.sounding) return;
    this.sounding = sounding;
    this.onSounding(sounding);
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
    this.clear();
    this.analyser.disconnect();
  }
}

export class Microphone {
  constructor(context, { onFrame, onLevel, onOnset = () => {}, onState, onError }) {
    this.context = context;
    this.callbacks = { onFrame, onLevel, onOnset, onState, onError };
    this.revision = 0;
    this.active = false;
    this.pending = false;
    this.loaded = null;
    this.nodes = [];
    this.playback = false;
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
        .addModule("/capture.js")
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
      const source = this.context.createMediaStreamSource(stream);
      this.nodes.push(source);
      const high = this.context.createBiquadFilter();
      this.nodes.push(high);
      high.type = "highpass";
      high.frequency.value = 70;
      high.Q.value = 0.707;
      const low = this.context.createBiquadFilter();
      this.nodes.push(low);
      low.type = "lowpass";
      low.frequency.value = 7000;
      low.Q.value = 0.707;
      const capture = new AudioWorkletNode(this.context, "zen-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.nodes = [source, high, low, capture];
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
        else if (data.type === "onset") this.callbacks.onOnset();
        else if (data.type === "flushed") this.flushed?.();
      };
      capture.onprocessorerror = () =>
        this.fail(
          "Microphone processing stopped. Start talking again to reconnect it.",
        );
      source
        .connect(high)
        .connect(low)
        .connect(capture)
        .connect(this.context.destination);
      this.active = true;
      this.setPlayback(this.playback);
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

  setPlayback(active) {
    this.playback = active;
    this.capture?.port.postMessage({ type: "playback", active });
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

export class Soundscape {
  constructor(context) {
    this.context = context;
    this.enabled = true;
    this.nodes = new Set();
  }
  play(kind) {
    if (!this.enabled || this.context.state !== "running") return;
    const notes = {
      connect: [523.25, 783.99],
      ready: [659.25, 880],
      mic: [587.33],
      mute: [392],
      stop: [440],
      error: [349.23, 329.63],
    }[kind] || [523.25];
    notes.forEach((frequency, index) => {
      const at = this.context.currentTime + index * 0.085;
      const oscillator = this.context.createOscillator(),
        gain = this.context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(frequency, at);
      oscillator.frequency.exponentialRampToValueAtTime(
        frequency * 0.985,
        at + 0.28,
      );
      gain.gain.setValueAtTime(0, at);
      gain.gain.linearRampToValueAtTime(0.027, at + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.3);
      oscillator.connect(gain).connect(this.context.destination);
      const item = { oscillator, gain };
      this.nodes.add(item);
      oscillator.onended = () => {
        oscillator.disconnect();
        gain.disconnect();
        this.nodes.delete(item);
      };
      oscillator.start(at);
      oscillator.stop(at + 0.32);
    });
  }
  dispose() {
    for (const { oscillator } of this.nodes) {
      try {
        oscillator.stop();
      } catch {}
    }
    this.nodes.clear();
  }
}
