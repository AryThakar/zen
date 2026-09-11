// Chrome AudioWorklet. Stream every 20 ms, including silence, so engine VAD owns endpoints.
// Weighted decimation preserves duration across 44.1/48 kHz render quanta; the input
// graph supplies the anti-alias filter. No microphone audio is connected to the speakers.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.sum = 0;
    this.weight = 0;
    this.frame = new Int16Array(320);
    this.count = 0;
    this.noise = 0.004;
    this.hot = 0;
    this.quiet = 0;
    this.speaking = false;
    this.playback = false;
    this.frames = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "playback") this.playback = !!data.active;
      if (data.type === "flush") {
        if (this.count) this.emit();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  emit() {
    const buffer = new ArrayBuffer(this.count * 2);
    const view = new DataView(buffer);
    let power = 0,
      crossings = 0;
    for (let i = 0; i < this.count; i++) {
      const value = this.frame[i];
      view.setInt16(i * 2, value, true);
      power += (value / 32768) ** 2;
      if (i && value >= 0 !== this.frame[i - 1] >= 0) crossings++;
    }
    const rms = Math.sqrt(power / this.count);
    const zcr = crossings / this.count;
    // An onset hint, not speech recognition: a sustained, speech-band rise over
    // the learned noise floor can stop playback before the engine's neural VAD.
    const threshold = Math.max(
      this.playback ? 0.026 : 0.012,
      this.noise * (this.playback ? 4 : 2.7),
    );
    const likely = rms > threshold && zcr > 0.008 && zcr < 0.48;
    this.hot = likely ? this.hot + 1 : 0;
    this.quiet = rms < threshold * 0.65 ? this.quiet + 1 : 0;
    if (!this.speaking && this.hot >= (this.playback ? 5 : 3)) {
      this.speaking = true;
      this.port.postMessage({ type: "onset" });
    }
    if (this.speaking && this.quiet >= 12) {
      this.speaking = false;
      this.port.postMessage({ type: "offset" });
    }
    if (!this.speaking && !likely)
      this.noise += (Math.min(rms, 0.025) - this.noise) * 0.025;
    if (++this.frames % 3 === 0) this.port.postMessage({ type: "level", rms });
    this.port.postMessage(buffer, [buffer]);
    this.count = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const raw of channel) {
      const value = Number.isFinite(raw) ? raw : 0;
      let left = 1;
      while (left > 1e-7) {
        const take = Math.min(left, this.ratio - this.weight);
        this.sum += value * take;
        this.weight += take;
        left -= take;
        if (this.weight >= this.ratio - 1e-7) {
          this.frame[this.count++] = Math.max(
            -32768,
            Math.min(32767, Math.round((this.sum / this.ratio) * 32768)),
          );
          this.sum = 0;
          this.weight = 0;
          if (this.count === this.frame.length) this.emit();
        }
      }
    }
    return true;
  }
}
registerProcessor("zen-capture", Capture);
