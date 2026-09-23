// Chrome AudioWorklet. Stream every 20 ms, including silence, so engine VAD owns endpoints.
// No microphone audio is connected to the speakers.
//
// Whether anyone is speaking is decided by the engine's neural detector alone. This used to
// run an energy detector of its own as well, whose onset messages nothing acted on: an energy
// rise cannot tell a voice from a door, so it was never allowed to cut a reply.
import { kaiserSinc } from "./sinc.mjs";

/// The rate the engine's detector and recogniser work at.
const TARGET_RATE = 16000;

/// Conversion from the device's rate to 16 kHz, band-limited first.
///
/// Anything above 8 kHz in a 16 kHz stream folds back below it, into the band recognition
/// listens to. This used to average each group of source samples behind one gentle 7 kHz
/// filter, and measured at 48 kHz that let a 10 kHz tone through at 6 kHz only 12 dB down, and
/// 9 kHz at 7 kHz only 8 dB down - where fricatives, sibilants and a laptop fan's hiss all live.
///
/// The filter is designed rather than tuned, from three numbers: flat to 7 kHz, which keeps
/// everything the recogniser uses; 60 dB down from 9 kHz, so nothing that would fold below
/// 7 kHz comes back above a microphone's own noise; and a Kaiser window, whose length and shape
/// follow from those for whatever rate the device runs at (Kaiser's design formulas, as given in
/// Oppenheim and Schafer, Discrete-Time Signal Processing, section 7.5.3). At 48 kHz that is 88
/// taps, about 0.9 ms of delay.
export class Decimator {
  static PASS = 7000;
  static STOP = 9000;
  static ATTENUATION = 60;
  /// Fractional positions the kernel is tabulated at, as in the playback converter. Only a rate
  /// that is not a whole multiple of 16 kHz, such as 44.1 kHz, ever reads between them.
  static PHASES = 512;

  constructor(from, to = TARGET_RATE) {
    const { PASS, STOP, ATTENUATION, PHASES } = Decimator;
    const width = (2 * Math.PI * (STOP - PASS)) / from;
    // Even, so the window sits symmetrically either side of each output instant.
    this.taps = 2 * Math.ceil((ATTENUATION - 8) / (2.285 * width) / 2);
    this.table = kaiserSinc({
      taps: this.taps,
      phases: PHASES,
      // The middle of the transition, as a fraction of the source Nyquist frequency. A device
      // slower than 16 kHz has nothing above its own Nyquist to remove.
      cutoff: Math.min(1, (PASS + STOP) / from),
      beta: 0.1102 * (ATTENUATION - 8.7),
    });
    this.from = from;
    this.to = to;
    // The last `taps` samples, each written twice, so the window is always one unbroken run.
    this.ring = new Float32Array(this.taps * 2);
    this.write = 0;
    this.received = 0;
    this.produced = 0;
  }

  /// Takes one source sample.
  push(sample) {
    this.ring[this.write] = sample;
    this.ring[this.write + this.taps] = sample;
    this.write = (this.write + 1) % this.taps;
    this.received++;
  }

  /// The next output sample, once every source sample it depends on has arrived; null before.
  ///
  /// Output `k` is centred on source position `k * from / to`, kept as a ratio of integers so
  /// hours of capture do not drift. It is due the moment the sample `taps / 2` past its centre
  /// arrives, and then the window is exactly the last `taps` samples received.
  next() {
    const numerator = this.produced * this.from;
    const index = Math.floor(numerator / this.to);
    const half = this.taps / 2;
    if (this.received <= index + half) return null;
    const phase = ((numerator - index * this.to) / this.to) * Decimator.PHASES;
    const row = Math.min(Decimator.PHASES - 1, Math.floor(phase));
    const blend = phase - row;
    const near = row * this.taps;
    const far = near + this.taps;
    let sum = 0;
    for (let tap = 0; tap < this.taps; tap++) {
      const weight = this.table[near + tap] + (this.table[far + tap] - this.table[near + tap]) * blend;
      sum += this.ring[this.write + tap] * weight;
    }
    this.produced++;
    return sum;
  }
}

class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.decimator = new Decimator(sampleRate);
    this.frame = new Int16Array(320);
    this.count = 0;
    this.frames = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "flush") {
        // What the filter still holds is the end of the last word: push it through.
        for (let i = 0; i < this.decimator.taps / 2; i++) this.convert(0);
        if (this.count) this.emit();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  emit() {
    const buffer = new ArrayBuffer(this.count * 2);
    const view = new DataView(buffer);
    let power = 0;
    for (let i = 0; i < this.count; i++) {
      const value = this.frame[i];
      view.setInt16(i * 2, value, true);
      power += (value / 32768) ** 2;
    }
    // For the orb only.
    if (++this.frames % 3 === 0)
      this.port.postMessage({ type: "level", rms: Math.sqrt(power / this.count) });
    this.port.postMessage(buffer, [buffer]);
    this.count = 0;
  }

  /// One device-rate sample in; whatever 16 kHz samples it completes go into the frame.
  convert(sample) {
    this.decimator.push(sample);
    for (let value = this.decimator.next(); value !== null; value = this.decimator.next()) {
      this.frame[this.count++] = Math.max(-32768, Math.min(32767, Math.round(value * 32768)));
      if (this.count === this.frame.length) this.emit();
    }
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const raw of channel) this.convert(Number.isFinite(raw) ? raw : 0);
    return true;
  }
}
registerProcessor("zen-capture", Capture);
