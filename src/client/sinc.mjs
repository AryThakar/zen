/// A Kaiser-windowed sinc low-pass, tabulated at fractional positions between samples, for the
/// page's two rate converters: Zen's voice on its way to the speaker, and the microphone on its
/// way to the recogniser.
///
/// Row `p` of the table holds the kernel for an output instant `p / phases` of a sample past the
/// tap grid, so a converter reads the row for its position and interpolates between neighbouring
/// rows. `cutoff` is the passband edge as a fraction of the source Nyquist frequency. Every row
/// is scaled to unity gain at DC, so conversion cannot move the level.
export function kaiserSinc({ taps, phases, cutoff, beta }) {
  const half = taps / 2;
  // The zeroth-order modified Bessel function of the first kind, by its power series.
  const bessel = (x) => {
    let sum = 1,
      term = 1;
    for (let k = 1; k < 32; k++) {
      term *= (x / (2 * k)) ** 2;
      sum += term;
    }
    return sum;
  };
  const scale = bessel(beta);
  const table = new Float32Array((phases + 1) * taps);
  for (let phase = 0; phase <= phases; phase++) {
    const offset = phase / phases;
    let sum = 0;
    for (let tap = 0; tap < taps; tap++) {
      const x = tap - half + 1 - offset;
      const a = cutoff * x;
      const sinc = a === 0 ? 1 : Math.sin(Math.PI * a) / (Math.PI * a);
      const r = x / half;
      const window = Math.abs(r) >= 1 ? 0 : bessel(beta * Math.sqrt(1 - r * r)) / scale;
      const value = sinc * window;
      table[phase * taps + tap] = value;
      sum += value;
    }
    for (let tap = 0; tap < taps; tap++) table[phase * taps + tap] /= sum;
  }
  return table;
}
