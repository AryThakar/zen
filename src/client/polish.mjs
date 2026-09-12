/// Playback polish for synthesised speech.
///
/// The synthesiser produces a clean but dry 24 kHz voice: correct, and flat in the way a
/// recording made in a cupboard is flat. This is the small amount of processing a voice would
/// get before broadcast, arranged in the one order that matters - sibilance is tamed *before*
/// anything adds harmonics, because an exciter fed raw sibilance turns every "s" into a hiss.
///
/// Everything here is deliberately gentle. Processing you can hear working is worse than none.
export function createVoicePolish(context) {
  const input = context.createGain();
  const output = context.createGain();

  const peaking = (frequency, gain, q) => {
    const node = context.createBiquadFilter();
    node.type = "peaking";
    node.frequency.value = frequency;
    node.gain.value = gain;
    node.Q.value = q;
    return node;
  };
  const pass = (type, frequency, q = 0.7) => {
    const node = context.createBiquadFilter();
    node.type = type;
    node.frequency.value = frequency;
    node.Q.value = q;
    return node;
  };

  // Rumble the voice never uses, and which costs headroom on small speakers.
  const rumble = pass("highpass", 75);
  // Small cut where a close voice sounds boxy.
  const body = peaking(300, -1.5, 0.9);
  // Consonant definition: what makes speech intelligible on a laptop speaker.
  const presence = peaking(3000, 1.6, 0.8);
  // Sibilance sits at 6-9 kHz. A static dip rather than a dynamic de-esser: the synthesiser's
  // level is consistent, and an over-eager de-esser turns every "s" into a lisp.
  const sibilance = peaking(7000, -3.0, 1.6);

  // Air. The source is 24 kHz, so it holds nothing above 12 kHz at all; this generates
  // harmonics that were never in it, which is what "air" on a studio voice actually is.
  // Fed from above 5 kHz and kept above 11 kHz, which puts the new harmonics clear of the
  // sibilance band: measured at 9 kHz they landed inside it and made every "s" brighter, the
  // exact failure an exciter is known for.
  const exciterIn = pass("highpass", 5000);
  // Saturation only generates harmonics where the curve bends, and this band sits around
  // -40 dBFS, which is the straight part. Drive it hard into the bend, then mix it back in
  // quietly: that is the whole trick of an exciter.
  const drive = context.createGain();
  drive.gain.value = 12;
  const shaper = context.createWaveShaper();
  const curve = new Float32Array(1024);
  for (let i = 0; i < curve.length; i++) {
    const x = (i / (curve.length - 1)) * 2 - 1;
    curve[i] = Math.tanh(x * 2.2);
  }
  shaper.curve = curve;
  shaper.oversample = "4x";
  const exciterOut = pass("highpass", 11000);
  const air = context.createGain();
  air.gain.value = 0.05;

  // A room, barely. A dry voice sounds like it is inside your head; a hint of early reflection
  // puts it somewhere. Kept short and quiet so it cannot smear consonants or leak into the
  // microphone during a barge-in.
  const ambience = context.createConvolver();
  ambience.buffer = roomImpulse(context);
  const wet = context.createGain();
  wet.gain.value = 0.05;

  // Everything above adds level; this catches the peaks rather than letting them clip.
  const limiter = context.createDynamicsCompressor();
  limiter.threshold.value = -3;
  limiter.knee.value = 0;
  limiter.ratio.value = 12;
  limiter.attack.value = 0.003;
  limiter.release.value = 0.15;

  input.connect(rumble).connect(body).connect(presence).connect(sibilance);
  sibilance.connect(limiter);
  sibilance.connect(exciterIn).connect(drive).connect(shaper).connect(exciterOut).connect(air).connect(limiter);
  sibilance.connect(ambience).connect(wet).connect(limiter);
  limiter.connect(output);

  return { input, output, nodes: [rumble, body, presence, sibilance, exciterIn, drive, shaper, exciterOut, air, ambience, wet, limiter] };
}

/// A very short room: noise decaying over 120 ms, with two early reflections so it reads as a
/// small space rather than as a wash of reverb.
function roomImpulse(context, seconds = 0.12, decay = 5) {
  const rate = context.sampleRate;
  const length = Math.max(1, Math.floor(rate * seconds));
  const buffer = context.createBuffer(2, length, rate);
  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
    }
    // Slightly different reflection times per channel: width without a phasey centre.
    data[Math.floor(rate * (channel ? 0.013 : 0.011))] += 0.35;
    data[Math.floor(rate * (channel ? 0.021 : 0.019))] += 0.22;
  }
  return buffer;
}
