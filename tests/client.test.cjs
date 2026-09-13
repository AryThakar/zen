const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

function captureAt(rate) {
  const messages = [];
  let Capture;
  const scope = {
    sampleRate: rate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (data) => messages.push(data) };
      }
    },
    registerProcessor: (_, ctor) => {
      Capture = ctor;
    },
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "../src/client/capture.js"), "utf8"),
    scope,
  );
  const capture = new Capture();
  const push = (samples) => {
    for (let i = 0; i < samples.length; i += 128)
      capture.process([[samples.subarray(i, i + 128)]]);
  };
  return {
    capture,
    push,
    messages,
    frames: () => messages.filter((m) => m.byteLength !== undefined),
  };
}

test("20 ms capture preserves duration and PCM alignment at 44.1 and 48 kHz", () => {
  for (const rate of [44100, 48000]) {
    const capture = captureAt(rate);
    capture.push(new Float32Array(rate).fill(0.25));
    assert.equal(capture.frames().length, 50);
    for (const frame of capture.frames()) {
      assert.equal(frame.byteLength, 640);
      assert.equal(new DataView(frame).getInt16(0, true), 8192);
    }
  }
});

test("muting flushes the final short frame before its confirmation, and keeps silence", () => {
  const { capture, push, messages, frames } = captureAt(48000);
  push(new Float32Array(48000 + 480).fill(0));
  capture.port.onmessage({ data: { type: "flush" } });
  assert.equal(frames().length, 51);
  assert.equal(frames().at(-1).byteLength, 320);
  assert.equal(messages.at(-1).type, "flushed");
});

test("onset rejects DC, brief impulses and low noise, then detects sustained speech-band energy", () => {
  const { push, messages } = captureAt(48000);
  push(new Float32Array(24000).fill(0.04));
  push(new Float32Array(24000).fill(0));
  const tone = (n, amplitude) =>
    Float32Array.from(
      { length: n },
      (_, i) => Math.sin((i * 2 * Math.PI * 240) / 48000) * amplitude,
    );
  push(tone(24000, 0.003));
  push(tone(960, 0.15));
  push(new Float32Array(12000));
  assert(!messages.some((e) => e.type === "onset"));
  push(tone(12000, 0.1));
  assert.equal(messages.filter((e) => e.type === "onset").length, 1);
  push(new Float32Array(24000));
  assert.equal(messages.filter((e) => e.type === "offset").length, 1);
});

test("capture sanitizes non-finite hardware samples", () => {
  const { push, frames } = captureAt(48000);
  push(new Float32Array(960).fill(NaN));
  for (const value of new Int16Array(frames()[0])) assert.equal(value, 0);
});

function audioGraph() {
  const sent = [],
    sources = [],
    gains = [],
    phrases = [];
  const node = () => ({
    connect() {
      return this;
    },
    disconnect() {},
  });
  const audio = {
    currentTime: 0,
    deviceTime: 0,
    state: "running",
    destination: {},
    createAnalyser: () => ({
      ...node(),
      getFloatTimeDomainData(data) {
        data.fill(0);
      },
    }),
    createGain: () => {
      // Ramps are recorded: a phrase ending on silence is a scheduled ramp to zero, and
      // nothing else in the graph shows whether it was scheduled.
      const ramps = [];
      const gainNode = {
        ...node(),
        gain: {
          ramps,
          cancelScheduledValues() {},
          setTargetAtTime() {},
          setValueAtTime(value, at) {
            ramps.push({ value, at, kind: "set" });
          },
          setValueCurveAtTime(points, at, duration) {
            ramps.push({
              value: points[points.length - 1],
              at,
              duration,
              kind: "curve",
              // Equal power means the squares sum to one across the pair, not the values.
              power: points[Math.floor(points.length / 2)] ** 2,
            });
          },
          linearRampToValueAtTime(value, at) {
            ramps.push({ value, at, kind: "ramp" });
          },
        },
      };
      gains.push(gainNode);
      return gainNode;
    },
    getOutputTimestamp() {
      return { contextTime: this.deviceTime };
    },
    createBuffer: (_, n) => ({
      duration: n / 24000,
      getChannelData: () => new Float32Array(n),
    }),
    createBufferSource() {
      const source = {
        ...node(),
        start(at) { this.at = at; },
        stop() {
          this.onended?.();
        },
      };
      sources.push(source);
      return source;
    },
  };
  return { audio, sent, sources, gains, phrases };
}

async function playback() {
  const { AudioPlayback } = await import("../src/client/audio.mjs");
  const graph = audioGraph();
  const player = new AudioPlayback(
    graph.audio,
    (e) => graph.sent.push(e),
    (...args) => graph.phrases.push(args),
  );
  player.reset(3);
  return { ...graph, player };
}
function queue(player, phrase, sequence) {
  player.begin({ generation: 3, phrase, text: `Phrase ${phrase}.` });
  player.queue({
    generation: 3,
    phrase,
    sequence,
    pcm: new Uint8Array(4800),
  });
  player.end({ generation: 3, phrase, text: `Phrase ${phrase}.`, pause_ms: 0 });
}

test("playback credits device output, and a phrase never waits for the next phrase", async () => {
  const { player, audio, sent, sources, phrases } = await playback();
  try {
    queue(player, 1, 1);
    queue(player, 2, 2);
    audio.currentTime = 0.3;
    audio.deviceTime = 0.1;
    sources[0].onended();
    player.tick();
    assert(!sent.some((e) => e.type === "audio_played"));
    audio.deviceTime = 0.2;
    player.tick();
    assert.deepEqual(
      sent.filter((e) => e.type === "audio_played").map((e) => e.sequence),
      [1],
    );
    assert.deepEqual(
      sent.filter((e) => e.type === "played").map((e) => e.phrase),
      [1],
    );
    assert.equal(phrases.filter((e) => e[0] === "end").length, 1);
  } finally {
    player.dispose();
  }
});

test("a chunk arriving before playback runs out stays contiguous even with little headroom", async () => {
  for (const headroom of [0.009, 0.005, 0.001]) {
    const { player, audio, sources } = await playback();
    try {
      player.begin({ generation: 3, phrase: 1, text: "A continuous sentence." });
      player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(12000) });
      const previousEnd = sources[0].at + sources[0].buffer.duration;
      audio.currentTime = previousEnd - headroom;
      player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: new Uint8Array(12000) });
      assert.equal(sources[1].at, previousEnd,
        "an on-time chunk must neither insert silence nor overlap what came before");
    } finally { player.dispose(); }
  }
});

test("a delayed phrase-end message does not restart a pause already heard", async () => {
  const { player, audio, sent, sources } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Done." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(4800) });
    audio.currentTime = audio.deviceTime = 0.6;
    sources[0].onended();
    player.end({ generation: 3, phrase: 1, pause_ms: 100 });
    player.tick();
    assert(sent.some(event => event.type === "played" && event.phrase === 1),
      "the pause belongs after the audio, not after IPC delivery");
  } finally { player.dispose(); }
});

test("the measured short first codec block tolerates a 20 ms delivery delay", async () => {
  const { player, audio, sources } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Hello. My name is Zen." });
    // The installed codec emits 80 ms first, then 160 ms about 125 ms later.
    // Add one capture-frame interval of scheduling delay to that measured arrival.
    for (const [index, [arrived, samples]] of [[0, 1920], [0.145, 3840], [0.227, 6000]].entries()) {
      audio.currentTime = arrived;
      player.queue({ generation: 3, phrase: 1, sequence: index + 1, pcm: new Uint8Array(samples * 2) });
    }
    for (let i = 1; i < sources.length; i++)
      assert.equal(sources[i].at, sources[i - 1].at + sources[i - 1].buffer.duration,
        "short initial codec chunks need enough startup headroom for IPC jitter");
  } finally { player.dispose(); }
});

test("codec blocks of one phrase are joined at the sample, never overlapped", async () => {
  // A crossfade used to sit here to cover codec seams. A few seams really are discontinuous,
  // but only three of twenty-five in a rendered passage, and overlapping all of them to cover
  // those three costs six milliseconds of speech per join and smears every one. Rendered both
  // ways through the same audio graph, joining is indistinguishable from the same audio played
  // as a single buffer and any crossfade is not.
  const { player, audio, sources } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "A smooth sentence." });
    let played = 0;
    for (let sequence = 1; sequence <= 6; sequence++) {
      player.queue({ generation: 3, phrase: 1, sequence, pcm: new Uint8Array(12000) });
      audio.currentTime = 0.15 * sequence;
      played += sources[sequence - 1].buffer.duration;
    }
    for (let i = 1; i < sources.length; i++)
      assert.equal(
        sources[i].at,
        sources[i - 1].at + sources[i - 1].buffer.duration,
        "block " + i + " must start exactly where the one before it ends",
      );
    const last = sources[sources.length - 1];
    assert.equal(
      Number((last.at + last.buffer.duration - sources[0].at).toFixed(6)),
      Number(played.toFixed(6)),
      "the phrase must occupy exactly as long as the audio in it, losing nothing to overlaps",
    );
  } finally { player.dispose(); }
});

test("a joined block plays at full gain, with no ramp to smear it", async () => {
  const { player, audio, gains } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "A smooth sentence." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(12000) });
    audio.currentTime = 0.15;
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: new Uint8Array(12000) });
    const arriving = gains.at(-1).gain.ramps;
    assert.deepEqual(
      arriving.map((r) => r.kind),
      ["set"],
      "a contiguous block needs one gain set and nothing else: " + JSON.stringify(arriving),
    );
    assert.equal(arriving[0].value, 1, "and it plays at full gain from its first sample");
  } finally { player.dispose(); }
});

test("the microphone is told Zen can be heard even with no turn in progress", async () => {
  // The opening greeting is dispatched straight to the synthesiser, so the session phase never
  // reaches speaking. Without this the capture worklet keeps its quiet-room onset threshold
  // while Zen is talking, and Zen barges in on his own greeting.
  const graph = audioGraph();
  const { AudioPlayback } = await import("../src/client/audio.mjs");
  const sounding = [];
  const player = new AudioPlayback(
    graph.audio,
    (e) => graph.sent.push(e),
    () => {},
    (active) => sounding.push(active),
  );
  player.reset(3);
  try {
    player.begin({ generation: 3, phrase: 1, text: "Good evening." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(12000) });
    assert.equal(sounding.at(-1), true, "scheduling audio must raise the threshold at once");
    graph.audio.currentTime = graph.audio.deviceTime = 30;
    player.tick();
    assert.equal(sounding.at(-1), false, "and drop it once the audio has been heard");
  } finally { player.dispose(); }
});

test("a phrase ends on silence rather than on whatever amplitude synthesis stopped at", async () => {
  // Synthesis stops when it runs out of text, sometimes with the waveform still at a fifth of
  // full scale. Played as-is that edge is heard as the last sound being clipped off.
  const { player, audio, gains } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Cut off mid" });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(12000) });
    audio.currentTime = 0.01;
    player.end({ generation: 3, phrase: 1, pause_ms: 0 });
    const ramps = gains[gains.length - 1].gain.ramps;
    assert(
      ramps.some((ramp) => ramp.kind === "ramp" && ramp.value === 0),
      "the final block must be ramped down to silence, got " + JSON.stringify(ramps),
    );
  } finally { player.dispose(); }
});

test("a stalled onended cannot hold up the whole acknowledgement ledger", async () => {
  // The ledger drives the engine: no acknowledgements, no more audio, and the buffer runs
  // dry into an audible gap. A main thread busy enough to delay one `onended` callback must
  // not be able to cause that, because the device clock already proves the block was played.
  const { player, audio, sent, sources } = await playback();
  try {
    queue(player, 1, 1);
    queue(player, 2, 2);
    audio.currentTime = audio.deviceTime = 0.05;
    player.tick();
    assert.equal(sent.length, 0, "nothing is credited before the device reaches it");
    // The device has long since played both blocks, but the callbacks never arrived.
    audio.currentTime = audio.deviceTime = 30;
    player.tick();
    const credited = sent.filter((event) => event.type === "audio_played");
    assert.deepEqual(
      credited.map((event) => event.sequence),
      [1, 2],
      "both blocks must be credited from the device clock alone",
    );
    assert.ok(sources.length >= 2);
  } finally {
    player.dispose();
  }
});

test("suspended audio and cancelled sources never credit unheard speech", async () => {
  const { player, audio, sent, sources } = await playback();
  try {
    queue(player, 1, 1);
    audio.currentTime = audio.deviceTime = 1;
    audio.state = "suspended";
    sources[0].onended();
    player.tick();
    assert.equal(sent.length, 0);
    player.clear();
    audio.state = "running";
    player.tick();
    assert.equal(sent.length, 0);
    queue(player, 2, 2);
    assert.equal(
      sources.length,
      1,
      "late output stays suppressed before the server cancels its generation",
    );
  } finally {
    player.dispose();
  }
});

test("new generations discard old playback and stale phrase events", async () => {
  const { player, audio, sent } = await playback();
  try {
    queue(player, 1, 1);
    player.reset(4);
    queue(player, 2, 2);
    audio.currentTime = audio.deviceTime = 5;
    player.tick();
    assert.equal(sent.length, 0);
    assert.equal(player.phrases.size, 0);
  } finally {
    player.dispose();
  }
});

test("malformed audio and excessive queued playback fail explicitly", async () => {
  const { player } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "hello" });
    // An odd byte count cannot be whole 16-bit samples.
    assert.throws(
      () =>
        player.queue({
          generation: 3,
          phrase: 1,
          sequence: 1,
          pcm: new Uint8Array(1),
        }),
      /buffer/,
    );
    // Audio for a phrase that was never announced has nothing to attach to.
    assert.throws(
      () =>
        player.queue({
          generation: 3,
          phrase: 2,
          sequence: 1,
          pcm: new Uint8Array(2),
        }),
      /Invalid/,
    );
    // Base64 was the old wire format; a string is no longer audio.
    assert.throws(
      () =>
        player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: "AAAA" }),
      /Invalid/,
    );
    player.next = 9;
    assert.throws(
      () =>
        player.queue({
          generation: 3,
          phrase: 1,
          sequence: 1,
          pcm: new Uint8Array(2),
        }),
      /buffer/,
    );
  } finally {
    player.dispose();
  }
});
