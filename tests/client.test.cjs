const test = require("node:test");
const fsp = require("node:fs/promises");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

/// The real capture worklet, loaded as the module it is, with the globals an AudioWorklet
/// scope provides. The module registers its processor once; each instance reads the rate it
/// runs at, and posts into the messages of the test that made it.
let Capture;
async function captureAt(rate) {
  if (!Capture) {
    globalThis.AudioWorkletProcessor = class {
      constructor() {
        const sink = globalThis.captureMessages;
        this.port = { postMessage: (data) => sink.push(data) };
      }
    };
    globalThis.registerProcessor = (_, processor) => {
      Capture = processor;
    };
    await import("../src/client/capture.mjs");
  }
  const messages = [];
  globalThis.sampleRate = rate;
  globalThis.captureMessages = messages;
  const capture = new Capture();
  const push = (samples) => {
    for (let i = 0; i < samples.length; i += 128)
      capture.process([[samples.subarray(i, i + 128)]]);
  };
  const flush = () => capture.port.onmessage({ data: { type: "flush" } });
  return {
    capture,
    push,
    flush,
    messages,
    frames: () => messages.filter((m) => m.byteLength !== undefined),
  };
}

test("a second of capture is a second at 16 kHz, in 20 ms frames, at 44.1 and 48 kHz", async () => {
  for (const rate of [44100, 48000]) {
    const capture = await captureAt(rate);
    capture.push(new Float32Array(rate).fill(0.25));
    // The band-limiting filter holds back under a millisecond, which ending capture releases.
    capture.flush();
    const frames = capture.frames();
    assert.equal(frames.length, 50);
    for (const frame of frames) assert.equal(frame.byteLength, 640);
    // Once the filter has the signal on both sides of it, a level goes through unchanged.
    const samples = frames.flatMap((frame) => Array.from(new Int16Array(frame)));
    const settled = samples.slice(100, -100);
    assert.ok(settled.every((value) => value === 8192), "a steady level must pass at unity");
  }
});

test("muting flushes the final short frame before its confirmation, and keeps silence", async () => {
  const { push, flush, messages, frames } = await captureAt(48000);
  push(new Float32Array(48000 + 480).fill(0));
  flush();
  assert.equal(frames().length, 51);
  assert.equal(frames().at(-1).byteLength, 320);
  assert.equal(messages.at(-1).type, "flushed");
});

test("capture decides nothing about speech; that is the engine's detector", async () => {
  const { push, messages } = await captureAt(48000);
  const tone = Float32Array.from(
    { length: 48000 },
    (_, i) => Math.sin((i * 2 * Math.PI * 240) / 48000) * 0.3,
  );
  push(tone);
  push(new Float32Array(48000));
  const kinds = new Set(messages.filter((e) => e.byteLength === undefined).map((e) => e.type));
  assert.deepEqual([...kinds], ["level"]);
});

test("capture sanitizes non-finite hardware samples", async () => {
  const { push, frames } = await captureAt(48000);
  push(new Float32Array(960).fill(NaN));
  for (const value of new Int16Array(frames()[0])) assert.equal(value, 0);
});

/// Stands in for the renderer running on the audio thread. It records what the page pushed,
/// and lets a test say how much of it has been played rather than waiting for a real device.
class FakeRenderer {
  constructor() {
    this.pushed = [];
    this.target = 0;
    this.clears = 0;
    this.epoch = 0;
    this.consumed = 0;
    this.underruns = 0;
    this.port = {
      onmessage: null,
      postMessage: (message) => {
        if (message.type === "push") this.pushed.push(new Float32Array(message.pcm));
        else if (message.type === "target") this.target = message.frames;
        else if (message.type === "clear") {
          this.clears++;
          this.pushed = [];
          this.epoch = message.epoch;
          this.consumed = 0;
          this.report();
        }
      },
    };
  }
  connect() { return this; }
  disconnect() {}
  /// Every frame the page has handed over, in order.
  stream() {
    const total = this.pushed.reduce((n, b) => n + b.length, 0);
    const all = new Float32Array(total);
    let at = 0;
    for (const block of this.pushed) { all.set(block, at); at += block.length; }
    return all;
  }
  /// Say that playback has reached `frames` into the stream.
  play(frames) {
    this.consumed = frames;
    this.report();
  }
  report() {
    this.port.onmessage?.({
      data: {
        type: "progress",
        epoch: this.epoch,
        consumed: this.consumed,
        queued: Math.max(0, this.stream().length - this.consumed),
        underruns: this.underruns,
        filling: false,
      },
    });
  }
}

function audioGraph(rate = 24000) {
  const sent = [], phrases = [], renderers = [];
  const node = () => ({ connect() { return this; }, disconnect() {} });
  globalThis.AudioWorkletNode = class {
    constructor() {
      const renderer = new FakeRenderer();
      renderers.push(renderer);
      return renderer;
    }
  };
  const audio = {
    sampleRate: rate,
    currentTime: 0,
    deviceTime: 0,
    outputLatency: 0,
    state: "running",
    destination: {},
    audioWorklet: { addModule: async () => {} },
    createAnalyser: () => ({
      ...node(),
      getFloatTimeDomainData(data) { data.fill(0); },
    }),
    createGain: () => ({ ...node(), gain: { value: 1, setTargetAtTime(value) { this.value = value; } } }),
    getOutputTimestamp() { return { contextTime: this.deviceTime }; },
  };
  return { audio, sent, phrases, renderers };
}

async function playback(rate = 24000) {
  const { AudioPlayback } = await import("../src/client/audio.mjs");
  const graph = audioGraph(rate);
  const player = new AudioPlayback(
    graph.audio,
    (e) => graph.sent.push(e),
    (...args) => graph.phrases.push(args),
  );
  player.reset(3);
  // Most tests are about the stream itself, so its start is not held.
  player.lags = [0];
  // The renderer is created once its module resolves.
  await player.loading;
  return { ...graph, player, renderer: graph.renderers[0] };
}

/// A block of speech that is obviously not silence, so a test can see where it landed.
function speech(frames, from = 1) {
  const pcm = new Uint8Array(frames * 2);
  const view = new DataView(pcm.buffer);
  for (let i = 0; i < frames; i++)
    view.setInt16(i * 2, Math.round(8000 * Math.sin((i + from) / 6)), true);
  return pcm;
}

function queue(player, phrase, sequence) {
  player.begin({ generation: 3, phrase, text: `Phrase ${phrase}.` });
  player.queue({ generation: 3, phrase, sequence, pcm: new Uint8Array(4800) });
  player.end({ generation: 3, phrase, text: `Phrase ${phrase}.`, pause_ms: 0 });
}

test("a phrase reaches the renderer as one stream, whatever the blocks arrived like", async () => {
  // The property the whole playback path now rests on. There are no start times to get right
  // and no seams to cover: the renderer plays what it is given, in order, and what it is given
  // is exactly the samples of the phrase.
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "One sentence." });
    const blocks = [1200, 6000, 300, 6000];
    const expected = [];
    for (const [index, frames] of blocks.entries()) {
      const pcm = speech(frames, index * 977);
      const view = new DataView(pcm.buffer);
      for (let i = 0; i < frames; i++) expected.push(view.getInt16(i * 2, true) / 32768);
      player.queue({ generation: 3, phrase: 1, sequence: index + 1, pcm });
    }
    player.end({ generation: 3, phrase: 1, text: "One sentence.", pause_ms: 0 });
    const stream = renderer.stream();
    assert.equal(stream.length, expected.length,
      "the stream must be exactly as long as the audio put into it");
    let worst = 0;
    for (let i = 0; i < expected.length; i++)
      worst = Math.max(worst, Math.abs(stream[i] - expected[i]));
    // Only the ending fade may differ, and only over its own few milliseconds.
    const fadeFrames = Math.round(0.006 * 24000);
    let body = 0;
    for (let i = 0; i < expected.length - fadeFrames; i++)
      body = Math.max(body, Math.abs(stream[i] - expected[i]));
    assert(body < 1e-4, "the body of the phrase must arrive unchanged, worst " + body);
    assert(worst > 0, "and the very end must be faded rather than left on an edge");
  } finally { player.dispose(); }
});

test("the start of a stream is held for as long as delivery has been running late", async () => {
  // Synthesis delivers a reply's opening in lumps, the fourth well behind the first three.
  // Starting on the first lumps ran dry just before it arrived: a gap in a word.
  const { player, renderer } = await playback();
  try {
    player.lags = [60, 20];
    assert.equal(player.playoutDelay(), 100, "the worst seen, plus how much it varied");
    player.begin({ generation: 3, phrase: 1, text: "Held." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(1920) });
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: speech(3840, 1920) });
    assert.equal(renderer.stream().length, 0, "nothing plays before the delay");
    await new Promise((resolve) => setTimeout(resolve, 150));
    assert(renderer.stream().length > 0, "and everything waiting goes out once it has passed");
    // The rest of the stream is not held again.
    const before = renderer.stream().length;
    player.queue({ generation: 3, phrase: 1, sequence: 3, pcm: speech(6000, 5760) });
    assert(renderer.stream().length > before);
  } finally { player.dispose(); }
});

test("audio arriving while earlier audio is still playing is never held", async () => {
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "First." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.end({ generation: 3, phrase: 1, text: "First.", pause_ms: 0 });
    player.lags = [500];
    player.begin({ generation: 3, phrase: 2, text: "Second." });
    const before = renderer.stream().length;
    player.queue({ generation: 3, phrase: 2, sequence: 2, pcm: speech(6000) });
    assert(renderer.stream().length > before, "a stream already playing continues at once");
  } finally { player.dispose(); }
});

test("how late delivery ran is learned from a stream that played out", async () => {
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Late." });
    // 80 ms of audio, then the next only after 200 ms: delivery fell about 120 ms behind.
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(1920) });
    await new Promise((resolve) => setTimeout(resolve, 200));
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: speech(1920, 1920) });
    player.end({ generation: 3, phrase: 1, text: "Late.", pause_ms: 0 });
    player.tick();
    assert.deepEqual(player.lags, [0], "not while it is still to be heard");
    renderer.play(renderer.stream().length);
    player.tick();
    const learned = player.lags.at(-1);
    assert(learned >= 115 && learned < 200, "learned " + learned);
    assert(player.playoutDelay() >= learned);
  } finally { player.dispose(); }
});

test("a stream cut short teaches the playout delay nothing", async () => {
  const { player } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Cut." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(1920) });
    await new Promise((resolve) => setTimeout(resolve, 100));
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: speech(1920, 1920) });
    player.reset(4);
    player.tick();
    assert.deepEqual(player.lags, [0]);
  } finally { player.dispose(); }
});

test("when a block arrives changes nothing about the stream", async () => {
  // Delivery used to decide where audio was placed, so a late block became a hole in a word.
  // Nothing here reads a clock, so a block delivered a second late is the same stream.
  const { player, renderer, audio } = await playback();
  const render = async (delays) => {
    const graph = await playback();
    graph.player.begin({ generation: 3, phrase: 1, text: "Timing." });
    for (const [index, delay] of delays.entries()) {
      graph.audio.currentTime = delay;
      graph.audio.deviceTime = delay;
      graph.player.queue({ generation: 3, phrase: 1, sequence: index + 1, pcm: speech(6000, index * 31) });
    }
    graph.player.end({ generation: 3, phrase: 1, text: "Timing.", pause_ms: 0 });
    const out = graph.renderer.stream();
    graph.player.dispose();
    return out;
  };
  try {
    const prompt = await render([0, 0.25, 0.5]);
    const late = await render([0, 5, 30]);
    assert.equal(prompt.length, late.length, "the stream must be the same length either way");
    let worst = 0;
    for (let i = 0; i < prompt.length; i++) worst = Math.max(worst, Math.abs(prompt[i] - late[i]));
    assert.equal(worst, 0, "and identical sample for sample, worst difference " + worst);
  } finally { player.dispose(); void renderer; void audio; }
});

test("a phrase ends on silence rather than on whatever amplitude synthesis stopped at", async () => {
  // Synthesis stops when it runs out of text, sometimes with the waveform still well away from
  // zero. The tail is held back precisely so that there is always something left to fade.
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Done." });
    // A constant, so anything other than a fade is obvious.
    const pcm = new Uint8Array(12000);
    new DataView(pcm.buffer).setInt16(0, 16000, true);
    for (let i = 0; i < 6000; i++) new DataView(pcm.buffer).setInt16(i * 2, 16000, true);
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm });
    const beforeEnd = renderer.stream().length;
    player.end({ generation: 3, phrase: 1, text: "Done.", pause_ms: 0 });
    const stream = renderer.stream();
    assert(stream.length > beforeEnd, "the held tail must be released when the phrase ends");
    assert(Math.abs(stream[stream.length - 1]) < 0.01,
      "the last sample must be silence, got " + stream[stream.length - 1]);
    assert(Math.abs(stream[beforeEnd - 1]) > 0.4,
      "and the fade must not have started before the tail");
  } finally { player.dispose(); }
});

test("the ending fade does not depend on when the phrase-end message arrives", async () => {
  // It used to: the fade was scheduled onto audio that might already have been played, so a
  // late message meant no fade at all. Holding the tail makes it unconditional.
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Late." });
    const pcm = new Uint8Array(12000);
    for (let i = 0; i < 6000; i++) new DataView(pcm.buffer).setInt16(i * 2, 16000, true);
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm });
    // Everything pushed so far has already been played by the time the end arrives.
    renderer.play(renderer.stream().length);
    player.end({ generation: 3, phrase: 1, text: "Late.", pause_ms: 0 });
    const stream = renderer.stream();
    assert(Math.abs(stream[stream.length - 1]) < 0.01,
      "the phrase still has to end on silence, got " + stream[stream.length - 1]);
  } finally { player.dispose(); }
});

test("the pause after a phrase belongs to the stream, not to the clock", async () => {
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Done." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    const beforePause = renderer.stream().length;
    player.end({ generation: 3, phrase: 1, text: "Done.", pause_ms: 200 });
    const added = renderer.stream().length - beforePause;
    const fade = Math.round(0.006 * 24000);
    assert(Math.abs(added - fade - 0.2 * 24000) <= 2,
      "200 ms of pause must be 200 ms of stream, got " + (added - fade));
  } finally { player.dispose(); }
});

test("credit follows what was heard, not what was delivered", async () => {
  const { player, renderer, sent } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Credit." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: speech(6000, 77) });
    player.tick();
    assert.deepEqual(sent.filter((e) => e.type === "audio_played").map((e) => e.sequence), [],
      "nothing has been heard yet, so nothing may be credited");
    renderer.play(6000);
    player.tick();
    assert.deepEqual(sent.filter((e) => e.type === "audio_played").map((e) => e.sequence), [1],
      "the first block has been heard and only that one");
    renderer.play(renderer.stream().length);
    player.tick();
    assert.deepEqual(sent.filter((e) => e.type === "audio_played").map((e) => e.sequence), [1, 2]);
  } finally { player.dispose(); }
});

test("a phrase's position is what has been heard of it, and its length once synthesis ends", async () => {
  // The board puts a phrase's sentences up from this, so it has to count what was heard of this
  // phrase alone - not the phrases before it in the stream - and know the length only once the
  // phrase is complete.
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "First." });
    assert.equal(player.position(1), null, "nothing of it has arrived");
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(2400) });
    player.end({ generation: 3, phrase: 1, text: "First.", pause_ms: 0 });
    player.begin({ generation: 3, phrase: 2, text: "Second." });
    player.queue({ generation: 3, phrase: 2, sequence: 2, pcm: speech(4800) });
    const second = player.phrases.get(2).from;
    renderer.play(second + 2400);
    const midway = player.position(2);
    assert.equal(Math.round(midway.heard), 100, "2,400 frames of it at 24 kHz is 100 ms");
    assert.equal(midway.length, null, "still being synthesised");
    player.end({ generation: 3, phrase: 2, text: "Second.", pause_ms: 200 });
    // Its own audio, without the pause the engine asked for after it.
    assert.equal(Math.round(player.position(2).length), 200);
  } finally { player.dispose(); }
});

test("the device's own latency is not credited as heard", async () => {
  // What the renderer has pulled is not yet what anyone has heard: the device holds its own
  // buffer beyond it. Time is what clears that buffer, so time is what this advances - feeding
  // the renderer more frames than it was given would prove nothing.
  const { player, renderer, audio, sent } = await playback();
  try {
    audio.outputLatency = 0.05;
    player.begin({ generation: 3, phrase: 1, text: "Latency." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(1200) });
    renderer.play(renderer.stream().length);
    player.tick();
    assert.equal(sent.filter((e) => e.type === "audio_played").length, 0,
      "50 ms of device buffer still holds it");
    audio.currentTime += 0.05;
    player.tick();
    assert.equal(sent.filter((e) => e.type === "audio_played").length, 1);
  } finally { player.dispose(); }
});

test("the last phrase of a reply completes without any more audio being sent", async () => {
  // The bug this exists for. Completion used to be measured as the played position minus the
  // output latency, but the played position stops rising the moment the queue empties - so the
  // final phrase could never reach its own end. Zen stayed in its speaking state and the reply
  // never entered the conversation. Nothing else is coming; only the clock finishes it.
  const { player, renderer, audio, sent, phrases } = await playback();
  try {
    audio.outputLatency = 0.08;
    player.begin({ generation: 3, phrase: 1, text: "The end." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.end({ generation: 3, phrase: 1, text: "The end.", pause_ms: 0 });
    renderer.play(renderer.stream().length);
    player.tick();
    assert.equal(sent.filter((e) => e.type === "played").length, 0,
      "the device has not finished emptying yet");
    audio.currentTime += 0.08;
    player.tick();
    assert.equal(sent.filter((e) => e.type === "played").length, 1,
      "the phrase must complete on the clock alone");
    assert(phrases.some(([kind]) => kind === "end"), "and be handed to the conversation");
    assert.equal(player.pending(), 0, "with nothing left outstanding");
  } finally { player.dispose(); }
});

test("the renderer is told to buffer while a phrase is open and to stop when none is", async () => {
  // Buffering is a state. With nothing outstanding the target is zero, so a reply that has
  // simply finished is not mistaken for the queue running dry.
  const { player, renderer } = await playback();
  try {
    assert.equal(renderer.target, 0, "nothing is expected before a phrase begins");
    player.begin({ generation: 3, phrase: 1, text: "Buffer." });
    assert(renderer.target > 0, "an open phrase means more audio is coming");
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.end({ generation: 3, phrase: 1, text: "Buffer.", pause_ms: 0 });
    assert.equal(renderer.target, 0, "and once it has ended, nothing more is expected");
  } finally { player.dispose(); }
});

test("running dry deepens the buffer rather than recovering to the same depth", async () => {
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Deeper." });
    const first = renderer.target;
    renderer.underruns = 1;
    renderer.report();
    assert(renderer.target > first,
      "a buffer that has been shown to be too shallow must not be restored to its old depth");
    const grown = renderer.target;
    for (let i = 2; i < 40; i++) { renderer.underruns = i; renderer.report(); }
    assert(renderer.target > grown, "and it keeps growing while the evidence keeps arriving");
    assert(renderer.target <= Math.round(0.32 * 24000) + 1, "but not without limit");
  } finally { player.dispose(); }
});

test("a phrase that finished playing is credited even if the turn is abandoned next", async () => {
  // A phrase can finish in the twenty milliseconds between one tick and the next. Tearing the
  // turn down used to drop its marker, so a phrase the listener had heard in full appeared
  // nowhere afterwards: not on screen, and not in the conversation the model is given, because
  // the engine learns that a phrase was spoken only from this acknowledgement.
  const { player, renderer, sent, phrases } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "All of this was heard." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.end({ generation: 3, phrase: 1, text: "All of this was heard.", pause_ms: 0 });
    // Heard in full, but no tick has run yet.
    renderer.play(renderer.stream().length);
    assert.deepEqual(sent, [], "nothing has been reported yet");
    player.clear();
    assert(
      sent.some((e) => e.type === "played" && e.phrase === 1),
      "the finished phrase must still be credited: " + JSON.stringify(sent),
    );
    assert(
      phrases.some(([kind, text]) => kind === "end" && text === "All of this was heard."),
      "and reach the transcript",
    );
  } finally { player.dispose(); }
});

test("new generations discard old playback and stale phrase events", async () => {
  const { player, renderer, sent, phrases } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Old." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    const cleared = renderer.clears;
    player.reset(4);
    assert.equal(renderer.clears, cleared + 1, "the renderer must be emptied, not left to finish");
    assert.equal(renderer.stream().length, 0);
    player.tick();
    assert.deepEqual(sent.filter((e) => e.generation === 3), [],
      "nothing from the abandoned turn may still be credited");
    player.begin({ generation: 4, phrase: 1, text: "New." });
    player.queue({ generation: 4, phrase: 1, sequence: 1, pcm: speech(6000) });
    renderer.play(renderer.stream().length);
    player.tick();
    assert(phrases.some(([kind, , , generation]) => kind === "start" && generation === 4));
  } finally { player.dispose(); }
});

test("a suspended context credits nothing", async () => {
  const { player, renderer, audio, sent } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Suspended." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    renderer.play(renderer.stream().length);
    audio.state = "suspended";
    player.tick();
    assert.deepEqual(sent, [], "a context that is not running has played nothing");
    audio.state = "running";
    player.tick();
    assert(sent.some((e) => e.type === "audio_played"));
  } finally { player.dispose(); }
});

test("malformed audio and out-of-order sequences fail explicitly", async () => {
  const { player } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Bad." });
    assert.throws(() => player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(0) }));
    assert.throws(() => player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: new Uint8Array(7) }));
    assert.throws(() => player.queue({ generation: 3, phrase: 2, sequence: 1, pcm: speech(100) }),
      "a block for a phrase that never began is not playable");
    player.queue({ generation: 3, phrase: 1, sequence: 5, pcm: speech(100) });
    assert.throws(() => player.queue({ generation: 3, phrase: 1, sequence: 5, pcm: speech(100) }),
      "sequences must advance");
    assert.throws(() => player.end({ generation: 3, phrase: 9, text: "", pause_ms: 0 }));
  } finally { player.dispose(); }
});

test("a phrase never waits for the next phrase to be credited", async () => {
  const { player, renderer, sent } = await playback();
  try {
    queue(player, 1, 1);
    renderer.play(renderer.stream().length);
    player.tick();
    assert(sent.some((e) => e.type === "played" && e.phrase === 1),
      "the first phrase must complete on its own audio");
  } finally { player.dispose(); }
});

/// Amplitude at one frequency, by correlation. Windowed, so a strong neighbour cannot leak
/// into the bin being measured and hide what is there.
function amplitudeAt(samples, hz, rate) {
  let real = 0, imaginary = 0;
  for (let n = 0; n < samples.length; n++) {
    const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (samples.length - 1));
    const angle = (2 * Math.PI * hz * n) / rate;
    real += samples[n] * window * Math.cos(angle);
    imaginary += samples[n] * window * Math.sin(angle);
  }
  return Math.sqrt(real * real + imaginary * imaginary) / samples.length;
}

test("converting to the device rate leaves no mirror image of the signal", async () => {
  // What WebAudio does if it is left to convert a 24 kHz buffer itself, measured in the engine
  // this app runs on: a 6 kHz tone came back with an image at 18 kHz 15.3 dB below it. Audible
  // as the metallic edge on the voice. A band-limited conversion has to leave that image far
  // enough down to be nothing at all.
  const { Resampler } = await import("../src/client/audio.mjs");
  const resampler = new Resampler(24000, 48000);
  const tone = new Float32Array(24000);
  for (let i = 0; i < tone.length; i++) tone[i] = 0.8 * Math.sin((2 * Math.PI * 6000 * i) / 24000);
  const out = resampler.process(tone).slice(2000, -2000);
  const wanted = amplitudeAt(out, 6000, 48000);
  const image = amplitudeAt(out, 18000, 48000);
  const db = 20 * Math.log10(image / wanted);
  assert(db < -80, "the mirror image sits only " + db.toFixed(1) + " dB down");
  const direct = new Float32Array(out.length);
  for (let i = 0; i < direct.length; i++)
    direct[i] = 0.8 * Math.sin((2 * Math.PI * 6000 * i) / 48000);
  const level = 20 * Math.log10(wanted / amplitudeAt(direct, 6000, 48000));
  assert(Math.abs(level) < 0.1,
    "and the signal itself must come through at its own level, off by " + level.toFixed(3) + " dB");
});

test("a stream converted in blocks is identical to the same stream converted whole", async () => {
  // The property the whole change rests on. The block boundary must leave no trace, or it has
  // simply moved the seam defect from WebAudio into this code.
  const { Resampler } = await import("../src/client/audio.mjs");
  const source = new Float32Array(24000);
  for (let i = 0; i < source.length; i++)
    source[i] = 0.7 * Math.sin((2 * Math.PI * 3210 * i) / 24000) +
                0.2 * Math.sin((2 * Math.PI * 9070 * i) / 24000);

  const whole = new Resampler(24000, 48000).process(source);
  const pieces = new Resampler(24000, 48000);
  const parts = [];
  for (let off = 0; off < source.length; off += 6000)
    parts.push(pieces.process(source.subarray(off, off + 6000)));
  const joined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const part of parts) { joined.set(part, at); at += part.length; }

  const shared = Math.min(whole.length, joined.length);
  let worst = 0;
  for (let i = 0; i < shared; i++) worst = Math.max(worst, Math.abs(whole[i] - joined[i]));
  assert(worst < 1e-6, "a block boundary changed the waveform by " + worst);
  const short = source.length * 2 - joined.length;
  assert(short >= 0 && short <= Resampler.TAPS + 2,
    "the only audio owed at the end is the kernel width still held, got " + short);
});

test("a rate the device already speaks is passed through untouched", async () => {
  const { Resampler } = await import("../src/client/audio.mjs");
  const input = new Float32Array([0.1, -0.2, 0.3, -0.4]);
  const out = new Resampler(24000, 24000).process(input);
  assert.deepEqual(Array.from(out), Array.from(input));
});

test("what reaches the renderer is already at the device rate", async () => {
  // The point of converting here: the renderer plays its queue sample for sample, so what it
  // is handed has to be in the device timeline already. A 0.25 s block at 24 kHz has to arrive
  // as 0.25 s of the device rate, not as 6000 frames of something else.
  const { player, renderer } = await playback(48000);
  try {
    player.begin({ generation: 3, phrase: 1, text: "Rate check." });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.queue({ generation: 3, phrase: 1, sequence: 2, pcm: speech(6000, 77) });
    player.end({ generation: 3, phrase: 1, text: "Rate check.", pause_ms: 0 });
    const seconds = renderer.stream().length / 48000;
    assert(Math.abs(seconds - 0.5) < 0.002,
      "half a second of speech must be half a second of device frames, got " + seconds);
  } finally { player.dispose(); }
});


/// Loads the real render.js with the globals an AudioWorklet would provide, and returns an
/// instance plus the messages it sends back. A stand-in cannot catch a bug in the thing it
/// stands in for, and both of the faults below were in this file.
async function renderer(rate = 48000) {
  const source = await fsp.readFile(require("path").resolve(__dirname, "../src/client/render.js"), "utf8");
  let made = null;
  const messages = [];
  const scope = {
    sampleRate: rate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {
          onmessage: null,
          postMessage: (m) => messages.push(m),
          send: (data) => this.port.onmessage({ data }),
        };
      }
    },
    registerProcessor: (_name, klass) => { made = klass; },
  };
  new Function(...Object.keys(scope), source)(...Object.values(scope));
  const node = new made();
  const quantum = () => {
    const out = [new Float32Array(128)];
    node.process([], [out]);
    return out[0];
  };
  return { node, messages, quantum, send: (data) => node.port.onmessage({ data }) };
}

/// A block of constant full-scale audio, so any change in amplitude is unmistakable.
function loud(frames) {
  return new Float32Array(frames).fill(1);
}

test("an abandoned turn is ramped away and never bursts back to full volume", async () => {
  // The ramp used to stop at its own length while the read did not, so whatever was left in
  // that render quantum went out at its original amplitude - a burst of the turn that was just
  // cancelled, at full volume, which is the one thing the ramp exists to prevent.
  const { node, quantum, send } = await renderer(48000);
  send({ type: "target", frames: 0 });
  send({ type: "push", pcm: loud(4096).buffer });
  quantum();
  send({ type: "clear", epoch: 1 });
  const cutFrames = Math.round(0.004 * 48000);
  let worstAfterFade = 0, seen = 0;
  for (let q = 0; q < 8; q++) {
    const out = quantum();
    for (let i = 0; i < out.length; i++, seen++)
      if (seen >= cutFrames) worstAfterFade = Math.max(worstAfterFade, Math.abs(out[i]));
  }
  assert(worstAfterFade < 1e-6,
    "audio escaped after the ramp finished, at " + worstAfterFade);
  void node;
});

test("the ramp of an abandoned turn falls monotonically to silence", async () => {
  const { quantum, send } = await renderer(48000);
  send({ type: "target", frames: 0 });
  send({ type: "push", pcm: loud(4096).buffer });
  quantum();
  send({ type: "clear", epoch: 1 });
  let previous = Infinity, reached = 0;
  for (let q = 0; q < 4; q++)
    for (const value of quantum()) {
      const level = Math.abs(value);
      assert(level <= previous + 1e-6, "the ramp went back up: " + level + " after " + previous);
      previous = level;
      reached = level;
    }
  assert.equal(reached, 0, "and must end at silence");
});

test("audio pushed during a cancellation ramp is not discarded with the old turn", async () => {
  // The page sends the next turn's audio as soon as it has any, which can be while the
  // previous turn is still ramping away. Sharing one queue meant the discard at the end of the
  // ramp threw the new turn's opening blocks out along with the old turn's tail.
  const { quantum, send, messages } = await renderer(48000);
  send({ type: "target", frames: 0 });
  send({ type: "push", pcm: loud(4096).buffer });
  quantum();
  send({ type: "clear", epoch: 2 });
  const fresh = new Float32Array(512).fill(0.5);
  send({ type: "push", pcm: fresh.buffer });
  for (let q = 0; q < 12; q++) quantum();
  const last = messages.filter((m) => m.type === "progress").at(-1);
  assert.equal(last.epoch, 2, "reports must belong to the turn that replaced the old one");
  assert(last.consumed > 0,
    "the new turn's audio has to reach the device, got consumed " + last.consumed);
});

test("an idle renderer does not flood the page with reports", async () => {
  // An empty renderer reporting every render quantum is several hundred messages a second on
  // the page's main thread, which is the thread this design exists to keep free.
  const { quantum, send, messages } = await renderer(48000);
  send({ type: "target", frames: 0 });
  messages.length = 0;
  for (let q = 0; q < 80; q++) quantum();
  assert(messages.length <= 12,
    "80 quanta of silence produced " + messages.length + " messages");
});

test("a shortfall while more audio is expected is counted, and one at the end is not", async () => {
  const { quantum, send, messages } = await renderer(48000);
  send({ type: "target", frames: 0 });
  send({ type: "push", pcm: loud(256).buffer });
  // Far enough past the audio to have reported at least once on the throttled interval.
  for (let q = 0; q < 24; q++) quantum();
  assert.equal(messages.filter((m) => m.type === "progress").at(-1).underruns, 0,
    "a reply that has simply finished has not run dry");
  send({ type: "target", frames: 4800 });
  send({ type: "push", pcm: loud(4800).buffer });
  for (let q = 0; q < 60; q++) quantum();
  assert(messages.filter((m) => m.type === "progress").at(-1).underruns > 0,
    "running out with more expected is an underrun and must be reported");
});

test("starvation fades to silence and recovery fades in without losing frames", async () => {
  for (const frames of [61, 128]) {
    const { node, quantum, send } = await renderer(48000);
    send({ type: "target", frames: 48 });
    send({ type: "push", pcm: new Float32Array(frames).fill(0.5).buffer });
    const tail = quantum();
    assert.equal(tail[frames - 1], 0, "the open stream must end on zero");
    assert.equal(node.consumed, frames);
    assert.equal(node.underruns, 1);
    assert(quantum().every((sample) => sample === 0));
    send({ type: "push", pcm: new Float32Array(512).fill(0.5).buffer });
    const resumed = quantum();
    assert.equal(resumed[0], 0, "rebuffering must not restart at full amplitude");
    for (let i = 1; i < resumed.length; i++) assert(resumed[i] >= resumed[i - 1]);
    assert.equal(node.consumed, frames + 128, "fades must not consume extra speech");
  }
});

test("a failed playback processor is retired and a new turn can create a working one", async () => {
  const { player, renderer, sent } = await playback();
  const errors = [];
  player.onError = (error) => errors.push(error);
  try {
    player.begin({ generation: 3, phrase: 1, text: "Old reply" });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    renderer.onprocessorerror();
    assert.equal(errors.length, 1);
    assert.equal(player.node, null);
    assert.equal(renderer.port.onmessage, null);
    assert.equal(player.pending(), 0);
    assert.equal(sent.length, 0, "lost audio was not heard");
    player.reset(4);
    await player.open();
    assert(player.node && player.node !== renderer);
    player.begin({ generation: 4, phrase: 2, text: "Fresh reply" });
    player.queue({ generation: 4, phrase: 2, sequence: 2, pcm: speech(6000) });
    assert(player.node.stream().length > 0);
  } finally { player.dispose(); }
});

test("a renderer module load failure is reported once and may be retried", async () => {
  const { AudioPlayback } = await import("../src/client/audio.mjs");
  const { audio } = audioGraph();
  audio.audioWorklet.addModule = async () => { throw new Error("module unavailable"); };
  const errors = [];
  const player = new AudioPlayback(audio, () => true, () => {}, (error) => errors.push(error));
  try {
    await player.loading;
    assert.equal(errors.length, 1);
    assert.equal(player.loading, null);
    audio.audioWorklet.addModule = async () => {};
    player.reset(1);
    await player.open();
    assert(player.node);
  } finally { player.dispose(); }
});

test("a refused acknowledgement stops credit without recursive sends during clear", async () => {
  const { player, renderer } = await playback();
  try {
    player.begin({ generation: 3, phrase: 1, text: "Hello" });
    player.queue({ generation: 3, phrase: 1, sequence: 1, pcm: speech(6000) });
    player.end({ generation: 3, phrase: 1, text: "Hello", pause_ms: 0 });
    renderer.play(renderer.stream().length);
    let attempts = 0;
    player.send = () => { attempts++; return false; };
    assert.equal(player.credit(), false);
    assert.equal(attempts, 1);
    assert.equal(player.markers.length, 0);
  } finally { player.dispose(); }
});
