const test = require('node:test');
const assert = require('node:assert/strict');
const tick = () => new Promise(resolve => setImmediate(resolve));

test('queued capture can expire without shifting the in-flight frame or losing controls', async () => {
  const { OrderedInput } = await import('../src/client/transport.mjs');
  let release;
  const received = [];
  const input = new OrderedInput(async (_, frame) => {
    received.push([frame[8], frame[9]]);
    if (received.length === 1) await new Promise(resolve => { release = resolve; });
  }, 7, error => { throw error; });
  input.send(0, Uint8Array.of(1));
  for (let i = 2; i <= 100; i++) input.send(0, Uint8Array.of(i));
  input.send(1, Uint8Array.of(255));
  release();
  await tick();
  assert.equal(input.dropped, 49);
  assert.deepEqual(received, [[0, 1], ...Array.from({length: 50}, (_, i) => [0, i + 51]), [1, 255]]);
  input.close();
});

test('shed capture is reported rather than discarded in silence', async () => {
  // Shedding throws away 20 ms of speech per frame. Recognition gets worse and nothing else
  // in the pipeline can tell that it did, so the page has to be told.
  const { OrderedInput } = await import('../src/client/transport.mjs');
  const shed = [];
  let release;
  const input = new OrderedInput(async () => {
    if (!release) await new Promise(resolve => { release = resolve; });
  }, 3, error => { throw error; }, { onShed: total => shed.push(total) });
  for (let i = 0; i < 40; i++) input.send(0, Uint8Array.of(i));
  assert.deepEqual(shed, [], 'a queue under the limit sheds nothing');
  for (let i = 0; i < 20; i++) input.send(0, Uint8Array.of(i));
  assert.ok(shed.length > 0, 'exceeding the limit must report');
  assert.equal(shed.at(-1), input.dropped, 'the report carries the running total');
  // Controls are never shed, however long the capture backlog.
  input.send(1, Uint8Array.of(9));
  assert.equal(input.queue.filter(item => item[8] === 1).length, 1);
  input.close();
});

test('a retired session cannot drain a new session or report a stale failure', async () => {
  const { OrderedInput } = await import('../src/client/transport.mjs');
  let fail;
  const sent = [], errors = [];
  const old = new OrderedInput(() => new Promise((_, reject) => { fail = reject; }), 1, e => errors.push(e));
  old.send(0, Uint8Array.of(1));
  old.send(1, Uint8Array.of(2));
  old.close();
  const fresh = new OrderedInput(async (_, frame) => sent.push(frame), 2, e => errors.push(e));
  fresh.send(1, Uint8Array.of(3));
  fail(new Error('Old IPC failed'));
  await tick();
  assert.equal(errors.length, 0);
  assert.equal(sent.length, 1);
  assert.equal(new DataView(sent[0].buffer).getBigUint64(0, true), 2n);
  assert.equal(sent[0][9], 3);
});

test('a failed control ends the transport instead of silently dropping it', async () => {
  const { OrderedInput } = await import('../src/client/transport.mjs');
  const errors = [];
  const input = new OrderedInput(async () => { throw new Error('queue stalled'); }, 1, e => errors.push(e));
  input.send(1, Uint8Array.of(1));
  await tick();
  assert.equal(errors.length, 1);
  assert.equal(input.closed, true);
  assert.equal(input.send(1, Uint8Array.of(2)), false);
});

test('hung IPC times out and a control-only backlog has a hard bound', async () => {
  const { OrderedInput } = await import('../src/client/transport.mjs');
  const errors = [];
  const input = new OrderedInput(() => new Promise(() => {}), 1, e => errors.push(e), {timeout: 20});
  input.send(1, new Uint8Array());
  await new Promise(resolve => setTimeout(resolve, 40));
  assert.equal(errors.length, 1);
  const full = new OrderedInput(() => new Promise(() => {}), 2, e => errors.push(e), {timeout: 20});
  for (let i = 0; i < 258; i++) full.send(1, new Uint8Array());
  assert.equal(full.closed, true);
  assert.equal(full.queue.length, 0);
  assert.equal(errors.length, 2);
});

test('a playback acknowledgement does not wait behind a second of captured speech', async () => {
  // The loop this exists to break. The engine stops sending audio at about two seconds
  // unacknowledged, so an acknowledgement stuck behind fifty queued capture frames starves the
  // player of the audio it is about to need. Nothing about an acknowledgement is sequenced
  // against speech - it describes the output stream - so it goes first.
  const { OrderedInput, PRIORITY_CONTROLS } = await import('../src/client/transport.mjs');
  assert.ok(PRIORITY_CONTROLS.has('audio_played'), 'playback progress must be a priority control');
  const order = [];
  let release;
  const input = new OrderedInput(async (_, frame) => {
    order.push(frame[8] === 0 ? 'capture' : JSON.parse(new TextDecoder().decode(frame.slice(9))).type);
    if (order.length === 1) await new Promise(resolve => { release = resolve; });
  }, 5, error => { throw error; });
  const control = (type, priority) =>
    input.send(1, new TextEncoder().encode(JSON.stringify({ type })), priority);

  input.send(0, Uint8Array.of(1));                 // in flight, holding the pump
  for (let i = 0; i < 40; i++) input.send(0, Uint8Array.of(i));
  control('end_audio', PRIORITY_CONTROLS.has('end_audio'));
  control('audio_played', PRIORITY_CONTROLS.has('audio_played'));
  release();
  await tick();

  const ack = order.indexOf('audio_played');
  const ended = order.indexOf('end_audio');
  assert.equal(order[1], 'audio_played',
    'the acknowledgement must be sent as soon as the pump is free: ' + order.slice(0, 3));
  assert.ok(ack < ended, 'and ahead of the capture backlog, not behind it');
  assert.equal(ended, order.length - 1,
    'while end_audio stays behind the speech it marks the end of');
  input.close();
});
