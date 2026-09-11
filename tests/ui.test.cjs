const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
let browser, server, origin;
const root = path.resolve(__dirname, '../src/client');

before(async () => {
  server = http.createServer(async (req, res) => {
    const name = new URL(req.url, 'http://localhost').pathname;
    const file = path.resolve(root, '.' + (name === '/' ? '/index.html' : name));
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      const content = await fs.readFile(file);
      const types = {'.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2'};
      res.writeHead(200, {'Content-Type': types[path.extname(file)] || 'application/octet-stream'}).end(content);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({channel: process.env.ZEN_TEST_BROWSER || 'msedge', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required']});
  await fs.mkdir(path.resolve(__dirname, '../target/ui'), {recursive: true});
});
after(async () => { await browser?.close(); await new Promise(resolve => server ? server.close(resolve) : resolve()); });

async function pageFor(options = {}) {
  const context = await browser.newContext({viewport: {width: 980, height: 760}, colorScheme: options.dark ? 'dark' : 'light'});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(({delayReady, delayAttach, failStartup}) => {
    const encoder = new TextEncoder();
    const fixture = window.__test = {commands: [], frames: [], generation: 0, revision: 0};
    fixture.emit = (event, index = fixture.frames.length - 1) => {
      const json = encoder.encode(JSON.stringify(event));
      const frame = new Uint8Array(1 + json.length);
      frame[0] = 1; frame.set(json, 1);
      fixture.frames[index].onmessage(frame.buffer);
    };
    window.__TAURI__ = {core: {
      Channel: class { onmessage() {} },
      async invoke(command, args) {
        fixture.commands.push({command, args: command === 'zen_input' ? null : args});
        if (command === 'zen_attach') {
          fixture.frames.push(args.frames);
          fixture.ready = () => fixture.emit(failStartup
            ? {type: 'error', code: 'backend_unavailable', detail: 'model is missing: C:/zen-ai/model/E2B/gemma.gguf'}
            : {type: 'ready'});
          if (!delayReady) fixture.ready(); // Deliberately before the attach promise resolves.
          if (delayAttach) await new Promise(resolve => { fixture.attach = resolve; });
          return ++fixture.revision;
        }
        if (command === 'zen_input') {
          if (args[8] === 0) { fixture.captureFrames = (fixture.captureFrames || 0) + 1; return; }
          const event = JSON.parse(new TextDecoder().decode(args.subarray(9)));
          fixture.commands.at(-1).event = event;
          if (event.type === 'text') {
            const generation = ++fixture.generation;
            fixture.emit({type: 'clear', generation});
            fixture.emit({type: 'transcript', generation, text: event.text});
            fixture.emit({type: 'state', generation, phase: 'thinking'});
          } else if (event.type === 'clear_history') {
            fixture.emit({type: 'clear', generation: ++fixture.generation});
            fixture.emit({type: 'history_cleared'});
            fixture.emit({type: 'state', generation: fixture.generation, phase: 'idle'});
          } else if (event.type === 'interrupt') {
            fixture.emit({type: 'clear', generation: ++fixture.generation});
            fixture.emit({type: 'state', generation: fixture.generation, phase: 'idle'});
          }
        }
      },
    }};
  }, options);
  await page.goto(origin);
  return {page, context, errors};
}

test('settings work before startup, and the first typed message sends after early ready', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByLabel('Instructions for this conversation', {exact: true}).fill('Keep it brief.');
    await page.getByRole('button', {name: 'Close settings'}).click();
    await page.getByLabel('Message Zen', {exact: true}).fill('Hello, Zen');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.text === 'Hello, Zen'));
    assert.equal(await page.locator('#captionText').textContent(), 'Hello, Zen');
    assert.equal(await page.evaluate(() => __test.commands.find(c => c.command === 'zen_attach').args.systemPrompt), 'Keep it brief.');
    assert.equal(await page.locator('#text').inputValue(), '');
    await page.getByRole('button', {name: 'Show conversation'}).click();
    assert.equal(await page.locator('#transcriptPanel').evaluate(el => el.open), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('button', {name: 'Show conversation'}).getAttribute('aria-expanded'), 'false');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('one microphone click waits for models, starts capture, and mute flushes the input', async () => {
  const {page, context, errors} = await pageFor({delayReady: true});
  try {
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Loading models…'}).waitFor();
    assert.equal(await page.evaluate(() => __test.captureFrames || 0), 0);
    await page.evaluate(() => __test.ready());
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await page.waitForFunction(() => __test.captureFrames > 2);
    await page.getByRole('button', {name: 'Mute microphone'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'end_audio'));
    assert.equal(await page.locator('#mic').getAttribute('aria-pressed'), 'false');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('unconfirmed microphone noise preserves a reply; confirmed speech clears it', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.evaluate(async () => {
      const {Microphone, AudioPlayback} = await import('/audio.mjs');
      const start = Microphone.prototype.start;
      Microphone.prototype.start = function (...args) {
        __test.mic = this;
        return start.apply(this, args);
      };
      const begin = AudioPlayback.prototype.begin;
      AudioPlayback.prototype.begin = function (...args) {
        __test.playback = this;
        return begin.apply(this, args);
      };
    });
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    const afterHint = await page.evaluate(() => {
      __test.emit({type: 'state', generation: 1, phase: 'preparing'});
      __test.emit({type: 'phrase_start', generation: 1, phrase: 1, text: 'Keep speaking.'});
      const frame = new Uint8Array(25 + 48000);
      const view = new DataView(frame.buffer);
      for (const offset of [1, 9, 17]) view.setBigUint64(offset, 1n, true);
      __test.frames.at(-1).onmessage(frame.buffer);
      __test.emit({type: 'state', generation: 1, phase: 'speaking'});
      __test.mic.callbacks.onOnset();
      return {accepting: __test.playback.accepting, sources: __test.playback.sources.size};
    });
    assert.equal(afterHint.accepting, true, 'an energy hint cannot revoke the reply');
    assert.equal(afterHint.sources, 1, 'the queued reply must remain playable');
    const afterSpeech = await page.evaluate(() => {
      __test.emit({type: 'clear', generation: 2});
      __test.emit({type: 'state', generation: 2, phase: 'listening'});
      return {sources: __test.playback.sources.size, generation: __test.playback.generation};
    });
    assert.deepEqual(afterSpeech, {sources: 0, generation: 2});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the conversation shows the filter result, never the raw recogniser hypothesis', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.locator('#text').fill('start');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.frames.length > 0);
    await page.evaluate(() => {
      __test.emit({type: 'clear', generation: 1});
      __test.emit({type: 'transcript_partial', generation: 1, text: 'नमस्ते आप कैसे हैं'});
      __test.emit({type: 'transcript', generation: 1, filtered: true, text: 'Hello, how are you?'});
    });
    assert.equal(await page.locator('#captionText').textContent(), 'Hello, how are you?');
    assert.equal(await page.locator('.transcript-entry').last().innerText(), 'YouHello, how are you?');
    assert.equal((await page.locator('.transcript-entry').allInnerTexts()).join(' ').includes('नमस्ते'), false);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('cancelling a pending attach prevents late ready from reviving capture', async () => {
  const {page, context, errors} = await pageFor({delayAttach: true, delayReady: true});
  try {
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.waitForFunction(() => __test.attach);
    await page.getByRole('button', {name: 'End session', exact: true}).click();
    await page.evaluate(() => { __test.ready(); __test.attach(); });
    await page.waitForFunction(() => __test.commands.some(c => c.command === 'zen_detach'));
    assert.equal(await page.locator('#connectionLabel').textContent(), 'On this device');
    assert.equal(await page.locator('#mic').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.evaluate(() => __test.captureFrames || 0), 0);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('startup failure restores usable controls and explains the failure', async () => {
  const {page, context, errors} = await pageFor({failStartup: true});
  try {
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('local engine stopped'));
    // A named file is something the person can go and fix; "check the model files" is not.
    assert.match(await page.locator('#status').textContent(), /gemma\.gguf/);
    assert.equal(await page.locator('#mic').isEnabled(), true);
    assert.equal(await page.locator('#mic').getAttribute('aria-pressed'), 'false');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('new conversation clears displayed history only after the engine acknowledges it', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.locator('#text').fill('Remember this test message');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => document.querySelectorAll('.transcript-entry').length === 1);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.locator('#prompt').fill('Ask one question at a time.');
    await page.getByRole('button', {name: 'Clear history & start fresh'}).click();
    await page.waitForFunction(() => document.querySelectorAll('.transcript-entry').length === 0);
    assert.equal(await page.evaluate(() => __test.commands.find(c => c.event?.type === 'clear_history').event.system_prompt), 'Ask one question at a time.');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the theme choice overrides the system preference and is remembered', async () => {
  // The page must honour an explicit choice in both directions: a dark system with Light
  // selected has to actually be light, which a `prefers-color-scheme` rule alone cannot do.
  const {page, context, errors} = await pageFor({dark: true});
  try {
    const background = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    const systemDark = await background();
    await page.getByRole('button', {name: 'Light', exact: true}).click();
    const forcedLight = await background();
    assert.notEqual(forcedLight, systemDark, 'Light must override a dark system');
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'light');
    await page.getByRole('button', {name: 'Night', exact: true}).click();
    assert.equal(await background(), systemDark);
    await page.reload();
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'dark',
      'the choice must survive a reload');
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.getByRole('button', {name: 'Night', exact: true}).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', {name: 'System', exact: true}).click();
    assert.equal(await page.evaluate(() => 'theme' in document.documentElement.dataset), false,
      'System must stamp nothing so prefers-color-scheme decides');
    assert.equal(await background(), systemDark);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the pause is left to the engine by default and can be taken over by hand', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const tunings = () => page.evaluate(() =>
      __test.commands.filter(c => c.event?.type === 'tuning').map(c => c.event.endpoint_ms));
    await page.locator('#text').fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'tuning'));
    // Null hands the decision to the segmenter, which learns it from how this person pauses.
    assert.deepEqual(await tunings(), [null]);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.locator('#endpointField').isVisible(), false);
    // The engine reports what it settled on, and the interface has to say so.
    await page.evaluate(() => __test.emit({type: 'endpoint', ms: 1080}));
    assert.match(await page.locator('#endpointNote').textContent(), /1\.08 seconds/);
    await page.getByRole('button', {name: 'Set myself', exact: true}).click();
    assert.equal(await page.locator('#endpointField').isVisible(), true);
    await page.locator('#endpoint').fill('1200');
    await page.locator('#endpoint').dispatchEvent('input');
    await page.waitForFunction(() => __test.commands.some(c => c.event?.endpoint_ms === 1200));
    assert.match(await page.locator('#endpointNote').textContent(), /1\.20 seconds/);
    await page.reload();
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.getByRole('button', {name: 'Set myself', exact: true}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#endpoint').inputValue(), '1200');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('light, dark, compact and narrow layouts render without horizontal overflow', async () => {
  for (const [name, width, height, dark] of [['light',980,760,false],['dark',980,760,true],['compact',800,560,false],['narrow',380,560,false]]) {
    const {page, context, errors} = await pageFor({dark});
    try {
      await page.setViewportSize({width, height});
      await page.emulateMedia({reducedMotion: 'reduce'});
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({path: path.resolve(__dirname, `../target/ui/${name}.png`), fullPage: true});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, name);
      await page.getByRole('button', {name: 'Settings', exact: true}).click();
      const bounds = await page.locator('#audioDialog').boundingBox();
      assert(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= width + 1 && bounds.y + bounds.height <= height + 1, name);
      assert.deepEqual(errors, [], name);
    } finally { await context.close(); }
  }
});
