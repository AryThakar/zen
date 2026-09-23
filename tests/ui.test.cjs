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
  const context = await browser.newContext({viewport: {width: 980, height: 760}, colorScheme: options.dark ? 'dark' : 'light',
    reducedMotion: options.reducedMotion ? 'reduce' : 'no-preference'});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  // The intro has a test of its own; everything else starts on the page it lands on.
  if (!options.intro)
    await page.addInitScript(() => document.addEventListener('DOMContentLoaded', () => {
      document.querySelector('.intro')?.remove();
      document.body.classList.add('landed');
    }));
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
    await page.getByRole('button', {name: 'Conversation', exact: true}).click();
    assert.equal(await page.locator('#historySheet').evaluate(el => el.open), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#historySheet').evaluate(el => el.open), false);
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

test('a reply plays on through microphone sound until the engine confirms speech', async () => {
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
    const whileListening = await page.evaluate(() => {
      __test.emit({type: 'state', generation: 1, phase: 'preparing'});
      __test.emit({type: 'phrase_start', generation: 1, phrase: 1, text: 'Keep speaking.'});
      const frame = new Uint8Array(25 + 48000);
      const view = new DataView(frame.buffer);
      for (const offset of [1, 9, 17]) view.setBigUint64(offset, 1n, true);
      __test.frames.at(-1).onmessage(frame.buffer);
      __test.emit({type: 'state', generation: 1, phase: 'speaking'});
      return {accepting: __test.playback.accepting, sources: __test.playback.pending() > 0 ? 1 : 0};
    });
    // The page cuts nothing on its own: sound at the microphone can be a door or the TV.
    await page.waitForFunction(() => __test.captureFrames > 5);
    assert.equal(whileListening.accepting, true, 'microphone sound alone cannot revoke the reply');
    assert.equal(whileListening.sources, 1, 'the queued reply must remain playable');
    assert.equal(await page.evaluate(() => __test.playback.accepting), true);
    const afterSpeech = await page.evaluate(() => {
      __test.emit({type: 'clear', generation: 2});
      __test.emit({type: 'state', generation: 2, phase: 'listening'});
      return {sources: __test.playback.pending() > 0 ? 1 : 0, generation: __test.playback.generation};
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
    assert.equal(await page.locator('.history-entry').last().textContent(), 'Hello, how are you?');
    assert.equal(await page.locator('.history-entry').last().getAttribute('data-speaker'), 'You');
    assert.equal((await page.locator('.history-entry').allTextContents()).join(' ').includes('नमस्ते'), false);
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
    assert.equal(await page.evaluate(() => document.body.dataset.active), 'false');
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
    await page.waitForFunction(() => document.querySelectorAll('.history-entry').length === 1);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.locator('#prompt').fill('Ask one question at a time.');
    await page.getByRole('button', {name: 'Start new'}).click();
    await page.waitForFunction(() => document.querySelectorAll('.history-entry').length === 0);
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
    await page.getByRole('button', {name: 'Dark', exact: true}).click();
    assert.equal(await background(), systemDark);
    await page.reload();
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'dark',
      'the choice must survive a reload');
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.getByRole('button', {name: 'Dark', exact: true}).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', {name: 'System', exact: true}).click();
    assert.equal(await page.evaluate(() => 'theme' in document.documentElement.dataset), false,
      'System must stamp nothing so prefers-color-scheme decides');
    assert.equal(await background(), systemDark);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the pause is learned by the engine and carried into the next session', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const seeds = () => page.evaluate(() =>
      __test.commands.filter(c => c.event?.type === 'learned_pause').map(c => c.event.ms));
    await page.locator('#text').fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    // Nothing learned yet, so nothing to hand back: the engine starts from its own default.
    assert.deepEqual(await seeds(), []);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    // There is no hand-set pause any more.
    assert.equal(await page.getByRole('button', {name: 'Set myself', exact: true}).count(), 0);
    assert.equal(await page.locator('#endpoint').count(), 0);
    // The engine reports what it settled on, and the interface has to say so.
    await page.evaluate(() => __test.emit({type: 'endpoint', ms: 1080}));
    assert.equal(await page.locator('#endpointNote').textContent(), '1.08 s');
    // Next launch, the learned value is shown straight away and handed back to the engine.
    await page.reload();
    await page.locator('#text').fill('Hello again');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'learned_pause'));
    assert.deepEqual(await seeds(), [1080]);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.locator('#endpointNote').textContent(), '1.08 s');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('a partly unheard turn is admitted to, not passed off as a full answer', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.locator('#text').fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    // A notice for an older turn says nothing about this one.
    await page.evaluate(() => __test.emit({type: 'notice', code: 'partly_unheard', generation: __test.generation - 1}));
    assert.doesNotMatch(await page.locator('#status').textContent(), /missed part/);
    await page.evaluate(() => __test.emit({type: 'notice', code: 'partly_unheard', generation: __test.generation}));
    assert.match(await page.locator('#status').textContent(), /missed part of what you said/);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('where the wait before an answer went is shown in settings', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.locator('#text').fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    const note = () => page.locator('#timingNote').textContent();
    const value = () => page.locator('#timingValue').textContent();
    // A report for an older turn is not about the answer now playing.
    await page.evaluate(() => __test.emit({type: 'timing', generation: __test.generation - 1, total_ms: 9000}));
    assert.equal(await value(), '—');
    assert.equal(await note(), 'After Zen’s first answer');
    await page.evaluate(() => __test.emit({
      type: 'timing', generation: __test.generation, total_ms: 1420, recognize_ms: 550,
      queue_ms: 300, repair_ms: 310, think_ms: 200, speak_ms: 360,
    }));
    assert.equal(await value(), '1.42 s');
    assert.equal(await note(),
      'Recognising 0.55 (0.30 queued) · Checking 0.31 · First word 0.20 · First sound 0.36');
    // A stage that did not run, such as asking again after hearing nothing, is left out.
    await page.evaluate(() => __test.emit({
      type: 'timing', generation: __test.generation, total_ms: 800, recognize_ms: 500,
      queue_ms: 0, repair_ms: null, think_ms: null, speak_ms: null,
    }));
    assert.equal(await value(), '0.80 s');
    assert.equal(await note(), 'Recognising 0.50');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('ending the session from settings asks first', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const settings = page.getByRole('button', {name: 'Settings', exact: true});
    await settings.click();
    // The header keeps its own power button for now, so this is the one inside settings.
    const end = page.locator('#endSession');
    // Nothing to end before a session exists.
    assert.equal(await end.isDisabled(), true);
    await page.getByRole('button', {name: 'Close settings'}).click();
    await page.locator('#text').fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    await settings.click();
    await end.click();
    // Asking is all it does: the session is still there.
    assert.equal(await page.locator('#confirmRow').isVisible(), true);
    assert.equal(await page.evaluate(() => __test.commands.some(c => c.command === 'zen_disconnect')), false);
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
    assert.equal(await page.locator('#endRow').isVisible(), true);
    await end.click();
    await page.locator('#confirmEnd').click();
    await page.waitForFunction(() => __test.commands.some(c => c.command === 'zen_disconnect'));
    assert.equal(await page.locator('#audioDialog').evaluate(d => d.open), false);
    // Opening settings again starts from the question, not the confirmation.
    await settings.click();
    assert.equal(await page.locator('#endRow').isVisible(), true);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('quitting from settings asks first, then quits the app itself', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    // Unlike ending a session, quitting is always there: with no session, Zen is still running.
    const quit = page.getByRole('button', {name: /^Quit Zen/});
    assert.equal(await quit.isEnabled(), true);
    await quit.click();
    assert.equal(await page.locator('#quitConfirmRow').isVisible(), true);
    assert.equal(await page.evaluate(() => __test.commands.some(c => c.command === 'zen_quit')), false);
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
    assert.equal(await page.locator('#quitRow').isVisible(), true);
    await quit.click();
    await page.getByRole('button', {name: 'Quit', exact: true}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.command === 'zen_quit'));
    // It is the app that goes, not the session: nothing is asked of the engine first.
    assert.equal(await page.evaluate(() => __test.commands.some(c => c.command === 'zen_disconnect')), false);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('a reply cut off ends in the history where Zen remembers it ending', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.locator('#text').fill('How do I cook pasta?');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    const zen = () => page.evaluate(() =>
      [...document.querySelectorAll('.history-entry[data-speaker="Zen"] p')].map(e => e.textContent));
    const generation = await page.evaluate(() => __test.generation);
    // Nothing of the reply was heard: nothing to show.
    await page.evaluate((g) => __test.emit({type: 'heard', generation: g, text: ''}), generation);
    assert.deepEqual(await zen(), []);
    // Part of the phrase playing at the cut was heard.
    await page.evaluate((g) => __test.emit({type: 'heard', generation: g, text: 'Sure. Boil the water first'}), generation);
    assert.deepEqual(await zen(), ['Sure. Boil the water first —']);
    // A report about another reply is not about this one.
    await page.evaluate((g) => __test.emit({type: 'heard', generation: g - 1, text: 'Old words'}), generation);
    assert.deepEqual(await zen(), ['Sure. Boil the water first —']);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('two quiet minutes turn the microphone off, unless the listener said never', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.evaluate(async () => {
      const {Soundscape} = await import('/audio.mjs');
      __test.tones = [];
      Soundscape.prototype.play = function (kind) { __test.tones.push(kind); };
    });
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    // Turning the microphone on restarts the engine's quiet timer.
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'active'));
    await page.evaluate(() => __test.emit({type: 'quiet'}));
    await page.getByRole('button', {name: 'Start talking', exact: true}).waitFor();
    assert.match(await page.locator('#status').textContent(), /two quiet minutes/);
    assert.ok(await page.evaluate(() => __test.commands.some(c => c.event?.type === 'end_audio')),
      'what was being said is flushed, as for a mute');
    // It falls asleep, and says so with its own chime rather than a mute's.
    assert.ok(await page.evaluate(() => __test.tones.includes('quiet')));
    assert.equal(await page.evaluate(() => __test.tones.includes('mute')), false);
    assert.equal(await page.evaluate(() => document.body.hasAttribute('data-sleep')), true);
    assert.equal(await page.locator('#stateLabel').textContent(), 'Sleeping');
    // Asleep is resting, not switched off: the orb keeps its light and the sky keeps moving, while
    // a quiet aurora breathes round the orb.
    await page.waitForTimeout(1500);
    const rest = await page.evaluate(() => ({
      orb: getComputedStyle(document.querySelector('.orb-stage')).filter,
      sky: getComputedStyle(document.querySelector('.space .far')).animationPlayState,
      aurora: getComputedStyle(document.querySelector('.orb-aurora')).animationName,
      glow: Number(getComputedStyle(document.querySelector('.orb-aurora')).opacity),
      rim: getComputedStyle(document.querySelector('.au-rim i')).animationPlayState,
    }));
    assert.deepEqual({...rest, glow: rest.glow > 0.4}, {orb: 'none', sky: 'running', aurora: 'rest-glow', glow: true, rim: 'running'});
    // Asked to stay on, it stays on. Turning the microphone back on wakes it first.
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    assert.equal(await page.evaluate(() => document.body.hasAttribute('data-sleep')), false);
    assert.doesNotMatch(await page.locator('#status').textContent(), /two quiet minutes/);
    assert.notEqual(await page.locator('#stateLabel').textContent(), 'Sleeping');
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByRole('button', {name: 'Never', exact: true}).click();
    assert.equal(await page.getByRole('button', {name: 'Never', exact: true}).getAttribute('aria-pressed'), 'true');
    await page.keyboard.press('Escape');
    await page.evaluate(() => __test.emit({type: 'quiet'}));
    await page.waitForTimeout(200);
    assert.equal(await page.getByRole('button', {name: 'Mute microphone'}).count(), 1, 'the microphone is still on');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('a starter is sent as if typed, and the starters step aside until the next fresh page', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const starters = page.getByRole('navigation', {name: 'Ways to start'});
    assert.equal(await starters.isVisible(), true);
    await starters.getByRole('button', {name: 'Help me unwind'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    assert.equal(await page.evaluate(() => __test.commands.find(c => c.event?.type === 'text').event.text), 'Help me unwind');
    assert.equal(await page.locator('#text').inputValue(), '');
    await starters.waitFor({state: 'hidden'});
    // A new conversation is a fresh page again.
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByRole('button', {name: 'Start new'}).click();
    await starters.waitFor({state: 'visible'});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the text box sends on Enter, takes a new line on Shift+Enter, and grows with what is typed', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const box = page.locator('#text');
    const send = page.getByRole('button', {name: 'Send message'});
    // Nothing to send, nothing to press.
    assert.equal(await send.isDisabled(), true);
    const oneLine = await box.evaluate(el => el.offsetHeight);
    await box.click();
    await page.keyboard.type('First line');
    await page.keyboard.press('Shift+Enter');
    await page.keyboard.type('second line');
    assert.equal(await send.isDisabled(), false);
    assert.equal(await box.inputValue(), 'First line\nsecond line');
    assert.ok(await box.evaluate(el => el.offsetHeight) > oneLine, 'two lines stand taller than one');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    assert.equal(await page.evaluate(() => __test.commands.find(c => c.event?.type === 'text').event.text), 'First line\nsecond line');
    assert.equal(await box.inputValue(), '');
    assert.equal(await box.evaluate(el => el.offsetHeight), oneLine, 'and it settles back once sent');
    assert.equal(await send.isDisabled(), true);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the page shows what Zen is doing: the aurora while it listens or answers, and each state named', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const state = () => page.evaluate(() => ({mic: document.body.dataset.mic, phase: document.body.dataset.phase,
      aurora: getComputedStyle(document.querySelector('.orb-aurora')).opacity !== '0'}));
    await page.emulateMedia({reducedMotion: 'reduce'});
    assert.deepEqual(await state(), {mic: 'off', phase: 'idle', aurora: false});
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await page.evaluate(() => __test.emit({type: 'state', generation: 0, phase: 'listening'}));
    assert.deepEqual(await state(), {mic: 'on', phase: 'listening', aurora: true});
    assert.equal(await page.locator('#stateLabel').textContent(), 'Listening');
    // With the microphone off, a typed question still lights it while Zen answers.
    await page.getByRole('button', {name: 'Mute microphone'}).click();
    await page.getByRole('button', {name: 'Start talking', exact: true}).waitFor();
    await page.evaluate(() => __test.emit({type: 'state', generation: 0, phase: 'speaking'}));
    assert.deepEqual(await state(), {mic: 'off', phase: 'speaking', aurora: true});
    assert.equal(await page.locator('#stateLabel').textContent(), 'Speaking');
    // Stop is in the row only while there is something to stop.
    assert.equal(await page.locator('#interrupt').isVisible(), true);
    await page.evaluate(() => __test.emit({type: 'state', generation: 0, phase: 'idle'}));
    assert.deepEqual(await state(), {mic: 'off', phase: 'idle', aurora: false});
    assert.equal(await page.locator('#interrupt').isVisible(), false);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('Zen\'s reply goes up a sentence at a time as it is said, and the board scrolls rather than drifts', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const board = await page.evaluate(async () => {
      const {Transcript} = await import('/transcript.mjs');
      const $ = (id) => document.getElementById(id);
      const transcript = new Transcript({list: document.createElement('div'), scroller: document.createElement('div'),
        caption: $('liveCaption'), speaker: $('captionSpeaker'), text: $('captionText'), frame: $('captionFrame')});
      const lines = () => [...$('captionText').querySelectorAll('.sentence')].map(e => (e.classList.contains('said') ? '~' : '') + e.textContent);
      const phrase = 'Dr. Rao said it first. Then it spread. Nobody checked.';
      const at = (words) => phrase.indexOf(words) / phrase.length * 3000;
      const seen = {};
      transcript.say(7, 1, phrase);
      // The first sentence goes up the moment the phrase is heard, and nothing else yet.
      seen.start = lines();
      transcript.reach(1, at('Then') - 300, 3000);
      seen.before = lines();
      transcript.reach(1, at('Then'), 3000);
      seen.second = lines();
      // Its length unknown, the phrase is timed from Zen's pace: 15 characters a second.
      transcript.reach(1, phrase.indexOf('Nobody') / 0.015, null);
      seen.third = lines();
      // The next phrase of the same reply goes up under it.
      transcript.say(7, 2, 'One more thing. And another.');
      seen.next = lines();
      // Cut off: what was said stays, and the rest never goes up.
      transcript.stop();
      transcript.reach(2, 60000, 3000);
      transcript.said(2);
      seen.stopped = lines();
      // What you say replaces the board, and the next reply starts a fresh one.
      transcript.show('You', 'Thanks.');
      seen.you = [$('captionSpeaker').textContent, $('captionText').textContent];
      transcript.say(8, 1, 'You are welcome.');
      seen.fresh = lines();
      const frame = $('captionFrame');
      seen.freshTop = frame.scrollTop;
      // Many sentences: the board reads down them like a prompter...
      for (let i = 0; i < 12; i++) transcript.say(9, i + 1, `Sentence number ${i} is here.`);
      await new Promise(resolve => setTimeout(resolve, 900)); // the glide settling
      seen.scrolls = frame.scrollHeight > frame.clientHeight && getComputedStyle(frame).overflowY === 'auto';
      // ...keeping the sentence being said in view, with what was just said above it, rather
      // than pinned to the bottom edge.
      const height = parseFloat(getComputedStyle($('captionText')).lineHeight);
      const said = $('captionText').lastElementChild.getClientRects()[0];
      seen.line = Math.round((said.top - frame.getBoundingClientRect().top) / height);
      // ...unless it has been scrolled up to read, and it never moves on its own.
      frame.scrollTop = 0;
      transcript.say(9, 13, 'One last one.');
      seen.held = frame.scrollTop;
      seen.transform = $('captionText').style.transform;
      return seen;
    });
    assert.deepEqual(board.start, ['Dr. Rao said it first.']);
    assert.deepEqual(board.before, ['Dr. Rao said it first.']);
    assert.deepEqual(board.second, ['~Dr. Rao said it first.', 'Then it spread.']);
    assert.deepEqual(board.third, ['~Dr. Rao said it first.', '~Then it spread.', 'Nobody checked.']);
    assert.deepEqual(board.next, ['~Dr. Rao said it first.', '~Then it spread.', '~Nobody checked.', 'One more thing.']);
    assert.deepEqual(board.stopped, board.next);
    assert.deepEqual(board.you, ['You', 'Thanks.']);
    assert.deepEqual(board.fresh, ['You are welcome.']);
    assert.equal(board.freshTop, 0, 'a reply starts at the top of the board');
    assert.equal(board.scrolls, true);
    assert.ok(board.line === 1 || board.line === 2, `the sentence being said is on line ${board.line + 1} of 4`);
    assert.equal(board.held, 0);
    assert.equal(board.transform, '');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the board keeps up with the voice itself, and keeps what you said in view while Zen thinks', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.evaluate(async () => {
      const {AudioPlayback} = await import('/audio.mjs');
      const begin = AudioPlayback.prototype.begin;
      AudioPlayback.prototype.begin = function (...args) {
        __test.playback = this;
        return begin.apply(this, args);
      };
    });
    await page.locator('#text').fill('Tell me about rain.');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.event?.type === 'text'));
    // Thinking: your words stay on the board, with the wave under them.
    const thinking = await page.evaluate(() => ({
      speaker: document.getElementById('captionSpeaker').textContent,
      text: document.getElementById('captionText').textContent,
      wave: getComputedStyle(document.querySelector('.caption-wave')).display,
      frame: document.getElementById('captionFrame').clientHeight,
    }));
    assert.deepEqual({...thinking, frame: thinking.frame > 100}, {speaker: 'You', text: 'Tell me about rain.', wave: 'flex', frame: true});
    // Three seconds of one phrase: its sentences go up as playback reaches them.
    const generation = await page.evaluate(() => __test.generation);
    await page.evaluate((g) => {
      __test.emit({type: 'state', generation: g, phase: 'preparing'});
      __test.emit({type: 'phrase_start', generation: g, phrase: 1, text: 'Rain is water. It falls from clouds when the drops grow heavy. That is all there is to it.'});
      for (let block = 1; block <= 3; block++) {
        const frame = new Uint8Array(25 + 48000);
        const view = new DataView(frame.buffer);
        view.setBigUint64(1, BigInt(g), true); view.setBigUint64(9, 1n, true); view.setBigUint64(17, BigInt(block), true);
        __test.frames.at(-1).onmessage(frame.buffer);
      }
      __test.emit({type: 'phrase_end', generation: g, phrase: 1, text: 'Rain is water. It falls from clouds when the drops grow heavy. That is all there is to it.', pause_ms: 0});
      __test.emit({type: 'state', generation: g, phase: 'speaking'});
    }, generation);
    const count = () => page.evaluate(() => document.querySelectorAll('#captionText .sentence').length);
    await page.waitForFunction(() => document.querySelectorAll('#captionText .sentence').length === 1);
    assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('.caption-wave')).display), 'none');
    assert.equal(await count(), 1, 'only the first sentence is up as the phrase begins');
    await page.waitForFunction(() => document.querySelectorAll('#captionText .sentence').length === 2, null, {timeout: 5000});
    await page.waitForFunction(() => document.querySelectorAll('#captionText .sentence').length === 3, null, {timeout: 5000});
    // Said in full, the reply is in the conversation as one entry.
    await page.waitForFunction(() => document.querySelectorAll('.history-entry[data-speaker="Zen"]').length === 1, null, {timeout: 5000});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('mute silences Zen - voice and cues - for this launch, and the reply carries on', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.evaluate(async () => {
      const {AudioPlayback, Soundscape} = await import('/audio.mjs');
      const begin = AudioPlayback.prototype.reset;
      AudioPlayback.prototype.reset = function (...args) {
        __test.playback = this;
        return begin.apply(this, args);
      };
      const play = Soundscape.prototype.play;
      __test.cues = [];
      Soundscape.prototype.play = function (kind, scale) {
        if (this.enabled) __test.cues.push(kind);
        return play.call(this, kind, scale);
      };
    });
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await page.getByRole('button', {name: 'Mute Zen', exact: true}).click();
    const muted = page.getByRole('button', {name: 'Unmute Zen', exact: true});
    assert.equal(await muted.getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#soundIcon').getAttribute('href'), '#i-sound-off');
    await page.waitForTimeout(150);
    assert.ok(await page.evaluate(() => __test.playback.volume.gain.value) < 0.01, 'the voice is silent');
    // No cue while muted.
    const before = await page.evaluate(() => __test.cues.length);
    await page.getByRole('button', {name: 'Mute microphone'}).click();
    await page.getByRole('button', {name: 'Start talking', exact: true}).waitFor();
    assert.equal(await page.evaluate(() => __test.cues.length), before);
    // Unmuted, the voice comes back, with a tick to say so.
    await muted.click();
    await page.waitForTimeout(150);
    assert.ok(await page.evaluate(() => __test.playback.volume.gain.value) > 0.99);
    assert.equal(await page.evaluate(() => __test.cues.at(-1)), 'send');
    // Not remembered: the next launch is never silent.
    await page.getByRole('button', {name: 'Mute Zen', exact: true}).click();
    await muted.waitFor();
    await page.reload();
    assert.equal(await page.getByRole('button', {name: 'Mute Zen', exact: true}).getAttribute('aria-pressed'), 'false');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the microphone reaches the recogniser band-limited: flat to 7 kHz, and nothing above 9 kHz folds back', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const measured = await page.evaluate(async () => {
      // The real capture worklet, fed a pure tone at a device rate, heard at 16 kHz.
      const level = async (frequency, rate) => {
        const context = new OfflineAudioContext(1, rate, rate);
        await context.audioWorklet.addModule('/capture.mjs');
        const tone = context.createOscillator();
        tone.frequency.value = frequency;
        const node = new AudioWorkletNode(context, 'zen-capture', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
        const frames = [];
        node.port.onmessage = ({data}) => { if (data instanceof ArrayBuffer) frames.push(new Int16Array(data)); };
        tone.connect(node).connect(context.destination);
        tone.start();
        await context.startRendering();
        await new Promise(resolve => setTimeout(resolve, 100));
        const samples = frames.flatMap(frame => Array.from(frame)).slice(1600);
        return {rms: Math.sqrt(samples.reduce((sum, v) => sum + (v / 32768) ** 2, 0) / samples.length), count: frames.length * 320};
      };
      const out = {};
      for (const rate of [48000, 44100]) {
        const reference = await level(1000, rate);
        const db = async (frequency) => 20 * Math.log10((await level(frequency, rate)).rms / reference.rms);
        out[rate] = {samples: reference.count, 7000: await db(7000), 9000: await db(9000), 10000: await db(10000), 12000: await db(12000)};
      }
      return out;
    });
    for (const [rate, m] of Object.entries(measured)) {
      // A second of sound is a second at 16 kHz, less the frame the filter's delay holds back.
      assert.ok(m.samples >= 15680 && m.samples <= 16000, `${rate}: ${m.samples} samples`);
      assert.ok(m[7000] > -0.5, `${rate}: 7 kHz at ${m[7000].toFixed(1)} dB`);
      for (const frequency of [9000, 10000, 12000])
        assert.ok(m[frequency] < -60, `${rate}: ${frequency} Hz folds back at ${m[frequency].toFixed(1)} dB`);
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the intro opens on the moon and the name with its sound, lands the page, and can be skipped', async () => {
  // Played through: the mark and the name over the page, which waits underneath, and the sound.
  let {page, context, errors} = await pageFor({intro: true, dark: true});
  try {
    const sounds = [];
    page.on('requestfinished', request => { if (request.url().endsWith('/startup.ogg')) sounds.push(request.url()); });
    await page.reload();
    await page.waitForTimeout(1200);
    const playing = await page.evaluate(() => ({
      intro: Number(getComputedStyle(document.querySelector('.intro')).opacity),
      name: document.querySelector('.intro-title').textContent,
      background: getComputedStyle(document.querySelector('.intro')).backgroundColor,
      card: Number(getComputedStyle(document.querySelector('.console')).opacity),
    }));
    assert.deepEqual(playing, {intro: 1, name: 'Zen', background: 'rgb(0, 0, 0)', card: 0});
    // The moon rises alone, then the name opens out from behind it onto the same line.
    const steps = await page.evaluate(() => ['.intro-mark', '.intro-word'].map(selector => {
      const style = getComputedStyle(document.querySelector(selector));
      return `${style.animationName} ${style.animationDuration} ${style.animationDelay}`;
    }));
    assert.deepEqual(steps, ['intro-rise 0.8s 0.2s', 'intro-open 0.8s 1.1s']);
    // Where it all stands at 2 s, the lockup complete: sized to the window, the moon as tall as the
    // capitals, and the pair on one line in the middle, the moon carried left of centre.
    const lockup = await page.evaluate(() => {
      for (const animation of document.querySelector('.intro').getAnimations({ subtree: true })) {
        animation.pause();
        animation.currentTime = 2000;
      }
      const box = (selector) => document.querySelector(selector).getBoundingClientRect();
      const mark = box('.intro-mark'), title = box('.intro-title'), row = box('.intro-lockup');
      return { mark: mark.height, rowCentre: row.left + row.width / 2, markCentre: mark.left + mark.width / 2,
        sameLine: Math.abs((mark.top + mark.bottom) / 2 - (title.top + title.bottom) / 2) < 4, width: innerWidth };
    });
    assert.ok(Math.abs(lockup.mark - 0.13 * 760) <= 1, `the moon's box is ${lockup.mark}px in a 760px window`);
    assert.ok(Math.abs(lockup.rowCentre - lockup.width / 2) <= 2, `the lockup is centred: ${lockup.rowCentre}`);
    assert.ok(lockup.markCentre < lockup.width / 2 - 60, `the moon moved left: ${lockup.markCentre}`);
    assert.equal(lockup.sameLine, true);
    await page.evaluate(() => document.querySelector('.intro').getAnimations({ subtree: true }).forEach(a => a.play()));
    assert.deepEqual(sounds.length, 1, 'the startup sound is fetched from the app itself');
    assert.equal(await page.evaluate(async () => (await new OfflineAudioContext(2, 48000, 48000)
      .decodeAudioData(await (await fetch('/startup.ogg')).arrayBuffer())).duration.toFixed(1)), '3.3');
    // It clears itself, and the page is there.
    await page.waitForFunction(() => !document.querySelector('.intro'), null, {timeout: 5000});
    await page.waitForFunction(() => getComputedStyle(document.querySelector('.console')).opacity === '1', null, {timeout: 5000});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
  // Skipped: a key goes straight to the page.
  ({page, context, errors} = await pageFor({intro: true}));
  try {
    await page.waitForTimeout(300);
    await page.keyboard.press('Shift');
    assert.equal(await page.evaluate(() => !!document.querySelector('.intro')), false);
    assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('.console')).opacity), '1');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
  // With reduced motion there is no intro and no sound: the page is simply there.
  ({page, context, errors} = await pageFor({intro: true, reducedMotion: true}));
  try {
    const sounds = [];
    page.on('request', request => { if (request.url().endsWith('/startup.ogg')) sounds.push(request.url()); });
    await page.reload();
    await page.waitForTimeout(500);
    assert.equal(await page.evaluate(() => !!document.querySelector('.intro')), false);
    assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('.console')).opacity), '1');
    assert.deepEqual(sounds, []);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('every sound cue plays', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const played = await page.evaluate(async () => {
      const {Soundscape} = await import('/audio.mjs');
      const context = new AudioContext();
      await context.resume();
      const sound = new Soundscape(context);
      const counts = {};
      for (const kind of ['connect', 'mic', 'ready', 'mute', 'quiet', 'stop', 'send', 'error']) {
        sound.dispose();
        sound.play(kind);
        counts[kind] = sound.nodes.size;
      }
      sound.play('no such cue');
      await context.close();
      return counts;
    });
    // Four partials to every note.
    assert.deepEqual(played, {connect: 4, mic: 8, ready: 8, mute: 8, quiet: 12, stop: 4, send: 4, error: 8});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the fonts are the app\'s own: loaded from it, nothing fetched from outside, nothing falling back', async () => {
  const {page, context, errors} = await pageFor();
  try {
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    await page.reload();
    const fonts = await page.evaluate(async () => {
      // English, and a name with a letter from the extended set.
      await Promise.all([document.fonts.load('600 26px Inter', 'Zen'), document.fonts.load('400 19px Newsreader', 'Zen'),
        document.fonts.load('400 19px Newsreader', 'Łukasz')]);
      return {
        loaded: [...document.fonts].filter(f => f.status === 'loaded').map(f => `${f.family.replace(/"/g, '')} ${f.unicodeRange.startsWith('U+0-FF') || f.unicodeRange.startsWith('U+0000-00FF') ? 'latin' : f.unicodeRange.startsWith('U+100') ? 'latin-ext' : 'all'}`).sort(),
        ready: document.fonts.check('400 19px Newsreader', 'Zen Łukasz') && document.fonts.check('600 26px Inter', 'Zen'),
      };
    });
    assert.deepEqual(fonts.loaded, ['Inter all', 'Newsreader latin', 'Newsreader latin-ext']);
    assert.equal(fonts.ready, true);
    assert.deepEqual(requests.filter(url => !url.startsWith(origin) && !url.startsWith('data:')), [], 'everything comes from the app itself');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the bar carries the time and the developer, and the sky is drawn', async () => {
  const {page, context, errors} = await pageFor();
  try {
    assert.match(await page.locator('#clock').textContent(), /\d{1,2}:\d{2}\s?(AM|PM)$/);
    assert.equal(await page.locator('.brand-by').textContent(), 'by AryThakar');
    // Each layer of stars is twice the window wide, so it can pan without a seam, and is drawn at
    // no more than one and a half times the screen's own pixels.
    const layers = await page.evaluate(() => [...document.querySelectorAll('.space canvas')].map(c =>
      ({wide: Math.round(c.getBoundingClientRect().width / innerWidth), scale: c.width / c.getBoundingClientRect().width})));
    assert.equal(layers.length, 3);
    for (const {wide, scale} of layers) {
      assert.equal(wide, 2);
      assert.ok(scale >= 0.99 && scale <= 1.5, `a layer is drawn at ${scale}x`);
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('settings are kept between launches, and the devices are not', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByLabel('Instructions for this conversation', {exact: true}).fill('Keep answers short.');
    await page.getByRole('button', {name: 'Dark', exact: true}).click();
    await page.getByRole('button', {name: 'Never', exact: true}).click();
    await page.keyboard.press('Escape');
    await page.reload();
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.getByLabel('Instructions for this conversation', {exact: true}).inputValue(), 'Keep answers short.');
    assert.equal(await page.getByRole('button', {name: 'Dark', exact: true}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByRole('button', {name: 'Never', exact: true}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#inputPicker').textContent(), 'System default');
    // The kept instructions are what the next session is started with.
    await page.keyboard.press('Escape');
    await page.getByLabel('Message Zen', {exact: true}).fill('Hello');
    await page.getByRole('button', {name: 'Send message'}).click();
    await page.waitForFunction(() => __test.commands.some(c => c.command === 'zen_attach'));
    assert.equal(await page.evaluate(() => __test.commands.find(c => c.command === 'zen_attach').args.systemPrompt), 'Keep answers short.');
    // Emptied, the instructions are Zen's own again, and stay that way.
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByLabel('Instructions for this conversation', {exact: true}).fill('');
    await page.reload();
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.getByLabel('Instructions for this conversation', {exact: true}).inputValue(), '');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('before Windows will name any device, the pickers say how to get there', async () => {
  const {page, context, errors} = await pageFor();
  try {
    // What Windows reports before the microphone has been allowed: devices, but no names.
    await plug(page, [
      {deviceId: '', kind: 'audioinput', label: '', groupId: ''},
      {deviceId: '', kind: 'audiooutput', label: '', groupId: ''},
    ]);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('inputPicker').getAttribute('aria-disabled') === 'true');
    // Marked disabled for assistive technology, but still there to click: that is how it says why.
    await page.locator('#inputPicker').click({force: true});
    const note = page.locator('#audioDialog .menu.note');
    assert.equal(await note.count(), 1);
    assert.match(await note.textContent(), /once the microphone is on/);
    assert.equal(await page.locator('.menu button').count(), 0, 'nothing to choose, so nothing offered');
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.menu').count(), 0);
    // Once names arrive, the picker is a picker again.
    await plug(page, [LAPTOP, USB]);
    await page.waitForFunction(() => document.getElementById('inputPicker').getAttribute('aria-disabled') === 'false');
    await page.locator('#inputPicker').click();
    assert.equal(await page.locator('.menu button').count(), 3);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the end of the greeting is cued, to anyone who can talk', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await page.evaluate(async () => {
      const {Soundscape} = await import('/audio.mjs');
      __test.tones = [];
      Soundscape.prototype.play = function (kind) { __test.tones.push(kind); };
    });
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    const readies = () => page.evaluate(() => __test.tones.filter(t => t === 'ready').length);
    assert.equal(await readies(), 0, 'nothing is cued while the greeting is still to come');
    await page.evaluate(() => __test.emit({type: 'greeted'}));
    assert.equal(await readies(), 1);
    // With the microphone off there is nobody to cue.
    await page.getByRole('button', {name: 'Mute microphone'}).click();
    await page.getByRole('button', {name: 'Start talking', exact: true}).waitFor();
    await page.evaluate(() => __test.emit({type: 'greeted'}));
    assert.equal(await readies(), 1);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

/// Counts microphone starts, and lets a test say what the next started track is called.
async function watchMicrophone(page, label = null) {
  await page.evaluate(async (label) => {
    const {Microphone} = await import('/audio.mjs');
    const start = Microphone.prototype.start;
    __test.micStarts = [];
    Microphone.prototype.start = async function (deviceId = '') {
      __test.micStarts.push(deviceId);
      // The browser's fake devices cannot open the made-up ids the tests plug in, so the real
      // fake input stands in for whichever one was asked for.
      await start.call(this, '');
      if (label) this.label = label;
    };
  }, label);
}

/// Chooses a device the way a person does: the row's button, then the item in the menu.
async function choose(page, row, name) {
  await page.locator(`#${row}Picker`).click();
  await page.locator('.menu button', {hasText: name}).first().click();
}

/// Replaces the device list the page sees, then tells it the devices changed.
async function plug(page, devices) {
  await page.evaluate((devices) => {
    navigator.mediaDevices.enumerateDevices = async () => devices;
    navigator.mediaDevices.dispatchEvent(new Event('devicechange'));
  }, devices);
}

const LAPTOP = {deviceId: 'laptop-mic', kind: 'audioinput', label: 'Laptop Microphone', groupId: 'laptop'};
const USB = {deviceId: 'usb-mic', kind: 'audioinput', label: 'USB Microphone', groupId: 'usb'};

test('the microphone is chosen in settings, switches live, and follows what is plugged in', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await watchMicrophone(page);
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await plug(page, [LAPTOP]);
    // Plugging something in only adds it to the list; the microphone in use does not move.
    await plug(page, [LAPTOP, USB]);
    await page.waitForFunction(() => document.querySelectorAll('#inputDevice option').length === 3);
    assert.deepEqual(await page.evaluate(() => __test.micStarts), ['']);
    // Choosing it in Settings moves capture straight away, with what was said kept.
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await choose(page, 'input', 'USB Microphone');
    await page.waitForFunction(() => __test.micStarts.length === 2);
    assert.deepEqual(await page.evaluate(() => __test.micStarts), ['', 'usb-mic']);
    assert.ok(await page.evaluate(() => __test.commands.some(c => c.event?.type === 'end_audio')),
      'what was said before the switch is flushed, not dropped');
    // Unplugging the chosen microphone falls back to the default instead of ending the turn.
    await plug(page, [LAPTOP]);
    await page.waitForFunction(() => /switched to the system default/.test(document.querySelector('#status').textContent));
    assert.deepEqual(await page.evaluate(() => __test.micStarts), ['', 'usb-mic', '']);
    assert.equal(await page.locator('#inputDevice').inputValue(), '');
    assert.equal(await page.locator('#inputDevice option[value="usb-mic"]').count(), 0);
    assert.equal(await page.getByRole('button', {name: 'Mute microphone'}).count(), 1,
      'capture is still running');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the device lists name real devices, and both are there before a session', async () => {
  const {page, context, errors} = await pageFor();
  try {
    // Windows' own list: the same array microphone three times, a driver name nobody reads,
    // and a device Steam installed.
    await plug(page, [
      {deviceId: 'default', kind: 'audioinput', label: 'Default - Microphone Array (2- Intel(R) Smart Sound Technology for Digital Microphones)', groupId: 'array'},
      {deviceId: 'communications', kind: 'audioinput', label: 'Communications - Microphone Array (2- Intel(R) Smart Sound Technology for Digital Microphones)', groupId: 'array'},
      {deviceId: 'array-mic', kind: 'audioinput', label: 'Microphone Array (2- Intel(R) Smart Sound Technology for Digital Microphones)', groupId: 'array'},
      {deviceId: 'steam-mic', kind: 'audioinput', label: 'Microphone (Steam Streaming Microphone)', groupId: 'steam'},
      {deviceId: 'buds', kind: 'audioinput', label: 'Headset (Galaxy Buds Hands-Free AG Audio)', groupId: 'buds'},
      {deviceId: 'speakers', kind: 'audiooutput', label: 'Speakers (Realtek(R) Audio)', groupId: 'realtek'},
    ]);
    const options = (id) => page.evaluate((id) => [...document.querySelectorAll(`#${id} option`)].map(o => o.textContent), id);
    await page.waitForFunction(() => document.querySelectorAll('#inputDevice option').length > 1);
    assert.deepEqual(await options('inputDevice'), ['System default', 'Microphone Array', 'Headset (Galaxy Buds Hands-Free AG Audio)']);
    assert.deepEqual(await options('outputDevice'), ['System default', 'Speakers (Realtek(R) Audio)']);
    // Both rows are offered before anyone has started talking.
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    assert.equal(await page.locator('#outputRow').isVisible(), true);
    assert.equal(await page.locator('#inputDevice').isVisible(), true);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('System default follows the device Windows moves to', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await watchMicrophone(page);
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    const asDefault = (device) => ({...device, deviceId: 'default', label: `Default - ${device.label}`});
    await plug(page, [asDefault(LAPTOP), LAPTOP]);
    await page.waitForFunction(() => document.querySelectorAll('#inputDevice option').length === 2);
    const before = await page.evaluate(() => __test.micStarts.length);
    // Earbuds connected: Windows moves its default, and so does Zen.
    await plug(page, [asDefault(USB), LAPTOP, USB]);
    await page.waitForFunction((n) => __test.micStarts.length === n + 1, before);
    assert.equal(await page.evaluate(() => __test.micStarts.at(-1)), '', 'capture reopens on the default');
    assert.equal(await page.locator('#inputDevice').inputValue(), '');
    // A named choice is the person's, and a change of default does not overrule it.
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await choose(page, 'input', 'USB Microphone');
    await page.waitForFunction(() => __test.micStarts.at(-1) === 'usb-mic');
    const named = await page.evaluate(() => __test.micStarts.length);
    await plug(page, [asDefault(LAPTOP), LAPTOP, USB]);
    await page.waitForTimeout(300);
    assert.equal(await page.evaluate(() => __test.micStarts.length), named, 'the named microphone stays');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the device menu opens inside the settings sheet, and a choice sticks', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await watchMicrophone(page);
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await plug(page, [LAPTOP, USB]);
    await page.waitForFunction(() => document.querySelectorAll('#inputDevice option').length === 3);
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    const picker = page.locator('#inputPicker');
    assert.equal(await picker.textContent(), 'System default');
    await picker.click();
    // Inside the dialog: a menu attached to the page would be inert behind the modal.
    assert.equal(await page.locator('#audioDialog .menu').count(), 1);
    assert.deepEqual(await page.locator('.menu button').allTextContents(),
      ['System default', 'Laptop Microphone', 'USB Microphone']);
    assert.equal(await page.locator('.menu button[aria-selected="true"]').textContent(), 'System default',
      'the current device is ticked');
    // Escape closes the menu and leaves the sheet open.
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.menu').count(), 0);
    assert.equal(await page.locator('#audioDialog').evaluate(el => el.open), true);
    await choose(page, 'input', 'USB Microphone');
    assert.equal(await page.locator('.menu').count(), 0);
    assert.equal(await picker.textContent(), 'USB Microphone');
    await page.waitForFunction(() => __test.micStarts.at(-1) === 'usb-mic', null, {timeout: 5000});
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('a Bluetooth hands-free microphone is called out', async () => {
  const {page, context, errors} = await pageFor();
  try {
    await watchMicrophone(page, 'Headset (WH-1000XM4 Hands-Free AG Audio)');
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor();
    await page.waitForFunction(() => /Bluetooth headset/.test(document.querySelector('#status').textContent));
    assert.match(await page.locator('#audioNote').textContent(), /call quality/);
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
