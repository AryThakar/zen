// Opt-in native integration check. Requires a release build and installed model assets.
// CDP and fake capture are enabled only in this test's child process environment.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('node:child_process');
const net = require('node:net');
const fs = require('node:fs/promises');
const path = require('node:path');

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
// Every process on the machine, as {pid, parent, name}.
function processes() {
  const listing = execFileSync('powershell', ['-NoProfile', '-Command',
    'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId) $($_.ParentProcessId) $($_.Name)" }'], {encoding: 'utf8'});
  return listing.trim().split(/\r?\n/).map(line => {
    const [pid, parent, ...name] = line.trim().split(' ');
    return {pid: Number(pid), parent: Number(parent), name: name.join(' ')};
  });
}
// Everything a process started, and everything those started.
function descendants(root) {
  const all = processes(), found = [], queue = [root];
  while (queue.length) {
    const pid = queue.shift();
    for (const process of all) if (process.parent === pid && !found.some(known => known.pid === process.pid)) {
      found.push(process);
      queue.push(process.pid);
    }
  }
  return found;
}
async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

test('real WebView2 completes spoken turns, releases workers, starts a fresh session, and quits', {timeout: 240000}, async () => {
  const target = path.resolve(__dirname, '../target');
  const executable = path.join(target, 'release/zen.exe');
  await fs.access(executable);
  // Refuse to disturb an engine already being used by a person or another test.
  const occupied = await fetch('http://127.0.0.1:8740/health', {signal: AbortSignal.timeout(1500)}).then(() => true, () => false);
  assert.equal(occupied, false, 'End the current Zen session before running native checks.');
  const port = await freePort();
  const profile = await fs.mkdtemp(path.join(target, 'native-profile-'));
  const child = spawn(executable, ['--run-for-seconds', '220'], {windowsHide: true,
    env: {...process.env, WEBVIEW2_USER_DATA_FOLDER: profile,
      WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-address=127.0.0.1 --remote-debugging-port=${port} --use-fake-device-for-media-stream --use-fake-ui-for-media-stream --mute-audio`},
    stdio: 'ignore'});
  let browser, page, launchError;
  child.on('error', error => { launchError = error; });
  try {
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      if (launchError) throw launchError;
      if (child.exitCode !== null) throw new Error(`Zen exited before opening its window: ${child.exitCode}`);
      if (await fetch(`http://127.0.0.1:${port}/json/version`, {signal: AbortSignal.timeout(1000)}).then(r => r.ok, () => false)) break;
      await pause(250);
    }
    // WebView2 owns this context. Do not apply desktop Chrome download/focus defaults.
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { noDefaults: true });
    const context = browser.contexts()[0];
    page = context.pages()[0] || await context.waitForEvent('page');
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForSelector('#text');
    // The microphone path end to end: the capture worklet, and the filter module it imports, load
    // from the app's own origin and frames flow out of it.
    await context.addInitScript(() => {
      const Real = window.AudioWorkletNode;
      window.__captured = 0;
      window.AudioWorkletNode = class extends Real {
        constructor(audio, name, options) {
          super(audio, name, options);
          if (name === 'zen-capture')
            this.port.addEventListener('message', ({data}) => { if (data instanceof ArrayBuffer) window.__captured++; });
        }
      };
    });
    await page.reload();
    await page.waitForSelector('#text');
    // The intro plays from the app's own files - its sound included - and clears itself.
    await page.waitForFunction(() => !document.querySelector('.intro'), null, {timeout: 10000});
    assert.ok(await page.evaluate(() => performance.getEntriesByType('resource').some(entry => entry.name.endsWith('/startup.ogg'))),
      'the startup sound is bundled in the app');
    await page.getByRole('button', {name: 'Start talking', exact: true}).click();
    await page.getByRole('button', {name: 'Mute microphone'}).waitFor({timeout: 120000});
    await page.waitForFunction(() => window.__captured >= 50, null, {timeout: 10000});
    await page.getByRole('button', {name: 'Mute microphone'}).click();
    await page.getByRole('button', {name: 'Start talking', exact: true}).waitFor();
    async function say(message) {
      const before = await page.locator('.history-entry[data-speaker="Zen"]').count();
      await page.locator('#text').fill(message);
      await page.getByRole('button', {name: 'Send message', exact: true}).click();
      await page.waitForFunction(count =>
        document.querySelectorAll('.history-entry[data-speaker="Zen"]').length > count &&
        ['idle', 'listening'].includes(document.querySelector('#orbState').dataset.phase),
        before, {timeout: 90000});
      const reply = await page.locator('.history-entry[data-speaker="Zen"] p').last().textContent();
      assert(reply.length > 4, 'A completed spoken reply should appear in history.');
      // Said in full, the whole reply is up on the board, a sentence at a time.
      const board = await page.locator('#captionText .sentence').allTextContents();
      assert(board.length >= 1, 'The reply goes up on the board as sentences.');
      assert.equal(board.join(' ').replace(/s+/g, ' '), reply.replace(/s+/g, ' ').trim());
      assert.equal(await page.locator('#status.error').count(), 0);
      return reply;
    }
    for (const message of ['My spare keys are in the blue drawer. Remember this and acknowledge briefly.', 'Say ready in one short sentence.']) {
      await say(message);
      if (message.includes('blue drawer')) {
        const recalled = await say('Which drawer did I say holds my spare keys?');
        assert.match(recalled, /blue/i, 'Recall must use earlier turns in this same open session.');
      }
      await page.screenshot({path: path.join(target, 'native-session.png')});
      await page.getByRole('button', {name: 'End session', exact: true}).click();
      await page.waitForFunction(() => document.body.dataset.active === 'false' && document.querySelector('#micLabel').textContent === 'Start talking', null, {timeout: 90000});
      assert.equal(await page.locator('.history-entry').count(), 0);
      const alive = await fetch('http://127.0.0.1:8740/health', {signal: AbortSignal.timeout(1000)}).then(() => true, () => false);
      assert.equal(alive, false, 'Disconnect must release the model server before returning.');
    }
    assert.deepEqual(errors, []);
    // Quit from Settings with a session running. Ending a session unloads the models; quitting
    // ends the app - the window, the tray, the engine and every process it started - so there
    // is nothing left to end from Task Manager.
    await say('Say ready in one short sentence.');
    const started = descendants(child.pid);
    for (const name of ['llama-server.exe', 'zen.exe', 'msedgewebview2.exe'])
      assert(started.some(process => process.name === name), `the session runs ${name} under Zen`);
    const exited = new Promise(resolve => child.exitCode !== null ? resolve() : child.once('exit', resolve));
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByRole('button', {name: /^Quit Zen/}).click();
    const asked = Date.now();
    await page.getByRole('button', {name: 'Quit', exact: true}).click({noWaitAfter: true}).catch(() => {});
    await Promise.race([exited, pause(30000)]);
    assert.notEqual(child.exitCode, null, 'Quit ends the Zen process');
    console.log(`quit: Zen exited ${Date.now() - asked} ms after Quit`);
    const settled = Date.now() + 10000;
    let left = [];
    do {
      const running = processes();
      left = started.filter(process => running.some(now => now.pid === process.pid && now.name === process.name));
      if (left.length) await pause(250);
    } while (left.length && Date.now() < settled);
    assert.deepEqual(left, [], 'nothing Zen started outlives it');
    const serving = await fetch('http://127.0.0.1:8740/health', {signal: AbortSignal.timeout(1000)}).then(() => true, () => false);
    assert.equal(serving, false, 'the model server is gone');
  } finally {
    if (page && !page.isClosed()) {
      await page.evaluate(() => window.__TAURI__.core.invoke('zen_disconnect')).catch(() => {});
    }
    await browser?.close().catch(() => {});
    child.kill();
    await pause(1000);
    assert(profile.startsWith(target + path.sep), 'Temporary profile must stay inside the test directory.');
    await fs.rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 300}).catch(() => {});
  }
});
