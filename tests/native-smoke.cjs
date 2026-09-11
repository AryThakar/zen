// Opt-in native integration check. Requires a release build and installed model assets.
// CDP and fake capture are enabled only in this test's child process environment.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const net = require('node:net');
const fs = require('node:fs/promises');
const path = require('node:path');

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

test('real WebView2 completes spoken turns, releases workers, and starts a fresh session', {timeout: 240000}, async () => {
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
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
    const context = browser.contexts()[0];
    page = context.pages()[0] || await context.waitForEvent('page');
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForSelector('#text');
    await page.getByRole('button', {name: 'Sound cues', exact: true}).click();
    for (const message of ['Say hello in one short sentence.', 'Say ready in one short sentence.']) {
      await page.locator('#text').fill(message);
      await page.getByRole('button', {name: 'Send message', exact: true}).click();
      await page.waitForFunction(() => document.querySelector('.transcript-entry[data-speaker="Zen"]'), {timeout: 90000});
      await page.waitForFunction(() => ['idle', 'listening'].includes(document.querySelector('#orbState').dataset.phase), {timeout: 30000});
      const reply = await page.locator('.transcript-entry[data-speaker="Zen"]').innerText();
      assert(reply.length > 4, 'A completed spoken reply should appear in history.');
      assert.equal(await page.locator('#status.error').count(), 0);
      await page.screenshot({path: path.join(target, 'native-session.png')});
      await page.getByRole('button', {name: 'End session', exact: true}).click();
      await page.waitForFunction(() => document.querySelector('#connectionLabel').textContent === 'On this device', {timeout: 90000});
      assert.equal(await page.locator('.transcript-entry').count(), 0);
      const alive = await fetch('http://127.0.0.1:8740/health', {signal: AbortSignal.timeout(1000)}).then(() => true, () => false);
      assert.equal(alive, false, 'Disconnect must release the model server before returning.');
    }
    assert.deepEqual(errors, []);
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
