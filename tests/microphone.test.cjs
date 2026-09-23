const test = require("node:test");
const assert = require("node:assert/strict");

// Permission grants are asynchronous in Chrome and can arrive after a disconnect.
test("a delayed microphone permission grant cannot revive a cancelled session", async () => {
  const { Microphone } = await import("../src/client/audio.mjs");
  const previousNavigator = Object.getOwnPropertyDescriptor(
    globalThis,
    "navigator",
  );
  const previousSecure = globalThis.isSecureContext;
  let grant,
    requested,
    stopped = 0;
  const requestedPromise = new Promise((resolve) => {
    requested = resolve;
  });
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getSupportedConstraints: () => ({
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }),
        getUserMedia: (constraints) => {
          assert.equal(constraints.audio.echoCancellation.ideal, true);
          requested();
          return new Promise((resolve) => {
            grant = resolve;
          });
        },
      },
    },
  });
  globalThis.isSecureContext = true;
  try {
    const context = {
      resume: async () => {},
      audioWorklet: { addModule: async () => {} },
    };
    const callbacks = {
      onFrame() {
        throw new Error("Cancelled capture sent audio");
      },
      onLevel() {},
      onState() {},
      onError() {},
    };
    const mic = new Microphone(context, callbacks);
    const start = mic.start();
    await requestedPromise;
    await mic.stop(false);
    grant({
      getTracks: () => [
        {
          stop() {
            stopped++;
          },
        },
      ],
    });
    await start;
    assert.equal(stopped, 1);
    assert.equal(mic.active, false);
    assert.equal(mic.pending, false);
    assert.equal(mic.stream, null);
  } finally {
    if (previousNavigator)
      Object.defineProperty(globalThis, "navigator", previousNavigator);
    else delete globalThis.navigator;
    if (previousSecure === undefined) delete globalThis.isSecureContext;
    else globalThis.isSecureContext = previousSecure;
  }
});

test("an interrupted worklet download can be retried", async () => {
  const { Microphone } = await import("../src/client/audio.mjs");
  const previousNavigator = Object.getOwnPropertyDescriptor(
    globalThis,
    "navigator",
  );
  const previousSecure = globalThis.isSecureContext;
  let downloads = 0,
    permissions = 0;
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getSupportedConstraints: () => ({}),
        getUserMedia: async () => {
          permissions++;
          throw new Error("Permission test");
        },
      },
    },
  });
  globalThis.isSecureContext = true;
  try {
    const context = {
      resume: async () => {},
      audioWorklet: {
        addModule: async () => {
          if (++downloads === 1) throw new Error("Network interrupted");
        },
      },
    };
    const mic = new Microphone(context, {
      onFrame() {},
      onLevel() {},
      onState() {},
      onError() {},
    });
    await assert.rejects(mic.start(), /Network interrupted/);
    await assert.rejects(mic.start(), /Permission test/);
    assert.equal(downloads, 2);
    assert.equal(permissions, 1);
    assert.equal(mic.pending, false);
  } finally {
    if (previousNavigator)
      Object.defineProperty(globalThis, "navigator", previousNavigator);
    else delete globalThis.navigator;
    if (previousSecure === undefined) delete globalThis.isSecureContext;
    else globalThis.isSecureContext = previousSecure;
  }
});
