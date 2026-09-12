import { AudioPlayback, Microphone, Soundscape } from "./audio.mjs";
import { OrderedInput } from "./transport.mjs";
import { LiquidOrb } from "./orb.mjs";
import { Transcript } from "./transcript.mjs";

const $ = (id) => document.getElementById(id);
const bytes = (text) => new TextEncoder().encode(text).length;

/// Appearance and listening pause are conveniences rather than conversation state, so they
/// are the one thing Zen remembers between launches. Nothing said, heard or typed is stored;
/// storage can also be unavailable outright, which must not stop the page loading.
const remembered = (key, fallback) => {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};
const remember = (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* A private window or blocked site data. The choice still applies to this session. */
  }
};

const themes = ["system", "light", "dark"];
/// "system" stamps no attribute, so the stylesheet falls through to `prefers-color-scheme`.
function applyTheme(choice) {
  const theme = themes.includes(choice) ? choice : "system";
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  for (const button of document.querySelectorAll("[data-theme-choice]"))
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.themeChoice === theme),
    );
  return theme;
}

/// Only used when the pause is set by hand. Left to itself the engine picks its own and
/// moves it as it learns, which is what "Automatic" means.
const ENDPOINT_DEFAULT = 900;
function endpointOf(value) {
  const ms = Math.round(Number(value) / 50) * 50;
  return Number.isFinite(ms) ? Math.min(2000, Math.max(450, ms)) : ENDPOINT_DEFAULT;
}

/// The page and the engine are the same process. Commands go out over Tauri's IPC
/// and ordered state and audio frames return on one channel.

/// Bytes of little-endian header the engine puts ahead of every audio block,
/// after its kind byte.
const AUDIO_HEADER = 24;

/// Frame kinds, the same in both directions: raw audio, or a UTF-8 JSON message.
const AUDIO_FRAME = 0;
const CONTROL_FRAME = 1;

const activePhases = new Set([
  "transcribing",
  "thinking",
  "preparing",
  "speaking",
]);
const labels = {
  idle: "Ready when you are",
  listening: "Listening",
  transcribing: "Understanding you",
  thinking: "Thinking",
  preparing: "Preparing a reply",
  speaking: "Speaking",
  loading: "Loading local models",
};
const errors = {
  invalid_prompt: "Please keep your instructions within 8,192 UTF-8 bytes.",
  invalid_input: "I lost part of that. Please say it again.",
  backend_unavailable:
    "The local engine stopped. Check the model files, available graphics memory, and whether another Zen engine is running. Then try again.",
  model_failed:
    "I couldn’t finish that thought. Try a shorter message or instructions.",
  recognition_failed: "I couldn’t make out that last part. Please try again.",
  synthesis_failed: "My voice paused. Please try that once more.",
  turn_timeout: "That took longer than expected. Let’s try again.",
  playback_stalled:
    "Playback paused. Tap Start talking or send a message to continue.",
  audio_stalled: "Microphone audio paused. Tap Start talking to reconnect it.",
  transcript_too_long:
    "That was too long to work with in one go. Try saying it in shorter pieces.",
  synthesis_backpressure:
    "My voice fell behind that reply. Try a shorter message or shorter instructions.",
  invalid_synthesis_audio: "My voice produced something unusable. Please try again.",
};
const fatalErrors = new Set(["invalid_prompt", "backend_unavailable"]);

export class VoiceClient {
  constructor(core = globalThis.window?.__TAURI__?.core) {
    this.invoke = core.invoke;
    this.Channel = core.Channel;
    /// Stamped on every frame so a reloaded page cannot have the previous page's
    /// audio credited to the new session. Null means nothing is attached.
    this.revision = null;
    this.attached = false;
    /// One ordered queue to the engine, drained by a single in-flight call.
    this.input = null;
    this.ready = false;
    this.generation = 0;
    this.phase = "idle";
    this.context = null;
    this.playback = null;
    this.mic = null;
    this.sound = null;
    this.inputLevel = 0;
    this.soundEnabled = true;
    this.deviceId = "";
    this.theme = applyTheme(remembered("zen.theme", "system"));
    this.endpointMs = endpointOf(remembered("zen.endpoint", ENDPOINT_DEFAULT));
    this.pauseAuto = remembered("zen.pause", "auto") !== "manual";
    /// What the engine reports it is actually waiting, which in automatic mode moves.
    this.endpointActual = null;
    this.micChanging = false;
    this.micOperation = 0;
    this.lifecycle = 0;
    this.transcript = new Transcript(
      $("transcript"),
      $("liveCaption"),
      $("captionSpeaker"),
      $("captionText"),
    );
    this.orb = new LiquidOrb($("orb"), () => ({
      input: this.mic?.active ? this.inputLevel : 0,
      output: this.playback?.level() || 0,
    }));
    this.bind();
    this.controls();
  }

  // Sound cues are optional and independent of the assistant's voice.
  feedback(event) {
    if (!this.soundEnabled) return;
    const tone = ["connect", "ready", "mic", "mute", "stop", "error"].includes(event) ? event : null;
    if (tone) this.sound?.play(tone);
  }

  /// Capture that could not keep up was discarded. Say so rather than letting recognition
  /// quietly get worse: the person can hear themselves fine and has no other way to know.
  /// Throttled, because shedding arrives in bursts and a status line that rewrites every
  /// frame is its own kind of noise.
  reportShedding(total) {
    const now = performance.now();
    if (this.shedAt && now - this.shedAt < 4000) return;
    this.shedAt = now;
    this.status(
      `Your microphone audio is arriving faster than Zen can take it, so about ${
        Math.round((total * 20) / 100) / 10
      } seconds of it was dropped. Recognition may suffer. Close other heavy apps, or end and restart the session.`,
      true,
    );
  }

  /// Applied live, so a change never costs a session restart. `null` hands the decision
  /// back to the engine, which learns it from how this person actually pauses.
  sendTuning() {
    if (this.ready)
      this.send({
        type: "tuning",
        endpoint_ms: this.pauseAuto ? null : this.endpointMs,
      });
  }

  showPauseMode() {
    for (const button of document.querySelectorAll("[data-pause-mode]"))
      button.setAttribute(
        "aria-pressed",
        String((button.dataset.pauseMode === "auto") === this.pauseAuto),
      );
    $("endpointField").hidden = this.pauseAuto;
    this.showEndpoint();
  }

  showEndpoint() {
    const seconds = (ms) => `${(ms / 1000).toFixed(2)} seconds`;
    if (!this.pauseAuto) {
      $("endpointNote").textContent = `Waiting ${seconds(this.endpointMs)} of silence.`;
      return;
    }
    // Say what it settled on. A number that moves on its own and is never shown is worse
    // than no number at all.
    $("endpointNote").textContent =
      this.endpointActual === null
        ? "Zen will settle on a pause as you talk."
        : `Currently waiting ${seconds(this.endpointActual)} of silence.`;
  }

  status(message, error = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", error);
  }

  /// Queue one frame for the engine. Everything the page sends — capture and control
  /// alike — goes through here, in order, with a single call in flight.
  ///
  /// IPC calls are independent promises, so anything sent concurrently can be delivered
  /// in any order. The turn logic cannot survive that: `end_audio` overtaking the frames
  /// it follows truncates the user's last word, and an acknowledgement arriving before
  /// the block it acknowledges fails the turn. A socket ordered these for us; here the
  /// ordering has to be maintained explicitly.
  enqueue(kind, payload) {
    return this.input?.send(kind, payload) ?? false;
  }

  send(event) {
    return this.enqueue(
      CONTROL_FRAME,
      new TextEncoder().encode(JSON.stringify(event)),
    );
  }

  controls() {
    const connected = this.attached;
    $("disconnect").hidden = !connected && !this.connecting && !this.stopping;
    $("disconnect").disabled = !!this.stopping;
    document.body.dataset.active = String(connected);
    // There is one page and one conversation. Speaking or typing is what starts a
    // session, so the controls stay live before one exists rather than waiting
    // behind a separate entry step.
    const busy = this.connecting || this.stopping || this.micChanging || !!this.mic?.pending;
    for (const id of ["text", "sendText"]) $(id).disabled = busy;
    $("mic").disabled = busy;
    $("startFresh").disabled = !this.ready || !!this.stopping;
    $("interrupt").disabled =
      !this.ready ||
      (!activePhases.has(this.phase) && !this.playback?.sources.size);
    const micOn = !!this.mic?.active;
    const displayPhase =
      this.phase === "listening" && !micOn ? "idle" : this.phase;
    $("stateLabel").textContent =
      displayPhase === "idle" && micOn
        ? "Ready to listen"
        : labels[displayPhase] || "Here with you";
    $("orbState").dataset.phase = displayPhase;
    $("mic").setAttribute("aria-pressed", String(micOn));
    $("micIcon").setAttribute("href", micOn ? "#i-muted" : "#i-mic");
    $("micLabel").textContent = this.stopping ? "Ending session…" : this.connecting ? "Loading models…" : this.mic?.pending
      ? "Allow microphone…"
      : micOn
        ? "Mute microphone"
        : "Start talking";
    $("micHint").textContent = micOn
      ? "Just speak naturally. You can interrupt me anytime."
      : "Your microphone is off.";
    $("sounds").setAttribute("aria-pressed", String(this.soundEnabled));
    $("connectionBadge").classList.toggle("connected", this.ready);
    $("connectionLabel").textContent = connected
      ? this.ready
        ? "Session ready"
        : "Starting session"
      : this.stopping ? "Stopping engine" : this.connecting ? "Starting session" : "On this device";
    $("sessionLabel").textContent = connected
      ? "Your conversation"
      : "Private by nature";
    $("inputDevice").disabled = this.micChanging || !!this.mic?.pending;
  }

  /// The microphone raises its onset threshold while Zen can be heard, so that his own voice
  /// coming back through the speakers is not mistaken for someone interrupting him. Driven by
  /// audio actually scheduled as well as by the phase, because the opening greeting is spoken
  /// without the turn machine and never reaches the speaking phase at all.
  micPlayback() {
    this.mic?.setPlayback(
      this.sounding === true ||
        this.phase === "speaking" ||
        this.phase === "preparing",
    );
  }

  setPhase(phase) {
    this.phase = phase;
    this.orb.phase = phase;
    $("orbState").dataset.phase = phase;
    $("stateLabel").textContent = labels[phase] || "Here with you";
    this.micPlayback();
    if (["thinking", "transcribing", "preparing"].includes(phase))
      this.transcript.waiting();
    else $("liveCaption").classList.remove("waiting");
    this.controls();
  }

  async audio() {
    if (!this.context) {
      // The device's own rate. Asking for the synthesiser's 24 kHz instead was measured to
      // change playback by -68 dBFS - nothing - while putting the browser's echo canceller on
      // a non-native graph, which is the one thing that must keep working while Zen is talking.
      const context = new AudioContext({ latencyHint: "interactive" });
      this.context = context;
      this.sound = new Soundscape(context);
      this.sound.enabled = this.soundEnabled;
      this.playback = new AudioPlayback(
        context,
        (event) => this.send(event),
        (kind, text, phrase, generation) => {
          if (generation !== this.generation) return;
          if (kind === "start") this.transcript.show("Zen", text);
          else this.transcript.append("Zen", text, generation);
        },
        (sounding) => {
          this.sounding = sounding;
          this.micPlayback();
        },
      );
      this.playback.reset(this.generation);
      this.mic = new Microphone(context, {
        onFrame: (frame) => {
          if (!this.ready || this.revision === null) return false;
          // Returning false here stops the microphone outright, so it must mean the
          // queue is genuinely stuck, not merely busy. `enqueue` sheds old audio first.
          return this.enqueue(AUDIO_FRAME, frame);
        },
        onLevel: (level) => {
          this.inputLevel = level;
        },
        // Energy alone also detects taps, fans and residual speaker echo. Keep sending
        // capture, but let confirmed engine VAD issue the generation-fenced clear.
        onState: () => {
          this.controls();
          this.wake();
          this.features();
        },
        onError: (message) => {
          this.send({ type: "end_audio" });
          this.status(message, true);
        },
      });
      context.onstatechange = () => {
        if (
          this.context !== context ||
          !this.ready ||
          context.state === "running"
        )
          return;
        this.mic?.stop(false);
        this.playback?.clear();
        this.send({ type: "interrupt" });
        this.status(
          "Audio paused. Tap Start talking or send a message to continue.",
        );
      };
    }
    await this.context.resume();
    return this.context;
  }

  /// Speaking and typing both need a session, and audio needs the gesture that
  /// triggered them: Chromium will not resume a context or open a microphone
  /// without one, embedded or not. Both paths come through here.
  async ensureSession() {
    if (this.stopping) return false;
    if (this.attached && this.ready) return true;
    if (this.connecting) return false;
    const prompt = $("prompt").value.trim();
    if (bytes(prompt) > 8192 || prompt.includes("\0")) {
      this.status(errors.invalid_prompt, true);
      return false;
    }
    this.connecting = true;
    const lifecycle = this.lifecycle;
    this.controls();
    try {
      await this.audio();
      if (lifecycle !== this.lifecycle) return false;
      this.feedback("connect");
      const ready = new Promise((resolve) => {
        this.readyResolve = resolve;
        this.readyTimer = setTimeout(() => {
          void this.disconnect("The models took too long to load. Check the installation and try again.", true);
        }, 390000);
      });
      await this.attach();
      return await ready;
    } catch {
      if (lifecycle === this.lifecycle)
        this.status(
          "Audio could not start. Check your sound device and try again.",
          true,
        );
      this.settleReady(false);
      return false;
    } finally {
      if (lifecycle === this.lifecycle) {
        this.connecting = false;
        this.controls();
      }
    }
  }

  settleReady(ready) {
    clearTimeout(this.readyTimer);
    this.readyResolve?.(ready);
    this.readyResolve = null;
  }

  /// Empty the rolling window without ending the session, adopting whatever
  /// instructions are currently in settings. The model stays loaded, so this is
  /// immediate rather than a five gigabyte reload.
  startFresh() {
    const prompt = $("prompt").value.trim();
    if (bytes(prompt) > 8192 || prompt.includes("\0")) {
      this.status(errors.invalid_prompt, true);
      return;
    }
    if (!this.ready) return;
    this.playback?.clear();
    if (!this.send({
      type: "clear_history",
      system_prompt: prompt || null,
    })) return;
    $("audioDialog").close();
    this.status("Starting a fresh conversation…");
  }

  /// Bind this page to the engine, starting a session if none is running. The
  /// engine loads models on the first attach, so this returns before it is ready
  /// and the UI waits for the `ready` event.
  async attach() {
    const lifecycle = this.lifecycle;
    this.ready = false;
    this.setPhase("loading");
    this.status("Loading models on your device. Your conversation will start automatically.");

    // One channel, because order matters coming back too. A phrase's audio must not
    // reach the scheduler before the `phrase_start` that announces it: the block has
    // nowhere to attach, playback throws, and the turn is cleared mid-sentence. Two
    // channels give no ordering between themselves, however fast each one is.
    const frames = new this.Channel();
    // Native events can arrive before the command's promise returns its revision.
    const pending = [];
    let bound = false;
    frames.onmessage = (frame) => {
      if (lifecycle !== this.lifecycle) return;
      if (!bound) {
        if (pending.length >= 256) {
          void this.disconnect("The engine sent an invalid startup sequence. Please try again.", true);
          return;
        }
        pending.push(frame);
      } else this.receiveFrame(frame);
    };

    const prompt = $("prompt").value.trim();
    try {
      this.attachRequest = this.invoke("zen_attach", {
        frames,
        systemPrompt: prompt || null,
      });
      const revision = await this.attachRequest;
      if (lifecycle !== this.lifecycle) {
        await this.invoke("zen_detach", { revision }).catch(() => {});
        return;
      }
      this.revision = revision;
      this.input = new OrderedInput(
        this.invoke,
        revision,
        () => {
          if (lifecycle === this.lifecycle)
            void this.disconnect("The engine stopped accepting audio. Start a new session to reconnect.", true);
        },
        {
          onShed: (total) => {
            if (lifecycle === this.lifecycle) this.reportShedding(total);
          },
        },
      );
      this.attached = true;
      bound = true;
      for (const frame of pending) {
        if (lifecycle !== this.lifecycle) break;
        this.receiveFrame(frame);
      }
    } catch (reason) {
      if (lifecycle !== this.lifecycle) return;
      this.erase();
      this.status(String(reason), true);
      return;
    }
    this.controls();
  }

  /// One frame from the engine: a kind byte, then either a JSON message or a
  /// little-endian audio header followed by 16-bit PCM.
  receiveFrame(frame) {
    if (this.revision === null) return;
    const view =
      frame instanceof ArrayBuffer
        ? new Uint8Array(frame)
        : ArrayBuffer.isView(frame)
          ? new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength)
          : Uint8Array.from(frame);
    if (!view.byteLength) return;
    const body = view.subarray(1);

    if (view[0] === CONTROL_FRAME) {
      try {
        this.receive(JSON.parse(new TextDecoder().decode(body)));
      } catch {
        this.playback?.clear();
        this.send({ type: "interrupt" });
        this.status("That turn could not be read. Please try again.", true);
      }
      return;
    }
    if (view[0] !== AUDIO_FRAME || body.byteLength <= AUDIO_HEADER) return;
    const head = new DataView(body.buffer, body.byteOffset, AUDIO_HEADER);
    try {
      this.playback?.queue({
        generation: Number(head.getBigUint64(0, true)),
        phrase: Number(head.getBigUint64(8, true)),
        sequence: Number(head.getBigUint64(16, true)),
        pcm: body.subarray(AUDIO_HEADER),
      });
    } catch {
      this.playback?.clear();
      this.send({ type: "interrupt" });
      this.status("That reply could not be played. Please try again.", true);
    }
  }

  receive(event) {
    switch (event.type) {
      case "history_cleared":
        this.transcript.reset();
        $("captionSpeaker").textContent = "A fresh page";
        $("captionText").textContent = "Nothing before this. Go ahead.";
        this.status("New conversation. Your instructions are applied.");
        break;
      case "ready":
        this.ready = true;
        this.settleReady(true);
        // A fresh engine starts at its own default, so the remembered pause has to be
        // restated. Sending it unconditionally keeps one code path instead of two.
        this.sendTuning();
        this.setPhase("idle");
        this.status("");
        this.feedback("ready");
        break;
      case "state":
        if (event.generation !== undefined && event.generation < this.generation) break;
        if (
          event.generation !== undefined &&
          event.generation !== this.generation
        ) {
          this.generation = event.generation;
          this.playback?.reset(this.generation);
        }
        this.setPhase(event.phase);
        break;
      case "clear":
        if (event.generation < this.generation) break;
        this.generation = event.generation;
        this.playback?.reset(this.generation);
        $("liveCaption").classList.remove("waiting", "partial");
        // Keep only the latest completed caption when a scheduled phrase was interrupted.
        {
          const last = this.transcript.entries.at(-1);
          if (last) this.transcript.show(last.who, last.body.textContent);
          else {
            $("captionSpeaker").textContent = "All the time you need";
            $("captionText").textContent = "Go ahead. I’m listening.";
          }
        }
        break;
      case "endpoint":
        if (Number.isFinite(event.ms)) {
          this.endpointActual = event.ms;
          if (this.pauseAuto) this.showEndpoint();
        }
        break;
      case "transcript_partial":
        // Partials are recogniser hypotheses and may still be in the speaker's language.
        // Keep them out of the visible transcript until the repair layer accepts the turn.
        if (event.generation === this.generation)
          this.transcript.show("You", "Understanding your words…", true);
        break;
      case "transcript":
        if (event.generation !== this.generation) break;
        this.status("");
        this.transcript.show("You", event.text);
        this.transcript.append("You", event.text, event.generation);
        break;
      case "phrase_start":
        this.playback?.begin(event);
        break;
      case "phrase_end":
        this.playback?.end(event);
        break;
      case "error": {
        // The engine names what actually failed when it can; a missing model file is
        // something the person can fix, and the generic sentence is not.
        const detail =
          typeof event.detail === "string" && event.detail.trim()
            ? ` (${event.detail.trim().slice(0, 300)})`
            : "";
        const message =
          (errors[event.code] || "This turn couldn’t finish. Please try again.") +
          detail;
        if (fatalErrors.has(event.code)) {
          void this.disconnect(message, true);
          break;
        } else {
          this.playback?.clear();
          if (["audio_stalled", "playback_stalled"].includes(event.code))
            this.mic?.stop(false);
        }
        this.status(message, true);
        this.feedback("error");
        break;
      }
    }
  }

  async toggleMic() {
    if (!this.ready || this.micChanging) return;
    const operation = ++this.micOperation;
    const session = this.revision;
    this.micChanging = true;
    this.controls();
    try {
      await this.audio();
      if (!this.ready || !this.mic || this.revision !== session) return;
      const mic = this.mic;
      if (mic.active) {
        await mic.stop();
        this.send({ type: "end_audio" });
        this.feedback("mute");
      } else {
        this.feedback("mic");
        await mic.start(this.deviceId);
        if (mic.active) {
          this.status("");
          await this.devices();
        }
      }
    } catch (error) {
      if (operation !== this.micOperation) return;
      const message =
        {
          NotAllowedError:
            "Microphone access is blocked. Enable microphone access for desktop apps in Windows Settings, then try again.",
          NotFoundError:
            "No microphone is available. Connect one and try again.",
          NotReadableError:
            "Your microphone is busy or unavailable. Check its connection.",
          OverconstrainedError:
            "That microphone is no longer available. Choose System default.",
        }[error.name] || error.message;
      this.status(message, true);
    } finally {
      if (operation === this.micOperation) {
        this.micChanging = false;
        this.controls();
      }
    }
  }

  stopReply() {
    this.playback?.clear();
    this.send({ type: "interrupt" });
    this.feedback("stop");
  }

  async wake() {
    if (!this.mic?.active || document.hidden) {
      const lock = this.wakeLock;
      this.wakeLock = null;
      await lock?.release().catch(() => {});
      return;
    }
    if (this.wakeLock || this.wakePending || !navigator.wakeLock) return;
    this.wakePending = true;
    const context = this.context;
    try {
      const lock = await navigator.wakeLock.request("screen");
      if (!this.mic?.active || this.context !== context || document.hidden)
        await lock.release();
      else {
        this.wakeLock = lock;
        lock.addEventListener("release", () => {
          if (this.wakeLock === lock) this.wakeLock = null;
        });
      }
    } catch {
      /* Screen wake lock is an optional convenience, not an audio prerequisite. */
    } finally {
      this.wakePending = false;
    }
  }

  features() {
    const settings = this.mic?.settings;
    $("audioFeatures").replaceChildren();
    if (!settings) {
      $("audioFeatures").textContent =
        "Start your microphone to see the processing available on this device.";
      return;
    }
    for (const [key, title] of [
      ["echoCancellation", "Echo cancellation"],
      ["noiseSuppression", "Noise suppression"],
      ["autoGainControl", "Automatic voice level"],
    ]) {
      const row = document.createElement("div");
      row.className = "audio-feature";
      const label = document.createElement("span"),
        value = document.createElement("span");
      label.textContent = title;
      value.textContent =
        settings[key] === true || typeof settings[key] === "string"
          ? "On"
          : settings[key] === false
            ? "Off"
            : "Not reported";
      row.append(label, value);
      $("audioFeatures").append(row);
    }
  }

  async devices() {
    if (!navigator.mediaDevices?.enumerateDevices || this.revision === null)
      return;
    const session = this.revision;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (session !== this.revision) return;
      for (const [id, kind] of [
        ["inputDevice", "audioinput"],
        ["outputDevice", "audiooutput"],
      ]) {
        const select = $(id),
          previous = select.value;
        select.replaceChildren(new Option("System default", ""));
        for (const device of devices.filter(
          (device) =>
            device.kind === kind &&
            device.deviceId &&
            device.deviceId !== "default" &&
            device.label,
        ))
          select.add(new Option(device.label, device.deviceId));
        if ([...select.options].some((option) => option.value === previous))
          select.value = previous;
      }
      const outputAvailable =
        !!this.context?.setSinkId && $("outputDevice").options.length > 1;
      $("outputDevice").hidden = $("outputLabel").hidden = !outputAvailable;
    } catch {
      $("audioNote").textContent =
        "Your audio devices could not be listed. System defaults are still available.";
    }
  }

  erase() {
    this.settleReady(false);
    this.lifecycle++;
    this.micOperation++;
    this.micChanging = false;
    this.connecting = false;
    this.ready = false;
    this.attached = false;
    this.revision = null;
    this.input?.close();
    this.input = null;
    this.mic?.stop(false);
    this.mic = null;
    this.playback?.dispose();
    this.playback = null;
    this.sound?.dispose();
    this.sound = null;
    if (this.context) {
      const old = this.context;
      this.context = null;
      old.onstatechange = null;
      old.close().catch(() => {});
    }
    this.inputLevel = 0;
    this.deviceId = "";
    // Theme and listening pause deliberately survive: they are not conversation state.
    this.generation = 0;
    for (const id of ["prompt", "text"]) $(id).value = "";
    for (const id of ["inputDevice", "outputDevice"])
      $(id).replaceChildren(new Option("System default", ""));
    $("promptCount").textContent = "";
    $("audioDialog").close();
    this.history(false);
    this.transcript.reset();
    this.setPhase("idle");
    this.features();
  }

  /// End the session in the engine. Native workers and the model exit, so no
  /// conversation state survives into whatever session is started next.
  async stop() {
    const attaching = this.attachRequest;
    this.input?.close();
    this.revision = null;
    this.attached = false;
    // Attach and disconnect are independent IPC commands. Wait for admission so
    // cancelling startup cannot run before a delayed attach and leave an orphan.
    if (attaching) await attaching.catch(() => {});
    return this.invoke("zen_disconnect");
  }

  async disconnect(message = "Session ended. Your next conversation starts fresh.", error = false) {
    if (this.stopping) return;
    const stopped = this.stop();
    this.erase();
    this.stopping = true;
    this.controls();
    this.status(error ? message : "Ending session and releasing the models…", error);
    try {
      await stopped;
      this.status(message, error);
    } catch {
      this.status("The engine could not confirm shutdown. Quit Zen from the tray and reopen it.", true);
    } finally {
      this.stopping = false;
      this.controls();
    }
  }

  history(open) {
    const panel = $("transcriptPanel");
    if (open && !panel.open) panel.showModal();
    else if (!open && panel.open) panel.close();
    $("history").setAttribute("aria-expanded", String(open));
    if (open) $("closeHistory").focus();
  }

  bind() {
    for (const button of document.querySelectorAll("[data-theme-choice]"))
      button.onclick = () => {
        this.theme = applyTheme(button.dataset.themeChoice);
        remember("zen.theme", this.theme);
      };
    $("endpoint").value = String(this.endpointMs);
    this.showPauseMode();
    for (const button of document.querySelectorAll("[data-pause-mode]"))
      button.onclick = () => {
        this.pauseAuto = button.dataset.pauseMode === "auto";
        remember("zen.pause", this.pauseAuto ? "auto" : "manual");
        this.showPauseMode();
        this.sendTuning();
      };
    $("endpoint").oninput = () => {
      this.endpointMs = endpointOf($("endpoint").value);
      remember("zen.endpoint", this.endpointMs);
      this.showEndpoint();
      // Applied live: changing how long Zen waits must not cost a session restart.
      this.sendTuning();
    };
    $("startFresh").onclick = () => this.startFresh();
    $("prompt").oninput = () => {
      const count = bytes($("prompt").value);
      $("promptCount").textContent = count
        ? `${count.toLocaleString()} / 8,192`
        : "";
    };
    $("mic").onclick = async () => {
      if (await this.ensureSession()) this.toggleMic();
    };
    $("interrupt").onclick = () => this.stopReply();
    $("disconnect").onclick = () => this.disconnect();
    $("sounds").onclick = () => {
      this.soundEnabled = !this.soundEnabled;
      if (this.sound) {
        this.sound.enabled = this.soundEnabled;
        if (!this.soundEnabled) this.sound.dispose();
        this.feedback("mic");
      }
      this.controls();
    };
    $("history").onclick = () => this.history(!$("transcriptPanel").open);
    $("transcriptPanel").addEventListener("close", () =>
      $("history").setAttribute("aria-expanded", "false"));
    $("closeHistory").onclick = () => {
      this.history(false);
      $("history").focus();
    };
    $("audioSettings").onclick = () => {
      this.features();
      this.devices();
      $("audioDialog").showModal();
    };
    $("closeAudio").onclick = () => $("audioDialog").close();
    $("audioDialog").addEventListener("click", (event) => {
      if (event.target === $("audioDialog")) {
        const r = event.target.getBoundingClientRect();
        if (
          event.clientX < r.left ||
          event.clientX > r.right ||
          event.clientY < r.top ||
          event.clientY > r.bottom
        )
          event.target.close();
      }
    });
    $("textForm").onsubmit = async (event) => {
      event.preventDefault();
      const text = $("text").value.trim();
      if (!text) return;
      if (bytes(text) > 8192 || text.includes("\0")) {
        this.status("Please shorten your message to 8,192 UTF-8 bytes.", true);
        return;
      }
      // Typing is also a way in. Start the session if this is the first thing the
      // person does, then wait for the engine before sending.
      if (!(await this.ensureSession())) return;
      const session = this.revision;
      try {
        await this.audio();
        if (!this.ready || session !== this.revision) return;
        this.playback.clear();
        if (!this.send({ type: "text", text })) return;
        $("text").value = "";
        this.status("");
      } catch {
        this.status(
          "Audio could not resume. Check your sound device and try again.",
          true,
        );
      }
    };
    $("inputDevice").onchange = async () => {
      this.deviceId = $("inputDevice").value;
      if (!this.mic?.active) return;
      const operation = ++this.micOperation;
      this.micChanging = true;
      this.controls();
      const mic = this.mic;
      try {
        await mic.stop();
        this.send({ type: "end_audio" });
        if (this.ready && this.mic === mic) await mic.start(this.deviceId);
      } catch {
        if (operation !== this.micOperation) return;
        this.status(
          "That microphone is unavailable. Choose another device and tap Start talking.",
          true,
        );
      } finally {
        if (operation === this.micOperation) {
          this.micChanging = false;
          this.controls();
        }
      }
    };
    $("outputDevice").onchange = async () => {
      const context = this.context;
      if (!context?.setSinkId) return;
      try {
        await context.setSinkId($("outputDevice").value);
        if (this.context === context) this.sound?.play("mic");
      } catch {
        if (this.context === context) {
          $("outputDevice").value =
            typeof context.sinkId === "string" ? context.sinkId : "";
          $("audioNote").textContent =
            "The speaker could not be changed. The previous output remains selected.";
        }
      }
    };
    navigator.mediaDevices?.addEventListener("devicechange", () =>
      this.devices(),
    );
    document.addEventListener("visibilitychange", () => {
      this.wake();
      if (!document.hidden) this.playback?.tick();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && $("transcriptPanel").open) {
        this.history(false);
        $("history").focus();
      }
    });
    // A reload must not end the conversation, only release this page. Quitting the
    // app revokes the session from the Rust side instead.
    addEventListener("pagehide", () => {
      if (this.revision !== null)
        this.invoke("zen_detach", { revision: this.revision }).catch(() => {});
      this.erase();
      this.orb.dispose();
    });
    if ("mediaSession" in navigator) {
      for (const action of ["pause", "stop"]) {
        try {
          navigator.mediaSession.setActionHandler(action, () => {
            this.mic?.stop(false);
            this.send({ type: "end_audio" });
            this.stopReply();
          });
        } catch {
          /* Optional Chrome hardware media controls. */
        }
      }
    }
  }
}

if (typeof document !== "undefined") {
  if (globalThis.window?.__TAURI__?.core) new VoiceClient();
  else {
    $("status").textContent = "Open the Zen desktop app to start a private voice conversation.";
    $("status").classList.add("error");
    for (const id of ["mic", "text", "sendText", "startFresh"]) $(id).disabled = true;
  }
}
