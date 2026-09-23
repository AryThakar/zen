import { AudioPlayback, Microphone, Soundscape } from "./audio.mjs";
import { OrderedInput, PRIORITY_CONTROLS } from "./transport.mjs";
import { LiquidOrb } from "./orb.mjs";
import { Transcript } from "./transcript.mjs";

const $ = (id) => document.getElementById(id);
const bytes = (text) => new TextEncoder().encode(text).length;

/// Settings - the instructions, theme, quiet-off and the learned pause - are what Zen
/// remembers between launches. Nothing said or heard is stored, and storage can be unavailable
/// outright, which must not stop the page loading.
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

/// The end-of-turn pause the engine last reported learning, handed back to it next session.
const LEARNED_PAUSE = "zen.pause.learned";
function learnedPause() {
  const ms = Math.round(Number(remembered(LEARNED_PAUSE, "")));
  return Number.isFinite(ms) && ms > 0 && ms <= 10_000 ? ms : null;
}
// The hand-set pause is gone, and the sound cues are no longer a setting of their own - the
// board's mute covers them - so their old values mean nothing now.
try {
  localStorage.removeItem("zen.pause");
  localStorage.removeItem("zen.endpoint");
  localStorage.removeItem("zen.sounds");
} catch {
  /* Storage unavailable: nothing to clean. */
}

// The page and the engine are the same process. Commands go out over Tauri's IPC, and ordered
// state and audio frames come back on one channel.

/// Bytes of little-endian header the engine puts ahead of every audio block,
/// after its kind byte.
const AUDIO_HEADER = 24;

/// Frame kinds, the same in both directions: raw audio, or a UTF-8 JSON message.
const AUDIO_FRAME = 0;
const CONTROL_FRAME = 1;

/// Windows lists one microphone three times: as itself, as whatever is currently the default,
/// and as whatever is currently the device for calls. Only the first belongs in a list someone
/// chooses from - the other two are what "System default" already means.
const DEVICE_ALIASES = new Set(["default", "communications"]);

/// Devices that are not really there. Steam installs one on every machine it touches, and the
/// virtual cables come with streaming and conferencing software.
const VIRTUAL_DEVICE = /steam streaming|virtual|vb-audio|voicemeeter|cable output|cable input/i;

/// What to call a device on screen.
///
/// Windows writes "<what it is> (<what it runs on>)", and the second half is the driver:
/// "Microphone Array (2- Intel(R) Smart Sound Technology for Digital Microphones)". Dropping it
/// leaves something a person recognises. Unless what is left is a single word - "Microphone",
/// "Headphones" - which on its own says nothing about which one it is, and then the whole label
/// stays.
function deviceName(label) {
  const bracket = label.indexOf("(");
  if (bracket < 1) return label;
  const outer = label.slice(0, bracket).trim();
  return outer.includes(" ") ? outer : label;
}

/// The menu behind a device row.
///
/// The `<select>` it is built from stays the value and the list of devices - everything that
/// fills it or reads it is unchanged - and this draws what a person actually sees and clicks.
/// The menu is attached inside the dialog rather than to the page, because an open modal makes
/// everything outside it inert: attached to the page it would draw and then ignore every click.
class Picker {
  constructor(button, select, onChoose) {
    this.button = button;
    this.select = select;
    this.onChoose = onChoose;
    this.menu = null;
    button.addEventListener("click", () => (this.menu ? this.close() : this.open()));
    button.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        this.open();
      }
    });
    this.show();
  }

  /// The button says what is chosen now.
  show() {
    const chosen = this.select.selectedOptions[0];
    this.button.querySelector(".chosen").textContent = chosen?.textContent || "System default";
    this.button.title = chosen?.title || "";
  }

  open() {
    this.close();
    const menu = document.createElement("div");
    menu.className = "menu";
    // Windows names devices only once the microphone has been allowed. Before that there is
    // nothing to choose between, so the menu says how to get there instead of offering
    // "System default" alone.
    const unnamed = this.button.getAttribute("aria-disabled") === "true";
    if (unnamed) {
      menu.classList.add("note");
      menu.setAttribute("role", "status");
      menu.textContent = this.button.dataset.note || "";
    } else menu.setAttribute("role", "listbox");
    for (const option of unnamed ? [] : this.select.options) {
      const item = document.createElement("button");
      item.type = "button";
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.value === this.select.value));
      item.textContent = option.textContent;
      item.title = option.title || option.textContent;
      item.onclick = () => {
        this.close();
        this.button.focus();
        if (option.value === this.select.value) return;
        this.select.value = option.value;
        this.show();
        this.onChoose();
      };
      menu.append(item);
    }
    (this.button.closest("dialog") || document.body).append(menu);
    this.place(menu);
    this.menu = menu;
    this.button.setAttribute("aria-expanded", "true");
    // Without `preventScroll`, focusing an item nudges the sheet, and the scroll that follows
    // would be taken for someone scrolling the menu away.
    (menu.querySelector('[aria-selected="true"]') || menu.querySelector("button"))?.focus({ preventScroll: true });
    this.dismiss = (event) => {
      if (!menu.contains(event.target) && event.target !== this.button) this.close();
    };
    this.keys = (event) => {
      const items = [...menu.querySelectorAll("button")];
      const at = items.indexOf(document.activeElement);
      if (event.key === "Escape") {
        // Escape closes the menu and stops there: a dialog closes on Escape by default, and
        // stopping the event alone does not stop that.
        event.preventDefault();
        event.stopPropagation();
        this.close();
        this.button.focus();
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const step = event.key === "ArrowDown" ? 1 : items.length - 1;
        items[(at + step + items.length) % items.length]?.focus();
      }
    };
    // The sheet can scroll under the menu - the click that opened it may have scrolled the row
    // into view - so the menu follows its row, and closes once that row is gone from the sheet.
    this.moved = (event) => {
      if (event && menu.contains(event.target)) return;
      const row = this.button.getBoundingClientRect();
      const sheet = (this.button.closest(".sheet-body") || this.button.closest("dialog"))?.getBoundingClientRect();
      if (sheet && (row.bottom < sheet.top || row.top > sheet.bottom)) this.close();
      else this.place(menu);
    };
    document.addEventListener("pointerdown", this.dismiss, true);
    document.addEventListener("keydown", this.keys, true);
    window.addEventListener("resize", this.moved);
    window.addEventListener("scroll", this.moved, true);
  }

  /// Under the button, or above it where there is no room below.
  place(menu) {
    const at = this.button.getBoundingClientRect();
    const height = menu.offsetHeight;
    const below = window.innerHeight - at.bottom - 8;
    const top = height <= below || at.top < height ? at.bottom + 6 : at.top - height - 6;
    menu.style.top = `${Math.max(8, Math.min(top, window.innerHeight - height - 8))}px`;
    menu.style.left = `${Math.max(8, Math.min(at.right - menu.offsetWidth, window.innerWidth - menu.offsetWidth - 8))}px`;
  }

  close() {
    if (!this.menu) return;
    document.removeEventListener("pointerdown", this.dismiss, true);
    document.removeEventListener("keydown", this.keys, true);
    window.removeEventListener("resize", this.moved);
    window.removeEventListener("scroll", this.moved, true);
    this.menu.remove();
    this.menu = null;
    this.button.setAttribute("aria-expanded", "false");
  }
}

/// The phases in which Zen is busy with, or saying, a reply: Stop has something to stop, and the
/// aurora around the orb is lit even with the microphone off.
const BUSY_PHASES = new Set(["transcribing", "thinking", "preparing", "speaking"]);
const labels = {
  idle: "Ready when you are",
  listening: "Listening",
  transcribing: "Understanding you",
  thinking: "Thinking",
  preparing: "Preparing a reply",
  speaking: "Speaking",
  loading: "Loading local models",
};
/// What the card says once Zen has gone to sleep after two quiet minutes.
const ASLEEP = "The microphone turned off after two quiet minutes.";
/// Settings kept between launches. The devices are not among them: Windows' own default is the
/// usual answer, and a device named last week may not be plugged in today.
const SETTINGS = { prompt: "zen.prompt", quietOff: "zen.quietOff" };
/// Cues played over Zen's voice are this much quieter, about 8 dB, so the two never sum past
/// full scale and the cue does not talk over the words.
const UNDER_VOICE = 0.4;
const errors = {
  invalid_prompt: "Please keep your instructions within 8,192 UTF-8 bytes.",
  invalid_input: "I lost part of that. Please say it again.",
  backend_unavailable:
    "The local engine stopped. Check the model files, available graphics memory, and whether another Zen engine is running. Then try again.",
  model_failed:
    "I couldn’t finish that thought. Try a shorter message or instructions.",
  recognition_failed: "I couldn’t make out that last part. Please try again.",
  recognition_incomplete: "I lost part of what you said. Please repeat the full question.",
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
/// Errors the engine sends that end the session. Instructions that are too long are caught here,
/// before they are ever sent, so the engine has no error of its own for them.
const fatalErrors = new Set(["backend_unavailable"]);
/// How long the page waits for the engine to report ready: the model server's 240 s to load,
/// then up to 60 s for each speech worker, and half a minute more, so that a failed start is
/// always reported by the engine itself before the page gives up on it.
const READY_TIMEOUT_MS = 390_000;

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
    /// Zen's voice and cues silenced from the board. For this launch only: a Zen that greets in
    /// silence the next time it opens would look broken rather than muted.
    this.muted = false;
    /// The phrase of Zen's being said now, whose sentences the board puts up as they come.
    this.speaking = null;
    /// Turn the microphone off after two minutes with nobody talking. On unless switched off.
    this.quietOff = remembered(SETTINGS.quietOff, "on") !== "off";
    this.deviceId = "";
    this.theme = applyTheme(remembered("zen.theme", "system"));
    /// What the engine reports it is actually waiting. It moves as the engine learns.
    this.endpointActual = learnedPause();
    this.micChanging = false;
    this.micOperation = 0;
    this.lifecycle = 0;
    /// Asleep: the microphone went off after two quiet minutes, and the page shows it resting.
    this.asleep = false;
    /// When each phrase of Zen's became audible, so the pace of its speech can be measured.
    this.phraseStarted = new Map();
    this.transcript = new Transcript({
      list: $("historyList"),
      scroller: $("historySheet").querySelector(".sheet-body"),
      caption: $("liveCaption"),
      speaker: $("captionSpeaker"),
      text: $("captionText"),
      frame: $("captionFrame"),
    });
    /// The aurora's three bands, and what they light.
    this.bands = [0, 0, 0];
    this.glowTime = 0;
    this.glowTargets = [...document.querySelectorAll(".orb-aurora, .glass-light")];
    this.orb = new LiquidOrb(
      $("orb"),
      () => ({
        input: this.mic?.active ? this.inputLevel : 0,
        output: this.playback?.level() || 0,
      }),
      (listen, voice, dt) => {
        this.glow(listen, voice, dt);
        this.caption();
      },
    );
    this.bind();
    this.controls();
  }

  /// The aurora around the orb, and the light it casts on the card, answer whichever voice is
  /// heard - yours while Zen listens, Zen's while it speaks - in three bands a little out of step,
  /// so the light ripples rather than pulsing as one. Each rises fast and falls slowly, like the
  /// orb, and the smoothing runs on time rather than frames, so the orb's frame rate does not
  /// change how the light moves. Dark, and left alone, while the aurora is hidden.
  glow(listen, voice, dt) {
    const lit = document.body.dataset.mic === "on" || BUSY_PHASES.has(document.body.dataset.phase);
    if (!lit && this.bands.every((band) => band < 0.001)) return;
    const heard = lit ? Math.min(1, Math.max(listen, voice)) : 0;
    this.glowTime += dt;
    for (const [i, rate, offset] of [[0, 2.3, 0], [1, 1.7, 2.1], [2, 2.9, 4.2]]) {
      const target = Math.min(1, heard * (0.72 + 0.28 * Math.sin(this.glowTime * rate + offset)));
      const step = 1 - Math.pow(1 - (target > this.bands[i] ? 0.35 : 0.06), dt * 60);
      this.bands[i] += (target - this.bands[i]) * step;
      for (const element of this.glowTargets) element.style.setProperty(`--b${i + 1}`, this.bands[i].toFixed(3));
    }
  }

  /// Keeps the board in step with Zen's voice: every sentence of the phrase playing that has
  /// begun goes up. Run with the orb, once a frame.
  caption() {
    if (this.speaking === null) return;
    const progress = this.playback?.position(this.speaking);
    if (progress) this.transcript.reach(this.speaking, progress.heard, progress.length);
  }

  /// Something has been said or asked, so the starters step aside until the next fresh page.
  begin() {
    document.body.toggleAttribute("data-started", true);
  }

  /// Anything that brings the listener back wakes Zen.
  wake() {
    if (!this.asleep) return;
    this.asleep = false;
    if ($("status").textContent === ASLEEP) this.status("");
    this.controls();
  }

  /// The text box grows with what is typed, to five lines, then scrolls; Send is live only while
  /// there is something to send.
  fit() {
    const text = $("text");
    text.style.height = "auto";
    const max = parseFloat(getComputedStyle(text).maxHeight);
    text.style.height = `${Math.min(text.scrollHeight, max || text.scrollHeight)}px`;
    text.classList.toggle("scrolls", text.scrollHeight > max);
    $("sendText").disabled = text.disabled || !text.value.trim();
  }

  /// The day and time in the bar, in the short form where the bar is narrow.
  clock() {
    const now = new Date();
    const short = innerWidth < 480;
    const day = now.toLocaleDateString("en-GB", short
      ? { weekday: "short", day: "numeric", month: "short" }
      : { weekday: "long", day: "numeric", month: "long" });
    const time = now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    $("clock").textContent = `${day} · ${time}`;
    $("clock").dateTime = now.toISOString();
    return now;
  }

  /// A sound cue, unless Zen is muted; lower while Zen's voice is playing.
  feedback(event) {
    if (this.muted) return;
    const tone = ["connect", "ready", "mic", "mute", "quiet", "stop", "send", "error"].includes(event) ? event : null;
    if (tone) this.sound?.play(tone, this.playback?.pending() > 0 ? UNDER_VOICE : 1);
  }

  /// Silences Zen - its voice and its cues - or brings it back. A reply carries on while muted,
  /// on the board and in the conversation, and the orb still moves as it is said.
  mute(muted) {
    this.muted = muted;
    this.playback?.mute(muted);
    if (this.sound) {
      this.sound.enabled = !muted;
      if (muted) this.sound.dispose();
    }
    // Coming back, a soft tick says the sound is on again.
    if (!muted) this.feedback("send");
    this.controls();
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

  /// Hands the engine the pause it learned last session, so it does not start from its default
  /// and spend the first few turns relearning how this person talks.
  sendLearnedPause() {
    const ms = learnedPause();
    if (this.ready && ms !== null) this.send({ type: "learned_pause", ms });
  }

  showEndpoint() {
    // Say what it settled on. A number that moves on its own and is never shown is worse
    // than no number at all.
    $("endpointNote").textContent =
      this.endpointActual === null
        ? "Learning"
        : `${(this.endpointActual / 1000).toFixed(2)} s`;
  }

  /// Where the wait before the last answer went. A slow reply is otherwise a guess: the
  /// recogniser, the check, the model and the voice each look the same from outside.
  showTiming(event) {
    const seconds = (ms) => (ms / 1000).toFixed(2);
    if (!Number.isFinite(event.total_ms)) return;
    // Shown only when it would not read as 0.00.
    const queued =
      Math.round(event.queue_ms / 10) > 0 ? ` (${seconds(event.queue_ms)} queued)` : "";
    const stages = [
      ["Recognising", event.recognize_ms, queued],
      ["Checking", event.repair_ms, ""],
      ["First word", event.think_ms, ""],
      ["First sound", event.speak_ms, ""],
    ]
      .filter(([, ms]) => Number.isFinite(ms))
      .map(([name, ms, note]) => `${name} ${seconds(ms)}${note}`);
    $("timingValue").textContent = `${seconds(event.total_ms)} s`;
    $("timingNote").textContent = stages.join(" · ");
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
  enqueue(kind, payload, priority = false) {
    return this.input?.send(kind, payload, priority) ?? false;
  }

  send(event) {
    return this.enqueue(
      CONTROL_FRAME,
      new TextEncoder().encode(JSON.stringify(event)),
      PRIORITY_CONTROLS.has(event.type),
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
    $("text").disabled = busy;
    $("sendText").disabled = busy || !$("text").value.trim();
    $("mic").disabled = busy;
    $("startFresh").disabled = !this.ready || !!this.stopping;
    $("endSession").disabled = !connected || !!this.stopping;
    $("interrupt").disabled =
      !this.ready ||
      (!BUSY_PHASES.has(this.phase) && !this.playback?.pending());
    const micOn = !!this.mic?.active;
    const displayPhase =
      this.phase === "listening" && !micOn ? "idle" : this.phase;
    $("stateLabel").textContent = this.asleep
      ? "Sleeping"
      : displayPhase === "idle" && micOn
        ? "Ready to listen"
        : labels[displayPhase] || "Here with you";
    $("orbState").dataset.phase = displayPhase;
    // The page's own state, for everything drawn around the conversation: the aurora is lit
    // while the microphone is on or a reply is under way, and the page rests while Zen sleeps.
    document.body.dataset.mic = micOn ? "on" : "off";
    document.body.dataset.phase = displayPhase;
    document.body.toggleAttribute("data-sleep", this.asleep);
    $("mic").setAttribute("aria-pressed", String(micOn));
    $("micIcon").setAttribute("href", micOn ? "#i-muted" : "#i-mic");
    $("micLabel").textContent = this.stopping ? "Ending session…" : this.connecting ? "Loading models…" : this.mic?.pending
      ? "Allow microphone…"
      : micOn
        ? "Mute microphone"
        : "Start talking";
    $("sounds").setAttribute("aria-pressed", String(this.muted));
    $("sounds").setAttribute("aria-label", this.muted ? "Unmute Zen" : "Mute Zen");
    $("sounds").title = this.muted ? "Unmute Zen" : "Mute Zen";
    $("soundIcon").setAttribute("href", this.muted ? "#i-sound-off" : "#i-sound");
    // Stop takes its place in the row only while there is something to stop.
    $("interrupt").classList.toggle("away", $("interrupt").disabled);
    $("inputPicker").disabled = this.micChanging || !!this.mic?.pending;
  }

  setPhase(phase) {
    const was = this.phase;
    this.phase = phase;
    this.orb.phase = phase;
    // While Zen works on a reply, a wave waits under what was said - which stays in view.
    $("liveCaption").classList.toggle("waiting", ["thinking", "transcribing", "preparing"].includes(phase));
    // What was being heard came to nothing - a cough, a word lost to noise - so the board goes
    // back to what it showed before instead of waiting on it for good.
    if (was === "transcribing" && ["listening", "idle"].includes(phase) && $("liveCaption").classList.contains("partial"))
      this.transcript.recall();
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
      this.sound.enabled = !this.muted;
      this.playback = new AudioPlayback(
        context,
        (event) => this.send(event),
        (kind, text, phrase, generation) => {
          if (generation !== this.generation) return;
          if (kind === "start") {
            this.phraseStarted.set(phrase, performance.now());
            this.speaking = phrase;
            this.transcript.say(generation, phrase, text);
          } else {
            // How long it took to say is how fast Zen speaks, which times the sentences of a
            // phrase whose length is not known yet.
            const began = this.phraseStarted.get(phrase);
            this.phraseStarted.delete(phrase);
            if (began !== undefined) this.transcript.measured(text.length, performance.now() - began);
            this.transcript.said(phrase);
            if (this.speaking === phrase) this.speaking = null;
            this.transcript.append("Zen", text, generation);
          }
        },
        () => {
          this.send({ type: "interrupt" });
          this.status("Voice playback stopped. Please try again.", true);
        },
      );
      this.playback.reset(this.generation);
      this.playback.mute(this.muted);
      // Chosen before there was anything to play through.
      const speaker = $("outputDevice").value;
      if (speaker && context.setSinkId)
        await context.setSinkId(speaker).catch(() => {});
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
        onState: () => {
          this.controls();
          this.keepScreenOn();
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
        }, READY_TIMEOUT_MS);
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
        // A fresh page offers its ways in again.
        document.body.removeAttribute("data-started");
        break;
      case "ready":
        this.ready = true;
        this.settleReady(true);
        // A fresh engine starts at its own default; the pause learned last time is where it
        // should start instead.
        this.sendLearnedPause();
        this.setPhase("idle");
        this.status("");
        // No tone here. The engine speaks a greeting the moment it reports ready - the two are
        // dispatched a line apart - so a cue at this instant lands on top of Zen's first words
        // and is heard as interference in the voice rather than as a signal. It comes once the
        // greeting is over instead.
        break;
      case "greeted":
        // Zen listens from here on. Nothing said during the greeting was used, so say when it
        // is the listener's turn - to anyone who can talk, that is.
        if (this.mic?.active) this.feedback("ready");
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
        this.phraseStarted.clear();
        // What was said of an interrupted reply stays on the board; the rest never goes up.
        this.speaking = null;
        this.transcript.stop();
        if ($("liveCaption").classList.contains("partial")) this.transcript.recall();
        break;
      case "heard":
        // A reply was cut off. The engine keeps what was certainly heard of it, and the
        // history shown here ends in the same place, marked the same way.
        if (event.generation === this.generation && typeof event.text === "string")
          this.transcript.cut("Zen", event.text, event.generation);
        break;
      case "notice":
        // Part of what was said came through and part plainly did not. The answer that follows
        // is to what was heard, so say so rather than let it look like a full reply.
        if (event.code === "partly_unheard" && event.generation === this.generation)
          this.status("I may have missed part of what you said.");
        break;
      case "quiet":
        void this.quiet();
        break;
      case "endpoint":
        if (Number.isFinite(event.ms)) {
          this.endpointActual = event.ms;
          remember(LEARNED_PAUSE, String(event.ms));
          this.showEndpoint();
        }
        break;
      case "timing":
        if (event.generation === this.generation) this.showTiming(event);
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

  /// `offCue` is the chime for turning it off: a mute, or falling quiet after two silent minutes.
  async toggleMic(offCue = "mute") {
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
        this.feedback(offCue);
      } else {
        this.feedback("mic");
        await mic.start(this.deviceId);
        if (mic.active) {
          // The listener is back, so the engine's quiet timer starts again from now.
          this.send({ type: "active" });
          this.asleep = false;
          this.status("");
          await this.devices();
          this.warnHandsFree();
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
        // A device changed while the microphone was being turned on or off. Turned off, there
        // is nothing to move; turned on, it may have opened a device that is now gone.
        if (this.micRestartPending) {
          this.micRestartPending = false;
          if (this.mic?.active) void this.restartMic();
          else this.micFallback = false;
        }
      }
    }
  }

  /// Nobody has spoken for two minutes: the microphone goes off, exactly as if muted, unless the
  /// listener has asked for it to stay on, and Zen goes to sleep until something wakes it.
  async quiet() {
    if (!this.quietOff || !this.mic?.active || this.micChanging) return;
    await this.toggleMic("quiet");
    if (this.mic?.active) return;
    this.asleep = true;
    this.status(ASLEEP);
    this.controls();
  }

  stopReply() {
    this.playback?.clear();
    this.send({ type: "interrupt" });
    this.feedback("stop");
  }

  /// Holds a screen wake lock while the microphone is live and the window visible.
  async keepScreenOn() {
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
    const row = (title, detail) => {
      const item = document.createElement("div");
      item.className = "row";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = title;
      item.append(label);
      if (detail !== undefined) {
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = detail;
        item.append(value);
      }
      return item;
    };
    if (!settings) {
      const hint = row("Audio processing");
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = "Shown once the microphone is on.";
      hint.firstChild.append(sub);
      $("audioFeatures").replaceChildren(hint);
      return;
    }
    $("audioFeatures").replaceChildren(
      ...[
        ["echoCancellation", "Echo cancellation"],
        ["noiseSuppression", "Noise suppression"],
        ["autoGainControl", "Automatic voice level"],
      ].map(([key, title]) =>
        row(
          title,
          // Chrome may report a mode name ("all", "remote-only") instead of true; both are on.
          settings[key] === true || typeof settings[key] === "string"
            ? "On"
            : settings[key] === false
              ? "Off"
              : "Not reported",
        ),
      ),
    );
  }

  /// Moves live capture onto the chosen device, keeping everything already said. Returns
  /// whether capture is running on it afterwards.
  ///
  /// A change that arrives while a switch is already under way is not dropped: the switch in
  /// flight was aimed at a device that may no longer be the choice - or may no longer exist, if
  /// it was unplugged a moment after being picked - so the newest choice is applied once that
  /// switch lands, whether it succeeded or not. `resume` is that replay: capture was meant to be
  /// running, so it starts even though the failed switch left it stopped.
  async restartMic(resume = false) {
    if (this.micChanging) {
      this.micRestartPending = true;
      return false;
    }
    const mic = this.mic;
    if (!mic || (!resume && !mic.active)) return false;
    const operation = ++this.micOperation;
    this.micChanging = true;
    this.controls();
    let running = false;
    try {
      if (mic.active) {
        await mic.stop();
        this.send({ type: "end_audio" });
      }
      if (this.ready && this.mic === mic) await mic.start(this.deviceId);
      running = operation === this.micOperation && mic.active;
    } catch {
      // A newer choice is about to be tried; reporting this one would only flash an error.
      if (operation === this.micOperation && !this.micRestartPending)
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
    if (operation === this.micOperation && this.micRestartPending) {
      this.micRestartPending = false;
      return this.restartMic(true);
    }
    if (running) {
      this.warnHandsFree();
      if (this.micFallback) {
        this.micFallback = false;
        this.status("Your microphone was disconnected, so Zen switched to the system default.");
      }
    }
    return running;
  }

  /// Using a Bluetooth headset's own microphone switches Windows to the hands-free profile,
  /// which carries audio both ways at phone-call quality - Zen's voice included. The device name
  /// is the only place that shows, so say it where the person can act on it.
  warnHandsFree() {
    const label = this.mic?.label || "";
    const handsFree = /hands-?free/i.test(label);
    $("audioNote").textContent = handsFree
      ? "This Bluetooth headset's microphone switches it to call quality, and Zen will sound muffled. For clear sound, choose your computer's microphone and keep the headset for listening."
      : this.audioNoteDefault;
    if (handsFree)
      this.status("Your Bluetooth headset's microphone lowers Zen's sound quality. Open Audio to change it.");
  }

  async devices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const session = this.revision;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (session !== this.revision) return;
      // The microphone is the one chosen in Settings, and only that choice moves it. Plugging a
      // device in only adds it to the list; unplugging the chosen one falls back to the default.
      const chosenGone =
        this.deviceId !== "" &&
        !devices.some((d) => d.kind === "audioinput" && d.deviceId === this.deviceId);
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
            !DEVICE_ALIASES.has(device.deviceId) &&
            device.label &&
            !VIRTUAL_DEVICE.test(device.label),
        )) {
          const option = new Option(deviceName(device.label), device.deviceId);
          // The full name, for a device whose short one is ambiguous.
          option.title = device.label;
          select.add(option);
        }
        if ([...select.options].some((option) => option.value === previous))
          select.value = previous;
      }
      this.inputPicker?.show();
      this.outputPicker?.show();
      // Both pickers are there from the start. Windows names the devices only once the
      // microphone has been allowed, so until then each picker explains that instead of
      // offering "System default" alone - which is what a fresh session uses anyway.
      const named = devices.some((device) => device.label);
      for (const id of ["inputPicker", "outputPicker"])
        $(id).setAttribute("aria-disabled", String(!named));
      $("outputRow").hidden = !("setSinkId" in AudioContext.prototype);
      // "System default" means whatever Windows is using now, so when that changes - earbuds
      // connected, a headset unplugged - capture moves with it. A named choice does not.
      const fallback =
        devices.find(
          (device) => device.kind === "audioinput" && device.deviceId === "default",
        )?.label || "";
      // Both names have to be known: Windows only names devices once the microphone has been
      // allowed, and a list that has simply gone quiet is not a device change.
      const defaultMoved =
        !!this.defaultInput && !!fallback && this.defaultInput !== fallback;
      this.defaultInput = fallback;
      // Or while a switch is still in flight: plugging earbuds in fires several device
      // changes, and the one that matters can land in the middle of the last one.
      if (defaultMoved && this.deviceId === "" && (this.mic?.active || this.micChanging))
        void this.restartMic();
      if (chosenGone) {
        // The microphone that was picked has been unplugged. Carry on with whatever Windows
        // uses now rather than stopping the conversation. If its track already ended, capture
        // is stopped and the picker is simply left on the default for the next start.
        this.deviceId = "";
        $("inputDevice").value = "";
        this.inputPicker?.show();
        this.micFallback = true;
        const switched = await this.restartMic();
        // Capture was off: nothing moved, so there is nothing to announce on the next start.
        if (!switched && !this.micRestartPending) this.micFallback = false;
      }
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
    // The instructions are a setting, kept like the theme; only the message being typed goes.
    $("text").value = "";
    this.fit();
    for (const id of ["inputDevice", "outputDevice"])
      $(id).replaceChildren(new Option("System default", ""));
    this.inputPicker?.show();
    this.outputPicker?.show();
    $("audioDialog").close();
    this.history(false);
    this.transcript.reset();
    this.phraseStarted.clear();
    this.speaking = null;
    // A session ended is a fresh page: awake, with its ways in on offer again.
    this.asleep = false;
    document.body.removeAttribute("data-started");
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
    const sheet = $("historySheet");
    if (open && !sheet.open) {
      sheet.showModal();
      // Opened on the latest of it, as a conversation is read.
      const body = sheet.querySelector(".sheet-body");
      body.scrollTop = body.scrollHeight;
    } else if (!open && sheet.open) sheet.close();
  }

  bind() {
    for (const button of document.querySelectorAll("[data-theme-choice]"))
      button.onclick = () => {
        this.theme = applyTheme(button.dataset.themeChoice);
        remember("zen.theme", this.theme);
      };
    this.showEndpoint();
    $("startFresh").onclick = () => this.startFresh();
    const promptCount = () => {
      const count = bytes($("prompt").value);
      $("promptCount").textContent = count
        ? `${count.toLocaleString()} / 8,192`
        : "";
    };
    // Kept until changed. Emptied, it is Zen's own voice again - which is the default.
    $("prompt").value = remembered(SETTINGS.prompt, "");
    promptCount();
    $("prompt").oninput = () => {
      remember(SETTINGS.prompt, $("prompt").value);
      promptCount();
    };
    const showQuietOff = () => {
      for (const button of document.querySelectorAll("[data-quiet-off]"))
        button.setAttribute("aria-pressed", String((button.dataset.quietOff === "on") === this.quietOff));
    };
    for (const button of document.querySelectorAll("[data-quiet-off]"))
      button.onclick = () => {
        this.quietOff = button.dataset.quietOff === "on";
        remember(SETTINGS.quietOff, this.quietOff ? "on" : "off");
        showQuietOff();
      };
    showQuietOff();
    $("mic").onclick = async () => {
      this.begin();
      if (await this.ensureSession()) await this.toggleMic();
    };
    $("interrupt").onclick = () => this.stopReply();
    $("disconnect").onclick = () => this.disconnect();
    $("sounds").onclick = () => this.mute(!this.muted);
    $("history").onclick = () => this.history(true);
    $("historySheet").addEventListener("close", () => $("history").focus());
    $("closeHistory").onclick = () => this.history(false);
    $("audioSettings").onclick = () => {
      this.features();
      this.devices();
      $("confirmRow").hidden = true;
      $("endRow").hidden = false;
      $("quitConfirmRow").hidden = true;
      $("quitRow").hidden = false;
      $("audioDialog").showModal();
    };
    $("closeAudio").onclick = () => $("audioDialog").close();
    // Ending the session unloads the models and clears the conversation, so it asks first.
    $("endSession").onclick = () => {
      $("endRow").hidden = true;
      $("confirmRow").hidden = false;
      $("cancelEnd").focus();
    };
    $("cancelEnd").onclick = () => {
      $("confirmRow").hidden = true;
      $("endRow").hidden = false;
      $("endSession").focus();
    };
    $("confirmEnd").onclick = () => {
      $("audioDialog").close();
      void this.disconnect();
    };
    // Quitting is more than ending the session: the app itself stops, tray and all, so nothing
    // is left running in the background. It asks first, the same way.
    $("quitApp").onclick = () => {
      $("quitRow").hidden = true;
      $("quitConfirmRow").hidden = false;
      $("cancelQuit").focus();
    };
    $("cancelQuit").onclick = () => {
      $("quitConfirmRow").hidden = true;
      $("quitRow").hidden = false;
      $("quitApp").focus();
    };
    $("confirmQuit").onclick = () => {
      $("confirmQuit").disabled = true;
      void this.invoke("zen_quit");
    };
    /// What the audio note says when there is nothing to warn about.
    this.audioNoteDefault = $("audioNote").textContent;
    // A click on the dimmed page around a sheet closes it.
    for (const sheet of [$("audioDialog"), $("historySheet")])
      sheet.addEventListener("click", (event) => {
        if (event.target !== sheet) return;
        const r = sheet.getBoundingClientRect();
        if (
          event.clientX < r.left ||
          event.clientX > r.right ||
          event.clientY < r.top ||
          event.clientY > r.bottom
        )
          sheet.close();
      });
    $("text").addEventListener("input", () => this.fit());
    // Enter sends and Shift+Enter starts a new line, as in any message box.
    $("text").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $("textForm").requestSubmit();
      }
    });
    // A starter is sent exactly as if it had been typed.
    for (const starter of document.querySelectorAll(".starters button"))
      starter.onclick = () => {
        $("text").value = starter.textContent;
        this.fit();
        $("textForm").requestSubmit();
      };
    $("textForm").onsubmit = async (event) => {
      event.preventDefault();
      const text = $("text").value.trim();
      if (!text) return;
      if (bytes(text) > 8192 || text.includes("\0")) {
        this.status("Please shorten your message to 8,192 UTF-8 bytes.", true);
        return;
      }
      this.begin();
      this.wake();
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
        this.fit();
        this.feedback("send");
        this.status("");
      } catch {
        this.status(
          "Audio could not resume. Check your sound device and try again.",
          true,
        );
      }
    };
    this.inputPicker = new Picker($("inputPicker"), $("inputDevice"), () =>
      $("inputDevice").dispatchEvent(new Event("change")),
    );
    this.outputPicker = new Picker($("outputPicker"), $("outputDevice"), () =>
      $("outputDevice").dispatchEvent(new Event("change")),
    );
    $("inputDevice").onchange = () => {
      this.deviceId = $("inputDevice").value;
      void this.restartMic();
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
          this.outputPicker?.show();
          $("audioNote").textContent =
            "The speaker could not be changed. The previous output remains selected.";
        }
      }
    };
    navigator.mediaDevices?.addEventListener("devicechange", () =>
      this.devices(),
    );
    document.addEventListener("visibilitychange", () => {
      this.keepScreenOn();
      if (!document.hidden) this.playback?.tick();
    });
    // The clock changes on the minute, so it is redrawn on the minute; and in its short form as
    // soon as the bar gets narrow.
    const nextMinute = () => {
      const now = this.clock();
      this.clockTimer = setTimeout(nextMinute, 60_000 - (now.getSeconds() * 1000 + now.getMilliseconds()));
    };
    nextMinute();
    addEventListener("resize", () => this.clock());
    this.fit();
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
    for (const starter of document.querySelectorAll(".starters button")) starter.disabled = true;
  }
}
