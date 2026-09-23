# Engine IPC contract

Zen's interface and its engine are one process. The window's webview owns capture and
playback; the Rust side owns VAD, recognition, generation and synthesis. They talk over
Tauri's IPC, so there is no listener, no port, no certificate and no pairing token — the
connection is the process itself, and nothing described here is reachable from another
machine.

This document is the contract between `src/client` and `src/bridge.rs`.

## Attaching

The page calls `zen_attach` with one channel and, optionally, instructions for this
session. The engine starts if it is not already running and returns a **revision**.

```js
const frames = new Channel();   // JSON state and transcript, and raw reply PCM
const revision = await invoke("zen_attach", { frames, systemPrompt });
```

One channel and not two, because two cannot be ordered against each other. An
acknowledgement arriving before the audio it acknowledges, or a phrase ending before its
last block, would both be real. The first byte of every frame says which kind it is.

The revision is the fence. A reloaded page gets a new revision, and frames stamped with an
older one are dropped rather than credited to the new page. Models load on the first
attach, so `zen_attach` returns before the engine is usable; wait for `ready`.

Instructions given at attach apply to the session they start. `clear_history` replaces them
without ending the session. The engine holds them in memory only; the page keeps its own copy
as a setting.

## Commands

| Command | Payload | Meaning |
| --- | --- | --- |
| `zen_attach` | `{ frames, systemPrompt? }` | Bind this page to a session; returns the revision. Instructions are at most 8192 UTF-8 bytes with no NUL. |
| `zen_input` | raw bytes | The single ordered path for everything the page sends. See below. |
| `zen_detach` | `{ revision }` | Release this page without ending the session, so a reload does not discard the conversation. |
| `zen_disconnect` | — | End the session. Native workers and the model exit; no state survives into the next session. |
| `zen_quit` | — | Quit the application: the window, the tray icon and every process under them. The window hides at once; a running session gets llama-server's shutdown deadline to close cleanly, then the process ends regardless. |

### The ordered path

Capture and control share one command and one call in flight:

```
[revision u64 LE][kind u8][payload]
  kind 0: PCM, mono signed 16-bit LE at 16000 Hz, 1-3200 samples (the client sends 320 / 20 ms)
  kind 1: a UTF-8 JSON control message
```

This is not incidental. IPC calls are independent promises and can be delivered in any
order — measured at four reorderings in three hundred frames, one jumping five places.
Five frames is a hundred milliseconds of speech, which is enough for an `end_audio` to
overtake the words it follows and clip them, and enough for an acknowledgement to arrive
before the block it acknowledges and fail the turn. A socket ordered these for free; here
the page must keep exactly one call in flight and queue behind it.

Apply device echo cancellation, noise suppression and gain control before sending capture;
the engine does not repeat them. When the queue exceeds about a second of audio the page
sheds its oldest **capture** frames and never control frames: losing 20 ms of speech is
recoverable, while a lost acknowledgement or end-of-utterance wedges the turn.

Control messages carry a `type` and are rejected outright if the type or any field is
unknown. A malformed frame fails the current turn; it does not tear down the session.

| Control | Meaning |
| --- | --- |
| `{"type":"end_audio"}` | Flush pending capture when muting or stopping the microphone. |
| `{"type":"text","text":"Hello"}` | Interrupt the current turn and submit nonblank text, at most 8192 UTF-8 bytes, no NUL. |
| `{"type":"interrupt"}` | Stop the current utterance, generation and output; stay attached. |
| `{"type":"active"}` | The listener is back: restart the two-minute quiet timer. The page sends it when the microphone is turned on. |
| `{"type":"clear_history","system_prompt":"..."}` | Empty the conversation and start a new one without unloading the models. `system_prompt` replaces the persona, at most 8192 UTF-8 bytes with no NUL; null or absent restores Zen's own. Cancels the current turn and is answered with `history_cleared`. |
| `{"type":"learned_pause","ms":1080}` | Start the adaptive end-of-turn pause from a value learned in an earlier session, 1-10000 ms, clamped into the engine's own 0.6-2.0 s range. Send it after `ready` with the last `endpoint` value received. The pause is never set by hand. |
| `{"type":"playback_started","generation":7,"phrase":3}` | The first samples of this phrase have actually begun playing. |
| `{"type":"audio_played","generation":7,"sequence":21}` | This entire PCM block reached playback. Acknowledge blocks in order. |
| `{"type":"played","generation":7,"phrase":3}` | All audio and the trailing pause for this phrase finished playing. |

Do not fake playback acknowledgements: they control backpressure and what the assistant
remembers saying. Acknowledgements from an old generation have no effect. Duplicate
completed block and phrase acknowledgements are idempotent; future or out-of-order
acknowledgements fail the turn. The page cannot supply assistant text in an
acknowledgement.

## Events channel

JSON, in order.

| Type | Fields / behavior |
| --- | --- |
| `ready` | Models loaded; input may begin. Zen then says a short greeting in its own words (written by the model from the day and time of day; a fixed line if that takes more than 4 s), reported as phase `speaking`. Microphone audio is ignored until it has finished, because the browser's echo canceller is still learning Zen's voice and would let part of it through as speech; typing or `interrupt` cut the greeting short. |
| `greeted` | The greeting has finished playing. Microphone audio counts as speech from 600 ms later, once the room is quiet. Play a short cue if the microphone is on, so the listener knows it is their turn. |
| `state` | `phase`, `generation` (absent while loading). Phases: loading, idle, listening, transcribing, thinking, preparing, speaking. |
| `quiet` | Nobody has spoken or typed, and Zen has said nothing, for two minutes. Sent once until something happens again. The page turns the microphone off, as a mute, unless the listener chose to keep it on. |
| `clear` | `generation`; cancel all queued audio, pending acknowledgements and partial input from earlier generations. Also emitted when a turn completes. |
| `transcript` | `generation`, `text`, `filtered: true`; the completed user turn: for speech, the transcript once the repair layer accepts it, and for a typed message, the message as typed. Raw recogniser output is never committed to the page. |
| `transcript_partial` | `generation`, `text`; an ordered ASR hypothesis. It can change, never commits history, and the page shows an understanding state rather than exposing unfiltered language. |
| `phrase_start` | `generation`, `phrase`, `text`; synthesis for this phrase has begun. This is not proof of audible playback. Show the phrase when its audio starts, and retain it only after playback completes. |
| `phrase_end` | `generation`, `phrase`, `text`, `pause_ms`; synthesis finished. Append `pause_ms` of silence to the stream after the phrase's last sample, and send `played` once that silence has been heard, so a message that arrives late cannot move the pause. |
| `notice` | `code`, `generation`. `partly_unheard`: part of the turn was clearly spoken but recognition returned no words for it even after a retry, so the reply answers only what was heard. When none of it was recognised, Zen asks for a repeat instead and no notice is sent. |
| `heard` | `generation`, `text`. A reply was cut off and kept up to that point: `text` is what was certainly heard of the phrase playing at the cut (possibly empty). Sent before the `clear` that cancels it, for a reply the conversation keeps. Show `text` and a dash after the phrases already completed, so the history matches what the model is given. |
| `timing` | `generation`, `total_ms` from the end of the question to the answer's first sound, and the stages inside it: `recognize_ms`, `repair_ms`, `think_ms` (request to first token), `speak_ms` (first token to first sound). A stage that did not run is `null`. `queue_ms` is the longest any piece of the question waited for the recogniser. Sent once per turn, when playback starts. Display only. |
| `history_cleared` | The conversation was emptied by `clear_history`; the next turn starts fresh under the instructions it carried. |
| `endpoint` | `ms`; what the segmenter is currently waiting before ending a turn. Sent when it changes, which in adaptive mode it does as the speaker is measured. Display only. |
| `error` | `code`, and on `backend_unavailable` a `detail` naming what failed — a missing model path, an occupied port. Operational text only, never private model or native error bodies. |

## Audio frames

Reply audio travels as raw bytes rather than base64 inside JSON, which costs a third more
bandwidth and a copy per block on the path carrying every sample Zen speaks. It shares the
one return channel with state, so a block and the events around it stay in the order the
engine put them in. Each frame is one block:

```
[generation u64 LE][phrase u64 LE][sequence u64 LE][pcm i16 LE ...]
```

PCM is mono at 24000 Hz. The 24-byte header is read with a `DataView`; the remainder is
queued for playback.

Phrase and sequence IDs increase throughout a session. Generations increase on
cancellation and completion. Handle `clear` immediately: audio queued for an abandoned
generation must never be acknowledged, because it was never heard.

The page resamples blocks continuously to the device rate and queues them in an AudioWorklet.
Acknowledgements follow the renderer's consumed sample count plus the device's output latency,
measured on the AudioContext clock. A suspended context and wall-clock timers alone cannot
prove playback. The final phrase can complete after the renderer has drained without requiring
another audio block. A local interruption suppresses late audio until a new generation arrives.
The start of each reply is held for a playout delay learned from how late recent replies were
delivered, at most a second. Underruns fade to silence, deepen the bounded startup buffer, and
fade in when audio resumes.

## Bounds and errors

A frame from the page is capped at 64 KiB and the engine's input queue holds 256 messages.
The page keeps one call in flight and queues behind it, shedding old capture beyond about a
second. The microphone is stopped only if the queue cannot drain at all, never for ordinary
jitter.

Roughly two seconds of PCM may await acknowledgement. Playback that stops acknowledging
audio for fifteen seconds cancels the turn. An open microphone utterance with no input for
three seconds is cancelled. Utterances are limited to three minutes and 128 capture
segments. Phrase staging is capped at 128 phrases beyond the bounded worker queue. Model
output and UTF-8 chunks have independent byte limits.

Session error: `backend_unavailable`. Turn errors: `invalid_input`, `recognition_failed`,
`recognition_incomplete`, `model_failed`, `synthesis_failed`, `synthesis_backpressure`,
`invalid_synthesis_audio`, `transcript_too_long`, `turn_timeout`, `audio_stalled`,
`playback_stalled`. A turn error clears pending playback and allows another turn; only a
session error ends the session. Instructions that break the limits are refused where they
arrive: `zen_attach` rejects its call, and `clear_history` fails as `invalid_input`.

`recognition_incomplete` means recognition did not finish within its deadline. The partial
hypothesis is not committed as a complete question. Audio the recogniser cannot take - it has
fallen behind, or the question has outgrown one turn - is not an error: the turn ends there
and is answered from what was heard. Transcript repair failures
fall back to the complete raw transcript only when their generation and input revision match.
Resuming speech while repair runs retains the earlier recognized prefix of that utterance.

A model or worker startup failure ends the session. A recoverable turn failure clears
pending playback and allows another turn once the state returns to listening. An explicit
disconnect clears even a failed session.
