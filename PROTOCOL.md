# Engine IPC contract

Zen's interface and its engine are one process. The window's webview owns capture and
playback; the Rust side owns VAD, recognition, generation and synthesis. They talk over
Tauri's IPC, so there is no listener, no port, no certificate and no pairing token — the
connection is the process itself, and nothing described here is reachable from another
machine.

This document is the contract between `src/client` and `src/bridge.rs`.

## Attaching

The page calls `zen_attach` with two channels and, optionally, instructions for this
session. The engine starts if it is not already running and returns a **revision**.

```js
const events = new Channel();   // JSON state and transcript
const audio = new Channel();    // raw reply PCM
const revision = await invoke("zen_attach", { events, audio, systemPrompt });
```

The revision is the fence. A reloaded page gets a new revision, and frames stamped with an
older one are dropped rather than credited to the new page. Models load on the first
attach, so `zen_attach` returns before the engine is usable; wait for `ready`.

Instructions are accepted only when a session starts. Changing them requires a fresh
session, and they are held in memory for that session alone.

## Commands

| Command | Payload | Meaning |
| --- | --- | --- |
| `zen_attach` | `{ events, audio, systemPrompt? }` | Bind this page to a session; returns the revision. Instructions are at most 8192 UTF-8 bytes with no NUL. |
| `zen_input` | raw bytes | The single ordered path for everything the page sends. See below. |
| `zen_detach` | `{ revision }` | Release this page without ending the session, so a reload does not discard the conversation. |
| `zen_disconnect` | — | End the session. Native workers and the model exit; no state survives into the next session. |

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
| `{"type":"tuning","endpoint_ms":1200}` | Fix how long a pause must run before the turn ends, 450-2000 ms. `null` or omitted hands the decision back to the segmenter, which learns it from the speaker. Applied to the live pipeline, so it costs no session restart. A fresh engine starts adaptive, so restate a fixed value after `ready`. |
| `{"type":"speech_hint"}` | Accepted for older pages and ignored. An energy hint cannot cancel a reply; confirmed engine VAD opens the utterance and clears old output while preserving capture pre-roll. |
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
| `ready` | Models loaded; input may begin. |
| `state` | `phase`, `generation` (absent while loading). Phases: loading, idle, listening, transcribing, thinking, preparing, speaking. |
| `clear` | `generation`; cancel all scheduled audio, pending acknowledgements and partial input from earlier generations. Also emitted when a turn completes. |
| `transcript` | `generation`, `text`, `filtered: true`; the completed user transcript after the repair layer accepts it. Raw recogniser output is never committed to the page. |
| `transcript_partial` | `generation`, `text`; an ordered ASR hypothesis. It can change, never commits history, and the page shows an understanding state rather than exposing unfiltered language. |
| `phrase_start` | `generation`, `phrase`, `text`; synthesis for this phrase has begun. This is not proof of audible playback. Show the phrase when its audio starts, and retain it only after playback completes. |
| `phrase_end` | `generation`, `phrase`, `text`, `pause_ms`; synthesis finished. Measure the trailing pause from the scheduled audio end before sending `played`, even if this message arrives later. |
| `endpoint` | `ms`; what the segmenter is currently waiting before ending a turn. Sent when it changes, which in adaptive mode it does as the speaker is measured. Display only. |
| `error` | `code`, and on `backend_unavailable` a `detail` naming what failed — a missing model path, an occupied port. Operational text only, never private model or native error bodies. |

## Audio channel

Reply audio travels as raw bytes on its own channel rather than base64 inside JSON, which
costs a third more bandwidth and a copy per block on the path carrying every sample Zen
speaks. Each message is one block:

```
[generation u64 LE][phrase u64 LE][sequence u64 LE][pcm i16 LE ...]
```

PCM is mono at 24000 Hz. The 24-byte header is read with a `DataView`; the remainder is
scheduled directly.

Phrase and sequence IDs increase throughout a session. Generations increase on
cancellation and completion. Handle `clear` immediately: a scheduled source's `onended`
callback triggered by **stopping** it must not acknowledge playback.

The page must play blocks in order, apply no additional duration-changing transformation,
and acknowledge only after real audio progress. It uses `getOutputTimestamp().contextTime`
and completed sources, falling back to the `AudioContext` clock minus output latency when
no device timestamp is available. A suspended context and wall-clock timers alone cannot
prove playback. A local interruption suppresses late audio until a new generation arrives.

## Bounds and errors

A frame from the page is capped at 64 KiB and the engine's input queue holds 256 messages.
The page keeps one call in flight and queues behind it, shedding old capture beyond about a
second. The microphone is stopped only if the queue cannot drain at all, never for ordinary
jitter.

Roughly two seconds of PCM may await acknowledgement. A stalled speaker cancels the turn
after fifteen seconds without progress. An open microphone utterance with no input for
three seconds is cancelled. Utterances are limited to three minutes and 128 capture
segments. Phrase staging is capped at 128 phrases beyond the bounded worker queue. Model
output and UTF-8 chunks have independent byte limits.

Session errors: `invalid_prompt`, `backend_unavailable`. Turn errors: `invalid_input`,
`recognition_failed`, `model_failed`, `synthesis_failed`, `synthesis_backpressure`,
`invalid_synthesis_audio`, `transcript_too_long`, `turn_timeout`, `audio_stalled`,
`playback_stalled`. A turn error clears pending playback and allows another turn; only a
session error ends the session.

A model or worker startup failure ends the session. A recoverable turn failure clears
pending playback and allows another turn once the state returns to listening. An explicit
disconnect clears even a failed session.
