<div align="center">

<img src="icons/128x128.png" width="96" alt="Zen app icon" />

# Zen

**A private voice assistant that runs entirely on your own PC.**

Speech recognition, a language model and streaming speech synthesis, all local, behind one
desktop window. Nothing you say, hear or type leaves the machine or is written to disk.

![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-0078D4)
![Rust 1.91+](https://img.shields.io/badge/rust-1.91%2B-CE422B)
![Tauri 2](https://img.shields.io/badge/Tauri-2-24C8DB)
![GPU: 4 GB VRAM](https://img.shields.io/badge/GPU-4%20GB%20VRAM-76B900)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

<img src="docs/screenshots/speaking-light.png" width="860" alt="Zen speaking a reply: the glass orb deforms with the reply audio and the caption shows the phrase being spoken" />

<sub>Screenshots are of the running app. Every reply shown was generated and spoken live by the local models.</sub>

</div>

## What it is

Zen is a desktop voice assistant you talk to like a person. Press **Start talking** and speak:
Zen listens, works out when you have finished, answers out loud, and stops the moment you cut
in. You can type instead whenever you like.

Everything runs on your own GPU and CPU:

| Stage | Model | Runtime | Runs on |
| --- | --- | --- | --- |
| Voice activity | Silero VAD v6 | ONNX Runtime, in process | CPU |
| Speech to text | Qwen3-ASR 1.7B (Q4_K) | CrispASR, isolated worker process | CPU |
| Transcript repair and reply | Gemma 4 E2B (QAT, UD-Q4_K_XL) with an MTP drafter | llama.cpp `llama-server`, supervised | GPU |
| Text to speech | Qwen3-TTS 12Hz 0.6B Base (Q4_K_M), cloned from a reference clip | qwentts.cpp, isolated worker process | GPU |

The application is Rust on [Tauri 2](https://tauri.app/). The interface is hand-written HTML,
CSS and JavaScript in WebView2, with no framework or bundler, and is compiled into the
executable.

## Highlights

- **Private by construction.** No cloud APIs, account or telemetry. The model server is bound
  to loopback (enforced in code), the speech workers use authenticated loopback sockets, and
  audio, transcripts and prompts are never written to disk.
- **Natural turn-taking.** Interrupt Zen at any point. Confirmed speech, not just a loud noise,
  stops the reply, and history keeps only the words you actually heard, so the model never
  believes it told you something you missed.
- **An end-of-turn pause that learns you.** A fixed silence threshold always cuts somebody off.
  Zen notices when you carry on straight after being cut off, waits longer next time, and
  relaxes again over clean turns, within 0.6–2.0 s.
- **Transcript repair before answering.** A dedicated model slot fixes misrecognised words and
  translates non-English speech into English. A guard rejects "corrections" that invent words
  you never said, and a repair that runs out of budget falls back to the raw transcript.
- **Streaming speech.** Replies are synthesised phrase by phrase and start playing about 0.6 s
  after synthesis begins. A jitter buffer, seam crossfades and an envelope-following limiter
  keep playback clean.
- **Fits a 4 GB laptop GPU.** One `llama-server` with two fixed slots, a stateless filter and a
  stateful talker with a cached prompt prefix, plus Multi-Token Prediction drafting and a q4_0
  KV cache. The whole stack, speech synthesis included, runs in about 3.6 GB of VRAM.
- **Hard to wedge.** ASR and TTS run in separate processes so their GGML builds cannot collide,
  and a crash there fails one turn rather than the app. A Windows job object kills every child
  process if Zen dies, so a crash never leaves the GPU memory and port held.
- **A calm interface.** Light and night themes, reduced-motion support, live captions, and a
  WebGL glass orb that deforms with the reply audio and swells with your voice.

## Performance

Measured on 11 September 2026 with the release build, on an NVIDIA RTX 3050 Laptop GPU (4 GB),
Intel Core i5-12450H and 16 GB RAM, Windows 11.

| Measurement | Result | Source |
| --- | --- | --- |
| LLM decode speed, talker slot with MTP | 81–97 tokens/s over three 100+ token replies | llama-server `timings`, production flags |
| LLM prompt processing, ~20-token prompt | 25–53 ms | same |
| VRAM, LLM alone (two 8k slots + drafter) | 1.6 GB | `nvidia-smi` |
| VRAM, live session (LLM + TTS) | 3.6 GB of 4 GB | `nvidia-smi` |
| TTS time to first audio | 622 ms | `zen.exe --self-test` |
| TTS speed | 2.80 s of speech in 1.30 s | `zen.exe --self-test` |
| TTS cancellation | acknowledged in 4 ms | `zen.exe --self-test` |
| ASR, 2.8 s utterance | 1.73 s | `zen.exe --self-test` |
| Typed message to Zen speaking, warm session | ~1.2 s (1177 ms, 1192 ms) | scripted session in the real window |

The first turn after launch also loads the models, which took about 14 s on this machine.

## How it works

```mermaid
flowchart TB
    subgraph W["Window (WebView2)"]
        MIC["Microphone<br/>echo cancel, noise, gain"] --> WL["AudioWorklet<br/>20 ms PCM at 16 kHz"]
        PB["Playback scheduler<br/>jitter buffer, limiter"]
    end
    subgraph E["Rust engine (same process)"]
        VAD["Silero VAD<br/>+ segmenter"] --> ASR["Qwen3-ASR<br/>worker process"]
        ASR --> FIL["Filter slot<br/>repair, translate"]
        FIL --> TALK["Talker slot<br/>Gemma 4 E2B"]
        TALK --> CH["Phrase chunker"]
        CH --> TTS["Qwen3-TTS<br/>worker process"]
    end
    WL -->|"ordered IPC"| VAD
    TTS -->|"24 kHz PCM"| PB
    PB -->|"playback acks"| CH
```

The webview owns the microphone and speaker, so capture gets Chromium's echo cancellation,
noise suppression and gain control; the Rust side receives finished 16 kHz audio and owns
everything else. A turn moves through `idle → listening → transcribing → thinking → preparing
→ speaking`, and every asynchronous job carries the generation it was issued under, so work
that finishes after an interruption is dropped instead of played.

The page acknowledges each block of audio only once it has actually played. Those
acknowledgements drive backpressure and decide what enters the conversation history.

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the full design and the reasoning behind it
- [PROTOCOL.md](PROTOCOL.md): the IPC contract between the interface and the engine

## Screenshots

| Ready to talk | Night theme, speaking |
| --- | --- |
| <img src="docs/screenshots/home-light.png" alt="Zen home screen before a session: the orb, Start talking button and a text box" /> | <img src="docs/screenshots/speaking-dark.png" alt="Zen in the night theme while speaking a reply" /> |
| **Conversation history** | **Settings** |
| <img src="docs/screenshots/conversation.png" alt="The conversation panel listing two exchanges between the user and Zen" /> | <img src="docs/screenshots/settings.png" alt="Settings: conversation instructions, theme and listening pause" /> |

## Requirements

- **Windows 11** (WebView2 is part of the OS). Windows is the only supported platform today.
- **NVIDIA GPU with 4 GB of VRAM** and a driver that supports CUDA 13, the version of the
  bundled CUDA runtime DLLs
- **16 GB RAM** and about **5.8 GB of disk** for the models and native libraries
- To build: **Rust 1.91+** (MSVC toolchain). **Node.js 22+** only for the interface tests.

## Getting started

### 1. Build

```powershell
git clone <repository-url> zen
cd zen
cargo build --release --locked
```

### 2. Install the models and native libraries

The weights and native runtimes (about 5.8 GB) are not part of this repository. Zen looks for
them in an **installation root**:

```text
<root>\
├── Zen.exe                  the built target\release\zen.exe, renamed
├── bin\                     llama.cpp CUDA build and CUDA runtime DLLs
├── lib\                     qwen.dll (qwentts.cpp) and its ggml DLLs
└── model\
    ├── E2B\                 Gemma 4 E2B GGUF and MTP drafter
    ├── Qwen ASR\            Qwen3-ASR GGUF, crispasr.dll and CPU ggml DLLs
    ├── Qwen TTS\            Qwen3-TTS talker and tokenizer GGUFs, your voice reference
    └── VAD\                 silero_vad_v6.onnx
```

[docs/SETUP.md](docs/SETUP.md) lists every file, where to get it, and the checksums of the
exact files Zen was tested with.

### 3. Run

Copy `target\release\zen.exe` into the root as `Zen.exe` and double-click it. There is no
installer, service or first-run setup. Zen finds the root beside the executable or in any
parent directory, so a clone inside the root also runs in place. Otherwise pass `--root PATH`
or set `ZEN_ROOT`.

Check the installation from a terminal first:

```powershell
.\Zen.exe --self-test
```

This loads the real models without opening a window, then checks synthesis, recognition,
conversation memory, both prompts and cancellation. A healthy install ends with
`Self-test passed`. If a model or library is missing, Zen names the missing file.

## Using Zen

- **Talk.** Press **Start talking**, allow the microphone, and speak normally. Pause when you
  are done; Zen answers out loud. Speak over it to interrupt.
- **Type.** Write in the box under the controls. Typing interrupts whatever Zen is doing.
- **Stop, sound cues, history.** The stop button ends the current reply, the speaker toggles
  the soft interface sounds, and the speech bubble opens the conversation so far.
- **Settings.** Per-conversation instructions, light or night theme, the listening pause
  (automatic, or set it yourself), and microphone and speaker selection.
- **Closing the window keeps Zen running** in the tray with the models loaded. Quit from the
  tray icon. Launching Zen again brings the existing window forward instead of starting a
  second engine.
- **End session** (the power button) releases the models and all conversation state.

Zen's default persona lives in [`src/prompts/talker.txt`](src/prompts/talker.txt) and is
compiled in. It speaks as a warm, direct woman, always replies in English, and says plainly
that it cannot set timers, browse or control devices rather than pretending to. The shipped
prompt addresses its user by name (the author's), and
[`src/prompts/filter.txt`](src/prompts/filter.txt) expects that name when repairing
transcripts. Change the name in both files for your own build, or use **Conversation
instructions** in Settings for a single session.

### Command-line options

| Option | Default / behaviour |
| --- | --- |
| `--root PATH` | Installation root holding `bin`, `lib` and `model`; found beside the executable by default |
| `--system-prompt TEXT` | Override the default talker instructions, up to 8192 UTF-8 bytes |
| `--system-prompt-file PATH` | Read the instructions from a UTF-8 file |
| `--endpoint-ms N` | Fix the end-of-turn pause, 450–2000 ms. Omitted, Zen learns it from how you pause |
| `--reply-tokens N` | Reply budget, 64–1024 (default 512), also reserved in the context budget |
| `--gain N` | Output gain, 0.5–4.0 (default 1.8), soft-limited |
| `--no-filter` | Skip transcript repair |
| `--run-for-seconds N` | Quit after N seconds |
| `--self-test` | Exercise native ASR, LLM, repair, TTS and cancellation without a window |

Your shell may keep command-line prompts in its history. Instructions typed into Settings are
held only in memory.

## Privacy

- No network access beyond the local machine, and no account.
- Nothing said, heard or typed is written to disk. Native library output goes to the null
  device, and llama-server's disk KV cache is disabled.
- The window keeps the last forty displayed messages in memory. Ending the session or quitting
  clears them, along with the model KV caches.
- The theme and the listening pause are the only things remembered between launches.

This is an application policy, not secure erasure of OS swap, crash dumps or backups made by
other tools. See [SECURITY.md](SECURITY.md) for what Zen exposes.

## Development

```powershell
cargo fmt --all -- --check
cargo clippy --release --locked --all-targets -- -D warnings
cargo test --release --locked --all-targets     # 236 tests, no models needed
npm ci
npm test                                        # interface unit tests
npx playwright install chromium                 # once, for the UI tests
npm run test:ui                                 # interface in headless Chromium
```

CI runs all of the above on every push and pull request. The tests cover turn races,
interruption, ordered playback and acknowledgements, cancellation, chunk planning, Unicode
and coalesced text, context-budget eviction, revision fencing across a page reload, and
session revocation.

Some things cannot be tested without the models: whether the repair prompt still transcribes
instead of answering, or whether synthesis resumes after a cancellation. `zen.exe --self-test`
checks those against the real models, and `npm run test:native` drives the real window over
WebView2's debugging port. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Project layout

```text
src/
├── bin/zen.rs          entry point
├── runtime.rs          CLI, native-worker entry, self-test
├── app.rs              window, tray, notifications, IPC commands, microphone permission
├── bridge.rs           engine options and the revision-fenced page attachment
├── remote.rs           session runner for the webview-owned audio path
├── session.rs          turn orchestration and generation fencing
├── turn.rs             turn state machine and timeouts
├── audio.rs            Silero VAD, segmenter, adaptive endpoint
├── asr.rs, input.rs    recognition and utterance assembly
├── conversation.rs     rolling history of what was actually heard
├── reply.rs            transcript repair protocol, speakable text, phrase chunking
├── voice.rs, tts.rs    cancellable synthesis
├── resample.rs         band-limited rate conversion
├── engine.rs           llama-server supervision, slots and streaming client
├── native.rs, job.rs   isolated worker processes and the job object
├── prompts/            talker and filter instructions, compiled in
└── client/             interface, microphone worklet, playback scheduler, orb shader
tests/                  Rust flow tests and Node interface tests
```

## Limitations

- Windows and NVIDIA only. The native libraries are Windows CUDA builds, and process
  containment uses Windows job objects.
- The models and native libraries are installed by hand; no packaged release is published.
  `tauri.conf.json` is configured for an NSIS installer, but the tested distribution is the
  single executable.
- Replies are always in English, and Zen has no tools: it cannot browse, set timers or control
  devices, and says so.
- Interrupting over open speakers depends on Chromium's echo canceller and the room.
  Headphones give the most reliable interruption. Unit tests simulate playback and do not
  measure acoustics.

## Acknowledgements

Zen stands on [llama.cpp](https://github.com/ggml-org/llama.cpp),
[CrispASR](https://github.com/CrispStrobe/CrispASR),
[qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), Google's Gemma 4 (GGUF by
[Unsloth](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF)), the Qwen team's Qwen3-ASR
and Qwen3-TTS, [Silero VAD](https://github.com/snakers4/silero-vad),
[Tauri](https://tauri.app/), [`ort`](https://github.com/pykeio/ort) and the
[Inter](https://rsms.me/inter/) typeface. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Zen is licensed under the [Apache License 2.0](LICENSE). The models and native libraries it
runs are separate works under their own licenses.
