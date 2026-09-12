# Installing the models and native libraries

Zen's source builds a single `zen.exe`. The language model, speech models and native runtimes
it drives are separate works under their own licenses, so they are not part of this
repository. This page lists every file Zen loads, where it goes, and where it comes from.

## Where Zen looks

Zen uses an **installation root**: a directory containing `bin`, `lib` and `model`. In order of
precedence:

1. `--root PATH` on the command line
2. The `ZEN_ROOT` environment variable
3. The executable's own directory, then each parent directory up to six levels, taking the
   first one that contains all three folders

The usual layout is `Zen.exe` sitting in the root. Because of the upward search, a clone of
this repository placed inside the root (`<root>\zen\`) runs straight from
`target\release\zen.exe` without copying anything.

## Layout

This is the exact set of files Zen loads, as tested:

```text
<root>\
├── Zen.exe
├── bin\
│   ├── llama-server.exe
│   ├── llama.dll  llama-common.dll  llama-server-impl.dll  mtmd.dll
│   ├── ggml.dll  ggml-base.dll  ggml-cpu.dll  ggml-cuda.dll
│   └── cudart64_13.dll  cublas64_13.dll  cublasLt64_13.dll
├── lib\
│   ├── qwen.dll
│   └── ggml.dll  ggml-base.dll  ggml-cpu.dll
└── model\
    ├── E2B\
    │   ├── gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
    │   └── mtp-gemma-4-E2B-it.gguf
    ├── Qwen ASR\
    │   ├── qwen3-asr-1.7b-q4_k.gguf
    │   └── crispasr.dll  ggml.dll  ggml-base.dll  ggml-cpu.dll  ggml-cuda.dll
    └── Qwen TTS\
        ├── qwen-talker-0.6b-base-Q4_K_M.gguf
        ├── qwen-tokenizer-12hz-Q4_K_M.gguf
        ├── user_ref_voice.wav
        └── user_ref_text.txt
```

Together these take about 5.8 GB.

## Models

| File | Size | Source |
| --- | --- | --- |
| `model\E2B\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | 2.62 GB | [unsloth/gemma-4-E2B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF) |
| `model\E2B\mtp-gemma-4-E2B-it.gguf` | 59 MB | same repository: the Multi-Token Prediction drafter |
| `model\Qwen ASR\qwen3-asr-1.7b-q4_k.gguf` | 1.49 GB | [cstr/qwen3-asr-1.7b-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) |
| `model\Qwen TTS\qwen-talker-0.6b-base-Q4_K_M.gguf` | 629 MB | [CC-TM/Qwen3-TTS-GGUF](https://huggingface.co/CC-TM/Qwen3-TTS-GGUF) |
| `model\Qwen TTS\qwen-tokenizer-12hz-Q4_K_M.gguf` | 255 MB | same repository |

Voice activity detection needs no download: Silero's export is compiled into `zen.exe`.

The SHA-256 checksums of the files Zen was tested with are below. For the five GGUF files these
match the checksums published on Hugging Face at the time of writing.

```text
e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889  gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
586f2460b909008640981ec34060aa864e03c144fbabfb3173c4335087e4aae0  mtp-gemma-4-E2B-it.gguf
ec197cef7ccc589fdcae1becc3f4a3de119d0a41e790b898b519b1a048dad8d4  qwen3-asr-1.7b-q4_k.gguf
4b468ec7b1f62b90ef4ca316c0aa57deadfd54b2cf9651703ea753cedaf04226  qwen-talker-0.6b-base-Q4_K_M.gguf
cf3788b4d50aaa665fb6e57c170396aae03a3555fea52d2b5d0cda902d658039  qwen-tokenizer-12hz-Q4_K_M.gguf
```

Check a download in PowerShell with `Get-FileHash <file> -Algorithm SHA256`.

### The voice reference

Qwen3-TTS Base clones a voice from a short reference recording, so Zen speaks in whatever voice
you give it. Supply two files in `model\Qwen TTS\`:

- `user_ref_voice.wav`: a short, clean recording of a single speaker. Zen accepts 16-bit
  integer or float WAV at any sample rate; stereo is mixed down to mono and the audio is
  resampled to 24 kHz.
- `user_ref_text.txt`: the exact words spoken in that recording, as UTF-8 text.

Both are required. Synthesis without the transcript still runs but sounds noticeably off, with
nothing to point at the cause. Only use a voice you have the right to use. The repository's
`.gitignore` excludes `*.wav` so a personal recording is not committed by accident.

## Native libraries

These are Windows x64 builds. Zen loads them at runtime (`libloading` for the DLLs, a child
process for `llama-server`), so nothing links against them at build time.

### `bin\`: llama.cpp with CUDA

A CUDA build of [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` with its
DLLs, and the CUDA 13 runtime and cuBLAS DLLs it needs. The build must be recent enough to
support Gemma 4 Multi-Token Prediction drafting (`--spec-type draft-mtp`). Zen was tested with
a build that reports `version: 1 (720d7fa)`.

Zen starts it with these flags (see `LlamaConfig::launch_args` in `src/engine.rs`):

```text
--jinja --no-mmproj --alias zen-e2b-8k-np2
--spec-draft-model model\E2B\mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-ngl 99 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
--fit off --n-gpu-layers 99 --ctx-size 16384 --parallel 2 --no-kv-unified --swa-full
--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --cont-batching
--batch-size 1024 --ubatch-size 512 --cache-prompt --cache-ram 0 --slots
--reasoning-format deepseek --reasoning off --reasoning-budget 0
--no-ui --host 127.0.0.1 --port 8740
```

Port 8740 must be free. Zen refuses to adopt an unrelated process already listening there.

### `model\Qwen ASR\`: CrispASR

`crispasr.dll`, the shared library from [CrispASR](https://github.com/CrispStrobe/CrispASR),
with the ggml DLLs it was built against beside it. Zen calls its session C API
(`crispasr_session_open`, `crispasr_session_transcribe` and related functions). Before loading
it, Zen points the DLL search path at this folder and preloads `ggml-base.dll`, `ggml-cpu.dll`
and `ggml.dll` from here, so these copies are bound rather than the llama.cpp ones in `bin\`.
Recognition runs on the CPU, using the number of cores minus four as threads, clamped to 2–8.

### `lib\`: qwentts.cpp

`qwen.dll`, the shared library from
[qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), built with `-DQWEN_SHARED=ON`
and CUDA. Zen uses its C API (`qt_init`, `qt_synthesize`, `qt_extract_voice_ref` and related
functions). No import library ships with the DLL, so it is loaded at runtime.

Windows looks for a DLL's dependencies one directory at a time, and `qwen.dll` needs both ggml
and CUDA. Zen therefore preloads `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` and
the ggml DLLs by absolute path before loading `qwen.dll`, from `bin\` when it contains
`ggml.dll`, otherwise from `lib\`. In the tested layout that is the CUDA ggml in `bin\`, so
`qwen.dll` must be built against a ggml compatible with the llama.cpp build there.

## Verifying the installation

With Zen closed, run the self-test from a terminal:

```powershell
.\Zen.exe --self-test
```

It needs no microphone or window. On the reference machine (RTX 3050 Laptop 4 GB, Core
i5-12450H, 16 GB RAM) it printed:

```text
Starting Zen from C:\zen-ai
Loading isolated speech recognition...
Loading isolated speech synthesis...
TTS: first chunk 622 ms, 2.80 s audio, 1.30 s wall
TTS delivery: largest chunk gap 148 ms, startup buffer needed 11 ms
ASR: 1728 ms, transcript: Hello, my name is Zen. I'm ready to help you.
Model: You said the blue drawer.
Talker: February has twenty-nine days in a leap year.
Talker: I can't set a timer for you, Arya. I can talk through anything with you instead.
Filter: "wut is the wether tooday" -> What is the weather today?
Filter: "can you turn on the kitchen lights" -> Can you turn on the kitchen lights?
Filter: "uh um uh" -> asked for a repeat: Sorry, I missed that - could you say it again?
Filter: "so i was going through the notes from the meeting yesterday and ... before friday" -> So I was going through the notes ... before Friday
TTS cancellation acknowledged in 4 ms
Self-test passed, including synthesis after interruption. Speaker acoustics and audible interruption timing require a live session.
```

(The long filter line is shortened here.) The model's wording varies from run to run; the
checks are on what a reply contains, not its exact text.

## Troubleshooting

- **"model is missing: …" or "llama-server executable is missing: …"**: the named file is not
  where Zen expects it. Check the layout above, or the root Zen reports in the first line of
  `--self-test`.
- **`failed to load qwen.dll` or `failed to load crispasr.dll`**: usually a missing dependency
  DLL. Check that the CUDA and ggml DLLs listed above are present.
- **Zen reports the port is occupied**: another process holds `127.0.0.1:8740`, often a
  `llama-server` left over from something else. End it and start Zen again. A `llama-server`
  started by Zen itself always exits with Zen, crash included.
- **Nothing happens on double-click**: Zen shows a dialog when it cannot start. If there is no
  dialog either, run `Zen.exe --self-test` from a terminal to see the error.
