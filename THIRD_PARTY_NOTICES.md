# Third-party notices

Zen's own source code is copyright AryThakar and licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). It builds on the work below, which remains
under its authors' licenses and is not affected by Zen's license.

## Compiled into `zen.exe`

| Component | Use in Zen | License |
| --- | --- | --- |
| [Inter](https://rsms.me/inter/) typeface | Interface font, `src/client/inter.woff2` | SIL Open Font License 1.1 — full text in [`src/client/inter-LICENSE.txt`](src/client/inter-LICENSE.txt) |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection, `src/silero_vad.onnx` | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), via the [`ort`](https://github.com/pykeio/ort) crate | Runs the Silero VAD model | MIT |
| [Tauri](https://github.com/tauri-apps/tauri) and the other Rust crates in `Cargo.lock` | Window, IPC, async runtime, HTTP, audio resampling | Each crate's own license, predominantly MIT and/or Apache-2.0 |

## Redistributed with Zen's releases

Not in this repository, but attached to the
[releases page](https://github.com/AryThakar/zen/releases) as a convenience, because upstream
publishes no binaries for it.

| Component | Use in Zen | License |
| --- | --- | --- |
| [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) and the ggml build beside it, as `qwentts-runtime-windows-x64-cuda13.zip` | Speech synthesis runtime | MIT — Copyright (c) 2023-2026 The omnivoice.cpp authors. Full text ships inside the archive as `LICENSE-qwentts.cpp.txt` |

Built from source, unmodified, with the configuration recorded in
[docs/BUILDING_NATIVE.md](docs/BUILDING_NATIVE.md).

## Obtained separately at install time

None of these are included in this repository or in its releases. See
[docs/SETUP.md](docs/SETUP.md) for where each file goes.

| Component | Use in Zen | License (as published by its authors) |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` and ggml) | Language model server | MIT |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) (`crispasr.dll`) | Speech recognition runtime | MIT |
| [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF) (Unsloth QAT GGUF and MTP drafter) | Transcript repair and replies | Apache-2.0, as listed on the model page |
| [Qwen3-ASR 1.7B](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) (GGUF) | Speech recognition model | Apache-2.0, as listed on the model page |
| [Qwen3-TTS 12Hz 0.6B Base](https://huggingface.co/CC-TM/Qwen3-TTS-GGUF) and its 12 Hz tokenizer (GGUF) | Speech synthesis model and codec | Apache-2.0, as listed on the model page |
| NVIDIA CUDA runtime and cuBLAS DLLs | GPU acceleration for llama.cpp and TTS | NVIDIA CUDA Toolkit EULA |

Check each project's current license before redistributing any of these files. If you build
any of them yourself, [docs/BUILDING_NATIVE.md](docs/BUILDING_NATIVE.md) records the sources,
versions and build flags used here.
