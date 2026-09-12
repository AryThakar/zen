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

## Obtained separately at install time

None of these are included in this repository. See [docs/SETUP.md](docs/SETUP.md) for where
each file goes.

| Component | Use in Zen | License (as published by its authors) |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` and ggml) | Language model server | MIT |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) (`crispasr.dll`) | Speech recognition runtime | MIT |
| [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) (`qwen.dll`) | Speech synthesis runtime | MIT |
| [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF) (Unsloth QAT GGUF and MTP drafter) | Transcript repair and replies | Apache-2.0, as listed on the model page |
| [Qwen3-ASR 1.7B](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) (GGUF) | Speech recognition model | Apache-2.0, as listed on the model page |
| [Qwen3-TTS 12Hz 0.6B Base](https://huggingface.co/CC-TM/Qwen3-TTS-GGUF) and its 12 Hz tokenizer (GGUF) | Speech synthesis model and codec | Apache-2.0, as listed on the model page |
| NVIDIA CUDA runtime and cuBLAS DLLs | GPU acceleration for llama.cpp and TTS | NVIDIA CUDA Toolkit EULA |

Check each project's current license before redistributing any of these files.
