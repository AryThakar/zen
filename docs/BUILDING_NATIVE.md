# Building the native libraries

[SETUP.md](SETUP.md) tells you where each native file goes and which prebuilt download to
take. This page is for when a prebuilt does not exist for you: a different GPU vendor, a
different operating system, recognition on the GPU rather than the processor, or simply a
llama.cpp newer than the release this was pinned to.

All three native projects are MIT licensed, so you may build and redistribute them provided
you carry their copyright and license text. Zen's own source is separately licensed; see
[LICENSE](../LICENSE) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## What has actually been tested

Only one configuration has been run end to end for this repository. Everything else in this
page is upstream capability, linked so you can follow it, and is marked as unverified because
nobody has run it here.

| Configuration | Verified here |
| --- | --- |
| Windows x64, NVIDIA, CUDA 13.3 | Yes — `--self-test` passes, numbers in [SETUP.md](SETUP.md) |
| Windows x64, Vulkan (AMD, Intel, NVIDIA) | No |
| Windows x64, processor only | No |
| Linux, CUDA or ROCm | No |
| macOS, Metal | No |
| Recognition on the GPU | No — Zen drives CrispASR on the processor, see below |

Treat an unverified row as "the upstream project supports this and these are its flags", not
as a claim that Zen works there.

## The one thing that will cost you hours

ggml can be built two ways, and mixing them fails in a way that does not look like a build
problem.

- **Backends linked in.** `ggml.dll` imports `ggml-cpu.dll` and `ggml-cuda.dll`, so loading
  ggml loads the backends and they register themselves. This is `GGML_BACKEND_DL=OFF`, and it
  is how `qwen.dll` is built here.
- **Backends loaded at run time.** `ggml.dll` imports only `ggml-base.dll`. The backend
  libraries are separate and the *application* has to load and register them. This is
  `GGML_BACKEND_DL=ON`, and it is how the official llama.cpp Windows releases are built.
  `llama-server` does that loading itself.

`qwen.dll` does not do that loading. Point it at an official llama.cpp `ggml.dll` and
synthesis fails with `qt_init: backend_init failed (no GGML backend available)`.

Worse is a half-match. A directory holding one build's `ggml.dll` beside another build's
`ggml-cuda.dll` registers no backend at all, reports nothing, and quietly synthesises on the
processor. Measured here, that turned 3.2 s of speech from 1.5 s of work into 16.7 s.

Two rules follow, and Zen looks after the first as far as it can:

1. The four ggml files beside `qwen.dll` must come from **one build**, the one `qwen.dll` was
   compiled against. Zen prefers a directory that holds all four; otherwise it falls back to
   `bin\`.
2. The CUDA runtime (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`) is versioned
   rather than built against anything here, so one copy serves both the server and synthesis.
   Do not duplicate it — `cublasLt64_13.dll` alone is 463 MB.

Check what you built before shipping it:

```powershell
cuobjdump ggml-cuda.dll | Select-String "arch =" | Group-Object | Select-Object Count,Name
```

## Which GPUs a build covers

`qwen.dll` is 309 KB and contains no GPU kernels at all. Every kernel is in `ggml-cuda.dll`, so
that file alone decides which cards your build runs on. Rebuilding `qwen.dll` for a wider
range of hardware does nothing; rebuilding its ggml is what matters.

Measured with `cuobjdump`. Machine code runs immediately; PTX is compiled by the driver the
first time that card sees it, which costs a one-off delay at startup and is then cached.

| Build | Machine code | PTX |
| --- | --- | --- |
| llama.cpp official `b10930` cuda-13.3 | `sm_86 89 120a 121a` | `sm_75 80 90` |
| qwentts.cpp as released here | `sm_86 89 120a` | `sm_75 80 90` |

Between them that covers RTX 20 through RTX 50, plus A100 and H100.

PTX only works forwards. A card newer than every architecture in the file can still compile the
highest PTX in it and run; a card older than all of them cannot run at all. A build made only
for the machine it was compiled on is the usual cause of "it works here and nowhere else":
before this was widened, the CUDA backend beside `qwen.dll` carried `sm_86` alone, so
synthesis ran on RTX 30-series cards and no others.

## qwentts.cpp — `qwen.dll`

Upstream publishes **no binaries at all**, so this one is always built from source. The
Windows x64 CUDA 13 build is released with Zen so most people do not have to.

- Source: <https://github.com/ServeurpersoCom/qwentts.cpp>
- License: MIT, "The omnivoice.cpp authors"
- Built here from commit `c044c6f0`

The exact configuration behind the released bundle, with Visual Studio 2022's environment
already loaded:

```bat
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DQWEN_SHARED=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="75-virtual;80-virtual;86-real;89-real;90-virtual;120-real"
cmake --build . --config Release
```

`GGML_BACKEND_DL` is left at its default of `OFF`, which is what makes the backends load with
ggml. `QWEN_SHARED=ON` is what produces a DLL rather than a static library; without it there is
nothing for Zen to load. `-real` embeds machine code for that architecture and `-virtual`
embeds PTX, which is exactly the split measured in the table above.

Take `qwen.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll` and `ggml-cuda.dll` from the build
directory into `lib\`. All five, from the same build.

For another backend, replace `-DGGML_CUDA=ON`. The repository carries `buildvulkan.cmd`,
`buildcuda.cmd` and `buildall.cmd` for Windows, and `buildcpu.sh`, `buildsycl.sh`,
`buildvulkan.sh` and `buildtermux.sh` for other platforms. None of those are verified here.

## CrispASR — `crispasr.dll`

- Source: <https://github.com/CrispStrobe/CrispASR>
- License: MIT, "The ggml authors"
- Prebuilt: yes, `libcrispasr-windows-x86_64.tar.gz` on the
  [releases page](https://github.com/CrispStrobe/CrispASR/releases)

Zen calls the session C API (`crispasr_session_open`, `crispasr_session_transcribe` and
related), so it needs the **shared library**, not the command-line tool. The
`crispasr-windows-*.zip` assets contain `crispasr.exe` only and are no use here; the
`libcrispasr-*` assets contain `bin\crispasr.dll` with the ggml set it was built against.

**Recognition runs on the processor.** Zen preloads only `ggml-base.dll`, `ggml-cpu.dll` and
`ggml.dll` for it (`src/asr.rs`), and the ASR `ggml.dll` imports no CUDA at all. So take the
50 MB processor build rather than the 483 MB CUDA one, and do not place a `ggml-cuda.dll` in
`model\Qwen ASR\` — it will never be loaded.

Moving recognition onto the GPU means building CrispASR with a GPU backend *and* teaching
`src/asr.rs` to preload that backend alongside the three it loads now. The first half is
upstream and straightforward; the second is a change to Zen and is not done.

## llama.cpp — `llama-server` and its DLLs

- Source: <https://github.com/ggml-org/llama.cpp>
- License: MIT, "The ggml authors"
- Prebuilt: yes, `llama-<build>-bin-win-cuda-13.3-x64.zip` plus
  `cudart-llama-bin-win-cuda-13.3-x64.zip` on the
  [releases page](https://github.com/ggml-org/llama.cpp/releases)

The official Windows CUDA builds work as-is and cover seven GPU architectures, which is wider
than a typical local build. Use them unless you need a backend they do not publish.

Any build must be recent enough to support Gemma 4 Multi-Token Prediction drafting. Check
before installing:

```powershell
.\llama-server.exe --help | Select-String "draft-mtp"
```

If that prints nothing, the build is too old and Zen's `--spec-type draft-mtp` will be
rejected. The flags Zen passes are listed in [SETUP.md](SETUP.md) and come from
`LlamaConfig::launch_args` in `src/engine.rs`.

Building it yourself, for a backend with no official release:

```bat
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON
```

Substitute `-DGGML_CUDA=ON`, `-DGGML_HIP=ON` for ROCm, `-DGGML_SYCL=ON` for Intel, or
`-DGGML_METAL=ON` on macOS. Upstream's
[build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) is the
authority. Note that `llama-server` runs as a separate process from Zen, so its ggml has no
bearing on `qwen.dll`'s — the two never share an address space.

## Silero VAD

Nothing to build. The model is compiled into `zen.exe` from `src/silero_vad.onnx` and runs on
ONNX Runtime through the `ort` crate, on the processor.

- Source: <https://github.com/snakers4/silero-vad>
- License: MIT

## Checking what you built

From the installation root, with Zen closed:

```powershell
.\Zen.exe --self-test
```

On the reference machine (RTX 3050 Laptop 4 GB, Core i5-12450H, 16 GB RAM) synthesis returns
its first chunk in about 600 ms and produces roughly three seconds of speech in about a second
and a half, and recognition of the same clip takes about 1.6 s. Synthesis running on the
processor instead is unmistakable: the same work takes ten to twenty times longer. If your
numbers look like that, the ggml set beside `qwen.dll` is not matched — re-read the first
section.
