# Contributing to Zen

Thanks for your interest in Zen. This covers building, verifying and submitting changes.

## Prerequisites

- **Windows 11** with WebView2 (ships with the OS)
- **Rust 1.91+** on the MSVC toolchain, with `clippy` and `rustfmt`
- **Node.js 22+** for the interface tests
- For anything that touches the models or audio: an NVIDIA GPU with **4 GB VRAM**, **16 GB RAM**,
  and the runtime assets installed as described in [docs/SETUP.md](docs/SETUP.md)

## Building

```powershell
cargo build --release --locked
```

The binary is `target\release\zen.exe`. It finds the installation root (the directory holding
`bin`, `lib` and `model`) by looking beside itself and then upwards, so a clone inside the
installation root runs in place. Otherwise pass `--root PATH` or set `ZEN_ROOT`.

## Verifying a change

Everything below except the last two commands runs without the models, and is what CI runs:

```powershell
cargo fmt --all -- --check
cargo clippy --release --locked --all-targets -- -D warnings
cargo test --release --locked --all-targets
npm ci
npm test
npm run test:ui    # uses the installed Microsoft Edge
```

With the assets installed, also run the native checks:

```powershell
.\target\release\zen.exe --self-test
npm run test:native
```

`--self-test` loads the real ASR, LLM and TTS, and runs the shipped prompts against the real
model. It catches a class of defect no offline test can reach: a prompt edit that makes the
repair layer answer a question instead of transcribing it, or call a clear sentence
unintelligible. Run it after any change to the engine, the audio path or `src/prompts/`.

Tests simulate playback acknowledgements, so they say nothing about room acoustics, echo or
how interruption feels. For changes to turn-taking, audio or the interface, try the change in
a real spoken session and describe what you observed in the pull request.

## Code style

- Every `.rs` file starts with `// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`.
- Each module opens with a `//!` comment explaining its role and the reasoning behind its
  design.
- Comments explain **why**, not what. Prefer a clearer name or structure over a comment that
  restates the code.
- Named constants instead of magic numbers.
- `cargo fmt` and `cargo clippy -D warnings` pass with zero diagnostics.

### Prompts

`src/prompts/talker.txt` and `src/prompts/filter.txt` are compiled in with `include_str!`.
The talker prompt is the talker slot's cached prefix: any edit, even whitespace, costs a full
re-prefill on the next session, so keep edits deliberate.

## Submitting changes

1. Fork the repository and create a branch.
2. Keep commits focused, with messages that say why.
3. Run the checks above.
4. Open a pull request describing the change, its motivation, and how you verified it.

## License

Zen is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). By contributing,
you agree that your contribution is licensed under those same terms, and that AryThakar may
also license it, as part of Zen, under other terms, including for commercial use.
