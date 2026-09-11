## What and why

<!-- What does this change, and what problem does it solve? -->

## How it was verified

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --release --locked --all-targets -- -D warnings`
- [ ] `cargo test --release --locked --all-targets`
- [ ] `npm test` and `npm run test:ui`
- [ ] `zen.exe --self-test` on a machine with the model assets (for engine, prompt or audio changes)
- [ ] Tried in a live spoken session (for anything touching turn-taking, audio or the interface)

<!-- For behaviour changes, describe what you observed in the running app, not only the tests. -->
