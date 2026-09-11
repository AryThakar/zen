# Security

## Reporting a vulnerability

Please report security problems privately, through GitHub's private vulnerability reporting
for this repository (**Security → Report a vulnerability**), rather than in a public issue.
Include what you found, how to reproduce it, and which commit you tested.

Only the latest commit on `main` is supported.

## What Zen exposes

Zen is designed to have no network surface beyond the local machine. When reviewing a change,
these are the properties it is meant to keep:

- **No remote services.** Zen makes no outbound requests. Speech recognition, the language
  model and speech synthesis all run locally.
- **Loopback only.** The supervised `llama-server` binds to `127.0.0.1:8740`; the engine
  refuses any configuration that is not a loopback address, and refuses to adopt an unrelated
  process already on that port. The ASR and TTS worker processes connect back to the engine
  over authenticated loopback sockets on ephemeral ports.
- **Interface isolation.** The interface is compiled into the executable and served from
  Tauri's private app origin under a restrictive Content Security Policy (`tauri.conf.json`):
  no remote scripts, styles or connections. Its only capabilities are the core Tauri
  defaults, notifications, and Zen's own engine commands (`capabilities/default.json`).
- **Validated input.** Every frame from the page is size-capped and revision-fenced; control
  messages with an unknown type or field are rejected. See [PROTOCOL.md](PROTOCOL.md).
- **No conversation data at rest.** Audio, transcripts, prompts and native library output are
  not written to disk, and llama-server's disk KV save/restore is disabled. This is an
  application policy, not secure erasure: it does not cover OS swap, crash dumps or backups
  made by other tools.
- **Contained native code.** The ASR and TTS libraries run in separate processes, and every
  child process is placed in a Windows job object with `KILL_ON_JOB_CLOSE`, so none survive
  the application.

Model weights and native libraries are obtained separately (see
[docs/SETUP.md](docs/SETUP.md)); only install them from sources you trust, since they run with
your user's privileges.
