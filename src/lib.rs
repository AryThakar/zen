// SPDX-License-Identifier: Apache-2.0
//! Zen: a private voice assistant that runs speech recognition, a language model and
//! streaming speech synthesis on the local machine, behind one desktop window.
//!
//! The crate is the whole application. `src/bin/zen.rs` only hands control to
//! [`runtime::main_entry`]; everything else lives in these modules, grouped by the path a
//! spoken turn takes through them:
//!
//! - **Process and window** - [`runtime`] parses the command line and runs the self-test,
//!   [`app`] owns the window, tray and IPC commands, and [`bridge`] fences each attached page
//!   by revision so a reload cannot be credited with an older page's frames.
//! - **Listening** - [`audio`] runs Silero VAD and the segmenter over the 16 kHz stream the
//!   webview captures, [`asr`] recognises bounded chunks, and [`input`] assembles them into
//!   one utterance that is finalised exactly once.
//! - **Deciding** - [`session`] and [`turn`] own the turn state machine and its cancellation
//!   fences, [`conversation`] holds the rolling history of what was actually heard, and
//!   [`reply`] repairs transcripts and cuts replies into speakable phrases.
//! - **Speaking** - [`voice`] runs cancellable synthesis on a worker thread over [`tts`],
//!   with [`resample`] for rate conversion.
//! - **Native processes** - [`engine`] supervises `llama-server` and its two slots,
//!   [`native`] isolates the ASR and TTS libraries in worker processes, and [`job`] ties every
//!   child to this process's lifetime.
//!
//! `PROTOCOL.md` at the repository root specifies the IPC contract between the interface in
//! `src/client` and [`bridge`].
pub mod app;
pub mod asr;
pub mod audio;
pub mod bridge;
pub mod conversation;
pub mod engine;
pub mod input;
pub mod job;
pub mod native;
pub(crate) mod remote;
pub mod reply;
pub mod resample;
pub mod runtime;
pub mod session;
pub mod tts;
pub mod turn;
pub mod voice;
