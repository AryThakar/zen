// SPDX-License-Identifier: Apache-2.0
//! In-process bridge between the embedded webview and the shared Session state machine.
//!
//! The webview and the engine are one process, so nothing here is addressed, paired or
//! authenticated: there is no listener, no token and no resume window. What survives from
//! the network design is the fence. A reloaded webview must not have its old frames credited
//! to the new session, so each attachment takes a revision and stale input is dropped.
//!
//! The webview still owns capture and playback. Chromium applies echo cancellation, noise
//! suppression and gain control, and acknowledges what it actually played; the engine owns
//! VAD, recognition, generation and synthesis.
use serde::Deserialize;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio::sync::Notify;

pub(crate) type Error = Box<dyn std::error::Error + Send + Sync>;

/// Largest single frame accepted from the webview. Capture sends 20 ms of 16 kHz mono,
/// so this is three orders of magnitude of headroom over the expected 640 bytes.
pub(crate) const MAX_FRAME: usize = 65_536;

/// Bytes of little-endian metadata ahead of the PCM in an audio frame:
/// generation, phrase, sequence.
const AUDIO_HEADER: usize = 24;

/// Frame kinds on the single ordered path back to the page. Both directions use one
/// ordered path for the same reason: a phrase's audio must not arrive before the
/// `phrase_start` that announces it, and separate channels give no ordering between
/// themselves. Reply audio still travels as raw bytes rather than base64 in JSON.
const EVENT_FRAME: u8 = 1;
const AUDIO_OUT_FRAME: u8 = 0;

#[derive(Clone)]
pub struct EngineOptions {
    pub root: PathBuf,
    pub system_prompt: String,
    pub endpoint: crate::audio::EndpointPolicy,
    pub reply_tokens: usize,
    pub filter: bool,
    pub gain: f32,
    pub run_for: Option<Duration>,
}

impl EngineOptions {
    pub fn validate(&self) -> Result<(), Error> {
        validate_prompt(&self.system_prompt)?;
        if !(64..=1024).contains(&self.reply_tokens) || !(0.5..=4.0).contains(&self.gain) {
            return Err("invalid endpoint, reply budget or output gain".into());
        }
        Ok(())
    }
}

pub(crate) fn validate_prompt(prompt: &str) -> Result<(), &'static str> {
    if prompt.trim().is_empty() || prompt.len() > 8_192 || prompt.contains('\0') {
        return Err("system prompt must contain 1..8192 UTF-8 bytes and no NUL");
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Control {
    Text {
        text: String,
    },
    Interrupt {},
    SpeechHint {},
    EndAudio {},
    /// How long a pause has to run before Zen takes the floor. `null` hands the decision to
    /// the segmenter, which learns it from the speaker's own within-turn pauses. Applied to
    /// the live pipeline, because this is tuned while talking to it.
    Tuning {
        #[serde(default)]
        endpoint_ms: Option<usize>,
    },
    /// Empty the rolling window and start fresh without ending the session,
    /// optionally adopting new instructions at the same time.
    ClearHistory {
        #[serde(default)]
        system_prompt: Option<String>,
    },
    Played {
        generation: u64,
        phrase: u64,
    },
    PlaybackStarted {
        generation: u64,
        phrase: u64,
    },
    AudioPlayed {
        generation: u64,
        sequence: u64,
    },
}

pub(crate) enum Input {
    Control(Control),
    Audio(Vec<u8>),
}

/// Frame kinds on the single ordered path from the page.
const AUDIO_FRAME: u8 = 0;
const CONTROL_FRAME: u8 = 1;

/// Decode one frame from the page: `[revision u64 LE][kind u8][payload]`.
///
/// Capture and control share one command, and the page keeps one call in flight, because
/// order is load-bearing. IPC calls are independent promises that can be delivered in any
/// order — measured at four reorderings in three hundred frames, one of them jumping five
/// places. Five frames is a hundred milliseconds of speech: enough for an `end_audio` to
/// overtake the words it follows and clip them, and enough for an acknowledgement to
/// arrive before the block it acknowledges and fail the turn.
pub(crate) fn parse_input_frame(frame: &[u8]) -> Result<(u64, Input), &'static str> {
    if frame.len() < 9 || frame.len() > MAX_FRAME {
        return Err("input frame out of range");
    }
    let revision = u64::from_le_bytes(frame[..8].try_into().map_err(|_| "short frame")?);
    let payload = &frame[9..];
    let input = match frame[8] {
        AUDIO_FRAME => {
            if payload.is_empty() || payload.len() > 6400 || !payload.len().is_multiple_of(2) {
                return Err("capture must contain 1..3200 whole 16-bit samples");
            }
            Input::Audio(payload.to_vec())
        }
        CONTROL_FRAME => {
            Input::Control(serde_json::from_slice(payload).map_err(|_| "unreadable control")?)
        }
        _ => return Err("unknown input kind"),
    };
    Ok((revision, input))
}

/// The webview's single return path.
struct Attachment {
    frames: Channel<InvokeResponseBody>,
}

struct Inner {
    state: Mutex<TransportState>,
    changed: Notify,
}

#[derive(Default)]
struct TransportState {
    attachment: Option<Attachment>,
    revision: u64,
    revoked: bool,
}

// Revisions must also fence different sessions, not only reloads of one session.
static NEXT_REVISION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub(crate) struct Transport {
    inner: Arc<Inner>,
}

impl Transport {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(TransportState::default()),
                changed: Notify::new(),
            }),
        }
    }

    /// Bind a freshly loaded webview to this session and fence everything sent by the
    /// previous one. Returns the revision the webview must quote on every frame.
    pub fn attach(&self, frames: Channel<InvokeResponseBody>) -> u64 {
        let mut slot = self.lock();
        if slot.revoked {
            return slot.revision;
        }
        let revision = NEXT_REVISION.fetch_add(1, Ordering::Relaxed);
        slot.revision = revision;
        slot.attachment = Some(Attachment { frames });
        drop(slot);
        self.inner.changed.notify_waiters();
        revision
    }

    pub fn detach(&self, revision: u64) {
        let mut slot = self.lock();
        if slot.revision == revision {
            slot.attachment = None;
            self.inner.changed.notify_waiters();
        }
    }

    /// End the session for good. The engine unwinds, native workers exit and the model
    /// stops, so no conversation state survives into whatever attaches next.
    pub fn revoke(&self) {
        let mut slot = self.lock();
        slot.revoked = true;
        slot.attachment = None;
        self.inner.changed.notify_waiters();
    }

    /// `(revision, attached, expired)`, matching what the session loop polls.
    pub fn status(&self) -> (u64, bool, bool) {
        let slot = self.lock();
        (slot.revision, slot.attachment.is_some(), slot.revoked)
    }

    pub fn accepts(&self, revision: u64) -> bool {
        let (current, attached, expired) = self.status();
        revision == current && attached && !expired
    }

    pub fn send(&self, value: serde_json::Value) {
        let json = value.to_string();
        let mut frame = Vec::with_capacity(1 + json.len());
        frame.push(EVENT_FRAME);
        frame.extend_from_slice(json.as_bytes());
        self.emit(frame);
    }

    /// One reply-audio block: the kind byte, a fixed header the page reads with a
    /// DataView, then interleaved 16-bit PCM.
    pub fn send_audio(&self, generation: u64, phrase: u64, sequence: u64, pcm: &[u8]) {
        let mut frame = Vec::with_capacity(1 + AUDIO_HEADER + pcm.len());
        frame.push(AUDIO_OUT_FRAME);
        frame.extend_from_slice(&generation.to_le_bytes());
        frame.extend_from_slice(&phrase.to_le_bytes());
        frame.extend_from_slice(&sequence.to_le_bytes());
        frame.extend_from_slice(pcm);
        self.emit(frame);
    }

    fn emit(&self, frame: Vec<u8>) {
        let mut slot = self.lock();
        if slot
            .attachment
            .as_ref()
            .is_some_and(|a| a.frames.send(InvokeResponseBody::Raw(frame)).is_err())
        {
            // The webview is gone. Drop it rather than queue for a listener that will
            // never read, and let the session loop notice on its next poll.
            slot.attachment = None;
            self.inner.changed.notify_waiters();
        }
    }

    pub async fn until_expired(&self) {
        loop {
            // Register before testing the flag. `notify_waiters` only wakes waiters that
            // are already registered, so checking first would let a revocation land in the
            // gap and leave this waiting for a notification that has already been sent.
            let waiting = self.inner.changed.notified();
            tokio::pin!(waiting);
            waiting.as_mut().enable();
            if self.status().2 {
                return;
            }
            waiting.await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TransportState> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for Transport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reloaded_webview_fences_the_frames_of_the_previous_one() {
        let transport = Transport::new();
        assert!(!transport.accepts(0), "nothing is accepted before attach");
        let first = transport.attach(test_channel());
        assert!(transport.accepts(first));
        let second = transport.attach(test_channel());
        assert_ne!(first, second);
        assert!(!transport.accepts(first), "old frames must not be credited");
        assert!(transport.accepts(second));
    }

    #[test]
    fn revoking_expires_the_session_and_detaching_does_not() {
        let transport = Transport::new();
        let revision = transport.attach(test_channel());
        transport.detach(revision);
        assert_eq!(transport.status(), (revision, false, false));
        transport.revoke();
        assert!(transport.status().2);
    }

    fn frame(revision: u64, kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = revision.to_le_bytes().to_vec();
        frame.push(kind);
        frame.extend_from_slice(payload);
        frame
    }

    #[test]
    fn capture_and_control_share_one_ordered_frame_format() {
        let pcm = frame(7, AUDIO_FRAME, &[1, 2, 3, 4]);
        let (revision, input) = parse_input_frame(&pcm).expect("audio frame");
        assert_eq!(revision, 7);
        assert!(matches!(input, Input::Audio(bytes) if bytes == vec![1, 2, 3, 4]));

        let json = br#"{"type":"audio_played","generation":3,"sequence":9}"#;
        let (revision, input) = parse_input_frame(&frame(7, CONTROL_FRAME, json)).expect("control");
        assert_eq!(revision, 7);
        assert!(matches!(
            input,
            Input::Control(Control::AudioPlayed {
                generation: 3,
                sequence: 9
            })
        ));
    }

    #[test]
    fn malformed_frames_are_refused_rather_than_reshaped() {
        assert!(parse_input_frame(&[0; 8]).is_err(), "no kind byte");
        assert!(
            parse_input_frame(&frame(1, AUDIO_FRAME, &[1, 2, 3])).is_err(),
            "half a sample is not a sample"
        );
        assert!(
            parse_input_frame(&frame(1, 9, &[])).is_err(),
            "unknown kind"
        );
        assert!(
            parse_input_frame(&frame(1, CONTROL_FRAME, b"{\"type\":\"nope\"}")).is_err(),
            "unknown control type"
        );
        assert!(
            parse_input_frame(&frame(1, AUDIO_FRAME, &vec![0; MAX_FRAME])).is_err(),
            "oversized frame"
        );
    }

    #[test]
    fn the_listening_pause_is_carried_as_an_ordinary_control() {
        let json = br#"{"type":"tuning","endpoint_ms":900}"#;
        let (_, input) = parse_input_frame(&frame(1, CONTROL_FRAME, json)).expect("control");
        assert!(matches!(
            input,
            Input::Control(Control::Tuning {
                endpoint_ms: Some(900)
            })
        ));
        // Omitted or null hands the decision back to the segmenter, which learns it.
        for body in [
            &br#"{"type":"tuning"}"#[..],
            &br#"{"type":"tuning","endpoint_ms":null}"#[..],
        ] {
            let (_, input) = parse_input_frame(&frame(1, CONTROL_FRAME, body)).expect("control");
            assert!(matches!(
                input,
                Input::Control(Control::Tuning { endpoint_ms: None })
            ));
        }
    }

    #[test]
    fn prompts_are_bounded_and_reject_interior_nul() {
        assert!(validate_prompt("be brief").is_ok());
        assert!(validate_prompt("   ").is_err());
        assert!(validate_prompt("a\0b").is_err());
        assert!(validate_prompt(&"x".repeat(8_193)).is_err());
    }

    #[test]
    fn stale_capture_cannot_enter_a_new_session() {
        let old = Transport::new();
        let revision = old.attach(test_channel());
        old.revoke();
        old.attach(test_channel());
        assert!(!old.accepts(revision), "revocation is permanent");
        let new = Transport::new();
        assert_ne!(revision, new.attach(test_channel()));
        assert!(!new.accepts(revision));
    }

    #[test]
    fn empty_and_oversized_capture_are_rejected_at_the_boundary() {
        assert!(parse_input_frame(&frame(1, AUDIO_FRAME, &[])).is_err());
        assert!(parse_input_frame(&frame(1, AUDIO_FRAME, &vec![0; 6402])).is_err());
        assert!(parse_input_frame(&frame(1, AUDIO_FRAME, &vec![0; 6400])).is_ok());
        assert!(parse_input_frame(&frame(
            1,
            CONTROL_FRAME,
            br#"{"type":"interrupt","extra":true}"#
        ))
        .is_err());
    }

    fn test_channel() -> Channel<InvokeResponseBody> {
        Channel::new(|_| Ok(()))
    }
}
