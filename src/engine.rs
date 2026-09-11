// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Resident E2B inference engine: one llama-server process, two slots, one file.
//!
//! Owns the physical facts of running the model: which slot a role runs in, how each slot's KV
//! cache is treated, the child process, and the HTTP protocol against it.
//!
//! The contract:
//!
//! - There are exactly [`PARALLEL_SLOTS`] slots, and a [`SlotId`] outside that range cannot be
//!   constructed - not by arithmetic, not by deserialization, not from a `/slots` response.
//! - FILTER is always slot 0 and TALKER is always slot 1. Fixed, not scheduled.
//! - The two slots run opposite [`KvPolicy`] disciplines.
//! - Every layer is pinned to the device; startup fails loudly rather than falling back to CPU.
//! - Thinking is off at the server, not per request.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tokio::{
    process::{Child, Command},
    sync::Mutex as AsyncMutex,
};

// ============================================================================================
// Slots
// ============================================================================================

/// Parallel slots `llama-server` is launched with.
///
/// Changing this alone is not enough: [`SlotKind`] enumerates the roles, and the two must agree.
/// A test below fails if they ever disagree.
pub const PARALLEL_SLOTS: usize = 2;

/// Which of the two roles a request runs as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotKind {
    /// Stateless. One classification per utterance - endpointing, routing, screening - then the
    /// prompt is thrown away. Carries no conversation.
    Filter,
    /// Stateful. Holds the rolling conversation and speaks to the user.
    Talker,
}

impl SlotKind {
    pub const ALL: [SlotKind; PARALLEL_SLOTS] = [SlotKind::Filter, SlotKind::Talker];

    /// The slot this role always runs in.
    ///
    /// Fixed rather than assigned: the talker's KV cache is only worth keeping if the talker is
    /// always the same slot, and a filter call landing in slot 1 would evict the conversation.
    pub const fn slot(self) -> SlotId {
        match self {
            SlotKind::Filter => SlotId(0),
            SlotKind::Talker => SlotId(1),
        }
    }

    pub const fn kv_policy(self) -> KvPolicy {
        match self {
            SlotKind::Filter => KvPolicy::PerRequest,
            SlotKind::Talker => KvPolicy::StablePrefix,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            SlotKind::Filter => "filter",
            SlotKind::Talker => "talker",
        }
    }
}

/// A validated slot index in `0..PARALLEL_SLOTS`.
///
/// The inner field is private and there is no public constructor taking a number, so an
/// out-of-range slot index cannot be formed anywhere in the crate - including from deserialized
/// data, which goes through [`SlotId::from_index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct SlotId(usize);

impl SlotId {
    pub const FILTER: SlotId = SlotId(0);
    pub const TALKER: SlotId = SlotId(1);

    /// The only fallible way in. Used by `serde` and by any boundary parsing an external number,
    /// including `/slots` responses from llama-server.
    pub fn from_index(index: usize) -> Result<Self, SlotIdError> {
        if index < PARALLEL_SLOTS {
            Ok(SlotId(index))
        } else {
            Err(SlotIdError { index })
        }
    }

    pub const fn index(self) -> usize {
        self.0
    }

    pub const fn kind(self) -> SlotKind {
        // Exhaustive over PARALLEL_SLOTS = 2; the test below breaks if that changes.
        if self.0 == 0 {
            SlotKind::Filter
        } else {
            SlotKind::Talker
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("slot index {index} is out of range; llama-server is launched with {PARALLEL_SLOTS} parallel slots")]
pub struct SlotIdError {
    pub index: usize,
}

impl TryFrom<usize> for SlotId {
    type Error = SlotIdError;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        SlotId::from_index(value)
    }
}

impl From<SlotId> for usize {
    fn from(value: SlotId) -> Self {
        value.0
    }
}

impl std::fmt::Display for SlotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "slot{}({})", self.0, self.kind().as_str())
    }
}

/// How a slot treats its KV cache between requests.
///
/// The two policies are genuinely opposite. Encoded as a type rather than an `if slot == Talker`
/// scattered through call sites, so a new code path has to name the policy it is assuming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPolicy {
    /// TALKER. The prompt prefix must be byte-identical across turns so `llama-server` can reuse
    /// the cached prefix. A single changed byte near the front re-prefills the whole
    /// conversation, which on a voice pipeline is heard as a stall before the assistant speaks.
    StablePrefix,
    /// FILTER. Carries no conversation: one utterance, one verdict, and the utterance is
    /// thrown away afterwards. The fixed instruction in front of it is still a prefix and is
    /// still cached - what this policy means is that nothing about a request depends on what
    /// the previous one contained.
    PerRequest,
}

impl KvPolicy {
    pub const fn prefix_stability_matters(self) -> bool {
        matches!(self, KvPolicy::StablePrefix)
    }
}

// ============================================================================================
// Errors
// ============================================================================================

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid engine configuration: {0}")]
    InvalidConfig(String),
    #[error("configured llama-server endpoint is already responding but is not owned by this runtime: {0}")]
    EndpointOccupied(String),
    #[error("owned llama-server process exited during startup")]
    ProcessExited,
    #[error("operation timed out: {0}")]
    Timeout(String),
    #[error("llama-server protocol error: {0}")]
    Protocol(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

// ============================================================================================
// Configuration
// ============================================================================================

pub const MIN_TOKENS_PER_SLOT: usize = 2_048;
pub const MAX_TOKENS_PER_SLOT: usize = 131_072;

/// Layer count passed to `--n-gpu-layers` to mean "all of them".
///
/// Deliberately paired with `--fit off` below - auto-fit would be free to walk this number back
/// down, which is the opposite of forcing the pin.
pub const GPU_LAYERS_ALL: usize = 99;

const MAX_PROTOCOL_BODY_BYTES: usize = 8 * 1024 * 1024;

#[cfg(windows)]
const WINDOWS_CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub draft_model: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub model_alias: String,
    pub tokens_per_slot: usize,
    pub gpu_layers: usize,
    pub draft_gpu_layers: usize,
    pub draft_max_tokens: usize,
    pub batch_size: usize,
    pub micro_batch_size: usize,
    pub kv_type_k: String,
    pub kv_type_v: String,
    /// Keep the full sliding-window KV instead of pruning it.
    ///
    /// Load-bearing for the talker slot: with SWA pruning active, llama-server cannot reuse the
    /// cached prefix across turns and re-prefills the conversation on every reply. That cost is
    /// heard directly as a stall before the assistant speaks, so the VRAM is bought deliberately.
    pub swa_full: bool,
    /// Whether the model is allowed to emit thinking tokens.
    ///
    /// Off for a voice assistant, and enforced at the server rather than left to each request:
    /// reasoning tokens are decode time the listener hears as silence, and they spend the same
    /// budget the spoken reply needs.
    pub thinking: bool,
    pub startup_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl LlamaConfig {
    /// The measured production shape: E2B pinned to the GPU, MTP draft, 8k per slot, two slots.
    pub fn from_zen_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        let model_dir = root.join("model").join("E2B");
        Self {
            executable: root.join("bin").join("llama-server.exe"),
            model: model_dir.join("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
            draft_model: Some(model_dir.join("mtp-gemma-4-E2B-it.gguf")),
            host: "127.0.0.1".into(),
            port: 8_740,
            model_alias: "zen-e2b-8k-np2".into(),
            tokens_per_slot: 8_192,
            gpu_layers: GPU_LAYERS_ALL,
            draft_gpu_layers: GPU_LAYERS_ALL,
            draft_max_tokens: 3,
            batch_size: 1_024,
            micro_batch_size: 512,
            kv_type_k: "q4_0".into(),
            kv_type_v: "q4_0".into(),
            swa_full: true,
            thinking: false,
            startup_timeout: Duration::from_secs(240),
            shutdown_timeout: Duration::from_secs(10),
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.executable.is_file() {
            return Err(EngineError::InvalidConfig(format!(
                "llama-server executable is missing: {}",
                self.executable.display()
            )));
        }
        if !self.model.is_file() {
            return Err(EngineError::InvalidConfig(format!(
                "model is missing: {}",
                self.model.display()
            )));
        }
        if let Some(draft) = &self.draft_model {
            if !draft.is_file() {
                return Err(EngineError::InvalidConfig(format!(
                    "draft model is missing: {}",
                    draft.display()
                )));
            }
        }
        if !matches!(self.host.as_str(), "127.0.0.1" | "localhost" | "::1") {
            return Err(EngineError::InvalidConfig(
                "the private model runtime must bind to loopback".into(),
            ));
        }
        if self.port == 0 || self.model_alias.trim().is_empty() {
            return Err(EngineError::InvalidConfig(
                "port and model alias must be configured".into(),
            ));
        }
        if !(MIN_TOKENS_PER_SLOT..=MAX_TOKENS_PER_SLOT).contains(&self.tokens_per_slot) {
            return Err(EngineError::InvalidConfig(format!(
                "tokens_per_slot must be in {MIN_TOKENS_PER_SLOT}..={MAX_TOKENS_PER_SLOT}"
            )));
        }
        self.tokens_per_slot
            .checked_mul(PARALLEL_SLOTS)
            .ok_or_else(|| EngineError::InvalidConfig("total context size overflow".into()))?;
        if self.gpu_layers == 0 {
            return Err(EngineError::InvalidConfig(
                "gpu_layers must be positive; this runtime does not support CPU-only operation"
                    .into(),
            ));
        }
        if self.batch_size < PARALLEL_SLOTS || self.micro_batch_size == 0 {
            return Err(EngineError::InvalidConfig("invalid batch sizing".into()));
        }
        if self.micro_batch_size > self.batch_size {
            return Err(EngineError::InvalidConfig(
                "micro batch cannot exceed batch size".into(),
            ));
        }
        Ok(())
    }

    pub fn total_context_tokens(&self) -> usize {
        self.tokens_per_slot * PARALLEL_SLOTS
    }

    pub fn base_url(&self) -> String {
        let host = if self.host == "::1" {
            "[::1]"
        } else {
            &self.host
        };
        format!("http://{host}:{}", self.port)
    }

    pub fn launch_args(&self) -> Vec<OsString> {
        let mut args = Vec::new();
        macro_rules! push {
            ($value:expr) => {
                args.push(OsString::from($value));
            };
        }
        push!("--jinja");
        push!("--model");
        push!(self.model.as_os_str());
        push!("--alias");
        push!(&self.model_alias);
        push!("--no-mmproj");

        if let Some(draft) = &self.draft_model {
            push!("--spec-draft-model");
            push!(draft.as_os_str());
            push!("--spec-type");
            push!("draft-mtp");
            push!("--spec-draft-n-max");
            push!(self.draft_max_tokens.to_string());
            push!("--spec-draft-ngl");
            push!(self.draft_gpu_layers.to_string());
            push!("--spec-draft-type-k");
            push!(&self.kv_type_k);
            push!("--spec-draft-type-v");
            push!(&self.kv_type_v);
        }

        // Auto-fit is off on purpose: it is allowed to reduce the offloaded layer count, which
        // would quietly undo the `-ngl 99` pin this runtime depends on.
        push!("--fit");
        push!("off");
        push!("--n-gpu-layers");
        push!(self.gpu_layers.to_string());

        push!("--ctx-size");
        push!(self.total_context_tokens().to_string());
        push!("--parallel");
        push!(PARALLEL_SLOTS.to_string());
        push!("--no-kv-unified");
        if self.swa_full {
            push!("--swa-full");
        }
        push!("--flash-attn");
        push!("on");
        push!("--cache-type-k");
        push!(&self.kv_type_k);
        push!("--cache-type-v");
        push!(&self.kv_type_v);
        push!("--cont-batching");
        push!("--batch-size");
        push!(self.batch_size.to_string());
        push!("--ubatch-size");
        push!(self.micro_batch_size.to_string());
        push!("--cache-prompt");
        push!("--cache-ram");
        push!("0");
        push!("--slots");
        push!("--reasoning-format");
        push!("deepseek");
        // Belt and braces: `--reasoning off` refuses thinking, and a zero budget ends it
        // immediately even if a template or request tries to turn it back on.
        push!("--reasoning");
        push!(if self.thinking { "on" } else { "off" });
        if !self.thinking {
            push!("--reasoning-budget");
            push!("0");
        }
        push!("--no-ui");
        push!("--host");
        push!(&self.host);
        push!("--port");
        push!(self.port.to_string());
        args
    }
}

// ============================================================================================
// Process ownership
// ============================================================================================

#[derive(Debug, Clone)]
pub struct ProcessOwner {
    inner: Arc<ProcessInner>,
}

#[derive(Debug)]
struct ProcessInner {
    config: LlamaConfig,
    child: AsyncMutex<Option<Child>>,
    pid: AtomicU32,
}

impl ProcessOwner {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            inner: Arc::new(ProcessInner {
                config,
                child: AsyncMutex::new(None),
                pid: AtomicU32::new(0),
            }),
        }
    }

    pub async fn spawn(&self) -> Result<u32, EngineError> {
        let mut child_guard = self.inner.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            if child.try_wait()?.is_none() {
                return child.id().ok_or(EngineError::ProcessExited);
            }
            *child_guard = None;
        }

        let mut command = Command::new(&self.inner.config.executable);
        command
            .args(self.inner.config.launch_args())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command
                .as_std_mut()
                .creation_flags(WINDOWS_CREATE_NO_WINDOW);
        }

        let child = command.spawn()?;
        // `kill_on_drop` covers an orderly exit. This covers the rest: a crash or an "End
        // task" would otherwise leave llama-server holding the GPU and port 8740 against
        // the next launch.
        #[cfg(windows)]
        if let Some(handle) = child.raw_handle() {
            crate::job::adopt(handle);
        }
        let pid = child.id().ok_or(EngineError::ProcessExited)?;
        self.inner.pid.store(pid, Ordering::Release);
        *child_guard = Some(child);
        drop(child_guard);

        Ok(pid)
    }

    pub fn pid(&self) -> Option<u32> {
        match self.inner.pid.load(Ordering::Acquire) {
            0 => None,
            pid => Some(pid),
        }
    }

    pub async fn is_alive(&self) -> Result<bool, EngineError> {
        let mut child = self.inner.child.lock().await;
        let alive = match child.as_mut() {
            Some(child) => child.try_wait()?.is_none(),
            None => false,
        };
        if !alive {
            *child = None;
            self.inner.pid.store(0, Ordering::Release);
        }
        Ok(alive)
    }

    pub async fn terminate(&self) -> Result<bool, EngineError> {
        let child = self.inner.child.lock().await.take();
        let Some(mut child) = child else {
            self.inner.pid.store(0, Ordering::Release);
            return Ok(false);
        };
        let _ = child.start_kill();
        let waited = tokio::time::timeout(self.inner.config.shutdown_timeout, child.wait()).await;
        self.inner.pid.store(0, Ordering::Release);
        match waited {
            Ok(result) => {
                let _ = result?;
                Ok(true)
            }
            Err(_) => Err(EngineError::Timeout(
                "llama-server did not exit before shutdown timeout".into(),
            )),
        }
    }
}

// ============================================================================================
// HTTP client
// ============================================================================================

#[derive(Debug, Clone)]
pub struct LlamaClient {
    client: reqwest::Client,
    base_url: String,
    model_alias: String,
    context_tokens: usize,
}

/// One slot as llama-server reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeSlotStatus {
    pub slot: SlotId,
    pub context_capacity: usize,
    pub is_processing: bool,
}

/// What llama-server says it is actually running, after its own clamping.
#[derive(Debug, Clone)]
pub struct ServerProperties {
    pub model_path: Option<String>,
    /// The PER-SLOT window. `--ctx-size` is divided across `--parallel` slots, so this is the
    /// total divided by [`PARALLEL_SLOTS`], not the value passed on the command line.
    pub per_slot_context: Option<usize>,
}

/// Decode statistics for one completed generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    pub time_to_first_token: Option<Duration>,
    pub wall_time: Duration,
    pub content: String,
    /// Thinking tokens, when the model emits them.
    ///
    /// Captured separately rather than discarded: with `--reasoning-format deepseek` a reasoning
    /// model can spend an entire token budget here and leave `content` empty, and a probe that
    /// only watched `content` would report "no tokens received" for a run that decoded fine.
    pub reasoning: String,
    pub prompt_tokens: Option<u64>,
    pub prompt_ms: Option<f64>,
    pub predicted_tokens: Option<u64>,
    pub predicted_ms: Option<f64>,
    /// Draft tokens proposed by the MTP model, when speculative decoding is active.
    pub draft_tokens: Option<u64>,
    /// Draft tokens the target model accepted.
    pub draft_accepted: Option<u64>,
}

impl GenerationMetrics {
    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        match (self.predicted_tokens, self.predicted_ms) {
            (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 * 1000.0 / ms),
            _ => None,
        }
    }

    pub fn prefill_tokens_per_second(&self) -> Option<f64> {
        match (self.prompt_tokens, self.prompt_ms) {
            (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 * 1000.0 / ms),
            _ => None,
        }
    }

    /// Fraction of proposed draft tokens the target model kept.
    pub fn draft_acceptance_rate(&self) -> Option<f64> {
        match (self.draft_tokens, self.draft_accepted) {
            (Some(drafted), Some(accepted)) if drafted > 0 => {
                Some(accepted as f64 / drafted as f64)
            }
            _ => None,
        }
    }
}

impl LlamaClient {
    pub fn new(config: &LlamaConfig) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .read_timeout(Duration::from_secs(15))
            .no_proxy()
            .build()?;
        Ok(Self {
            client,
            base_url: config.base_url(),
            model_alias: config.model_alias.clone(),
            context_tokens: config.tokens_per_slot,
        })
    }

    pub async fn healthy(&self) -> bool {
        self.client
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }

    async fn post_json(&self, path: &str, body: Value) -> Result<Value, EngineError> {
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .timeout(Duration::from_secs(5))
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        let bytes = bounded_response(response, MAX_PROTOCOL_BODY_BYTES).await?;
        if !status.is_success() {
            // Server error bodies can echo user content. Keep them out of diagnostics.
            return Err(EngineError::Protocol(format!(
                "{path} returned HTTP {status}"
            )));
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Count the actual chat template, then discard oldest exchanges until the reply fits.
    /// Oversized instructions or the latest input fail explicitly; they are never truncated.
    pub async fn fit_messages<'a>(
        &self,
        messages: &[(&'a str, String)],
        reply_tokens: usize,
    ) -> Result<Vec<(&'a str, String)>, EngineError> {
        let limit = self
            .context_tokens
            .saturating_sub(reply_tokens)
            .saturating_sub(256);
        let mut fitted = messages.to_vec();
        if fitted.iter().map(|(_, text)| text.len()).sum::<usize>() > 262_144 {
            return Err(EngineError::Protocol(
                "conversation exceeds text limit".into(),
            ));
        }
        loop {
            let wire: Vec<_> = fitted
                .iter()
                .map(|(role, content)| json!({"role":role,"content":content}))
                .collect();
            let template = self.post_json("/apply-template", json!({"messages":wire,"add_generation_prompt":true,"chat_template_kwargs":{"enable_thinking":false}})).await?;
            let prompt = template
                .get("prompt")
                .and_then(Value::as_str)
                .ok_or_else(|| EngineError::Protocol("template response has no prompt".into()))?;
            let tokens = self
                .post_json(
                    "/tokenize",
                    json!({"content":prompt,"add_special":true,"parse_special":true}),
                )
                .await?;
            let count = tokens
                .get("tokens")
                .and_then(Value::as_array)
                .ok_or_else(|| EngineError::Protocol("tokenizer response has no tokens".into()))?
                .len();
            if count <= limit {
                return Ok(fitted);
            }
            let first = usize::from(fitted.first().is_some_and(|(role, _)| *role == "system"));
            let last_user = fitted
                .iter()
                .rposition(|(role, _)| *role == "user")
                .unwrap_or(first);
            if first >= last_user {
                return Err(EngineError::Protocol(
                    "system prompt and latest input exceed context capacity; shorten them".into(),
                ));
            }
            fitted.remove(first);
            while first < fitted.len() && fitted[first].0 == "assistant" {
                fitted.remove(first);
            }
        }
    }

    async fn get_json(&self, path: &str, timeout: Duration) -> Result<Value, EngineError> {
        let response = self
            .client
            .get(format!("{}{path}", self.base_url))
            .timeout(timeout)
            .send()
            .await?;
        let status = response.status();
        let bytes = bounded_response(response, MAX_PROTOCOL_BODY_BYTES).await?;
        if !status.is_success() {
            return Err(EngineError::Protocol(format!(
                "GET {path} returned HTTP {status}: {}",
                bounded_text(&bytes, 2_048)
            )));
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub async fn properties(&self) -> Result<ServerProperties, EngineError> {
        let raw = self.get_json("/props", Duration::from_secs(10)).await?;
        let per_slot_context = raw
            .get("default_generation_settings")
            .and_then(|settings| settings.get("n_ctx"))
            .or_else(|| raw.get("n_ctx"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok());
        let model_path = raw
            .get("model_path")
            .or_else(|| raw.get("model"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok(ServerProperties {
            model_path,
            per_slot_context,
        })
    }

    pub async fn slots(&self) -> Result<Vec<NativeSlotStatus>, EngineError> {
        let value = self.get_json("/slots", Duration::from_secs(10)).await?;
        let entries = value
            .as_array()
            .ok_or_else(|| EngineError::Protocol("/slots did not return an array".into()))?;
        entries.iter().cloned().map(parse_slot).collect()
    }

    /// Streams a chat completion, sending each token as it arrives.
    ///
    /// `messages` are (role, content) pairs. Tokens go out through `sender` so the caller can
    /// start speaking the first phrase while the rest is still being generated - the difference
    /// between audio at 700 ms and audio when the whole reply is finished.
    ///
    /// Returns the full text. A closed channel ends generation early, which is how an
    /// interruption stops the model rather than letting it run to completion unheard.
    pub async fn stream_completion(
        &self,
        kind: SlotKind,
        messages: &[(&str, String)],
        max_tokens: usize,
        temperature: f32,
        sender: std::sync::mpsc::Sender<String>,
    ) -> Result<Completion, EngineError> {
        self.stream_completion_with(kind, messages, max_tokens, temperature, move |token| {
            sender.send(token).is_ok()
        })
        .await
    }

    /// Runs token delivery and completion on the same task, preserving their order.
    /// Dropping this future closes the HTTP response, cancelling the server request.
    pub async fn stream_completion_with<F>(
        &self,
        kind: SlotKind,
        messages: &[(&str, String)],
        max_tokens: usize,
        temperature: f32,
        mut on_token: F,
    ) -> Result<Completion, EngineError>
    where
        F: FnMut(String) -> bool + Send,
    {
        let messages = self.fit_messages(messages, max_tokens).await?;
        let wire: Vec<Value> = messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        let body = json!({
            "model": self.model_alias, "messages": wire, "stream": true,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": if temperature <= 0.15 { 1.0 } else { 0.95 },
            // Both slots cache. The talker's prefix is the conversation and has to stay
            // byte-identical to be reused at all; the filter's is one fixed instruction of
            // about 260 tokens that never changes between calls, which makes it the easiest
            // prefix in the system to keep and the most wasteful to throw away - it was being
            // re-prefilled once per utterance, in front of every reply.
            "id_slot": kind.slot().index(), "cache_prompt": true,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .timeout(Duration::from_secs(60))
            .json(&body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let _ = bounded_response(response, 4_096).await?;
            return Err(EngineError::Protocol(format!(
                "chat completion returned HTTP {status}"
            )));
        }
        let mut decoder = CompletionDecoder::default();
        let mut stream = response.bytes_stream();
        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            for text in decoder.push(&chunk?)? {
                full.push_str(&text);
                if full.len() > 131_072 {
                    return Err(EngineError::Protocol("reply exceeded text limit".into()));
                }
                if !on_token(text) {
                    return Err(EngineError::Protocol("reply consumer disconnected".into()));
                }
            }
            if decoder.done {
                return Ok(Completion {
                    text: full,
                    truncated: decoder.truncated,
                });
            }
        }
        decoder.finish()?;
        Ok(Completion {
            text: full,
            truncated: decoder.truncated,
        })
    }
    /// Streams one completion and measures it.
    ///
    /// `id_slot` pins the request to a specific slot so the caller controls which KV cache is
    /// touched; without it llama-server picks a slot itself and the two policies blur together.
    pub async fn measured_completion(
        &self,
        kind: SlotKind,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<GenerationMetrics, EngineError> {
        let body = json!({
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "timings_per_token": true,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "id_slot": kind.slot().index(),
            "cache_prompt": true,
            // Redundant with the server's `--reasoning off`, kept so a request is still correct
            // if it is ever replayed against a server launched without that flag.
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let bytes = serde_json::to_vec(&body)?;
        if bytes.len() > MAX_PROTOCOL_BODY_BYTES {
            return Err(EngineError::InvalidConfig(
                "chat request exceeded the 8 MiB protocol limit".into(),
            ));
        }

        let started = Instant::now();
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(bytes)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let bytes = bounded_response(response, MAX_PROTOCOL_BODY_BYTES).await?;
            return Err(EngineError::Protocol(format!(
                "chat completion returned HTTP {status}: {}",
                bounded_text(&bytes, 4_096)
            )));
        }

        let mut metrics = GenerationMetrics::default();
        let mut buffer = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk?);
            if buffer.len() > MAX_PROTOCOL_BODY_BYTES {
                return Err(EngineError::Protocol(
                    "SSE buffer exceeded the protocol limit".into(),
                ));
            }
            // SSE events are newline-delimited; hold the trailing partial line for the next chunk.
            while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=position).collect::<Vec<_>>();
                let line = String::from_utf8_lossy(&line);
                let Some(payload) = line.trim().strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() || payload == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(payload) else {
                    continue;
                };
                apply_event(&mut metrics, &event, started);
            }
        }

        metrics.wall_time = started.elapsed();
        Ok(metrics)
    }
}

async fn bounded_response(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, EngineError> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(EngineError::Protocol(
                "HTTP response exceeded size limit".into(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// One finished completion.
pub struct Completion {
    pub text: String,
    /// The server stopped on `finish_reason: "length"` rather than because the model was
    /// done. The text is a prefix of what it meant to say, which is fine for a reply - the
    /// listener hears a thought that ends early - and is not fine for anything that has to
    /// return a *whole* answer. The filter is the case that matters: a truncated `CLEAN:`
    /// line is the speaker's question with the end cut off, and it still passes every
    /// resemblance check, because everything left in it really was said.
    pub truncated: bool,
}

#[derive(Default)]
struct CompletionDecoder {
    buffer: Vec<u8>,
    data: String,
    done: bool,
    finished: bool,
    truncated: bool,
}

impl CompletionDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, EngineError> {
        let mut tokens = Vec::new();
        // Enforce the limit while reading, including a peer that never sends a newline.
        for byte in bytes {
            self.buffer.push(*byte);
            if self.buffer.len() + self.data.len() > 262_144 {
                return Err(EngineError::Protocol(
                    "SSE event exceeded size limit".into(),
                ));
            }
            if *byte != b'\n' {
                continue;
            }
            let line = std::str::from_utf8(&self.buffer)
                .map_err(|_| EngineError::Protocol("invalid UTF-8 in completion stream".into()))?
                .trim_end_matches(['\r', '\n'])
                .to_string();
            self.buffer.clear();
            if line.is_empty() {
                if self.data.is_empty() {
                    continue;
                }
                let data = std::mem::take(&mut self.data);
                if data.trim() == "[DONE]" {
                    self.done = true;
                    continue;
                }
                let event: Value = serde_json::from_str(&data)?;
                if let Some(error) = event.get("error") {
                    return Err(EngineError::Protocol(format!("completion failed: {error}")));
                }
                if let Some(choice) = event
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|c| c.first())
                {
                    if let Some(text) = choice
                        .pointer("/delta/content")
                        .and_then(Value::as_str)
                        .filter(|t| !t.is_empty())
                    {
                        if self.done || self.finished {
                            return Err(EngineError::Protocol("text after completion".into()));
                        }
                        tokens.push(text.to_string());
                    }
                    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                        if !matches!(reason, "stop" | "length") {
                            return Err(EngineError::Protocol(format!(
                                "unsupported completion reason: {reason}"
                            )));
                        }
                        self.finished = true;
                        self.truncated = reason == "length";
                    }
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
            }
        }
        Ok(tokens)
    }
    fn finish(&self) -> Result<(), EngineError> {
        if (self.done || self.finished) && self.buffer.is_empty() && self.data.is_empty() {
            Ok(())
        } else {
            Err(EngineError::Protocol(
                "completion stream ended before a complete finish event".into(),
            ))
        }
    }
}

fn apply_event(metrics: &mut GenerationMetrics, event: &Value, started: Instant) {
    let delta = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"));

    for (key, sink) in [
        ("content", &mut metrics.content),
        ("reasoning_content", &mut metrics.reasoning),
    ] {
        let Some(text) = delta
            .and_then(|delta| delta.get(key))
            .and_then(Value::as_str)
        else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        // First *token-bearing* delta, not first SSE frame: llama-server emits a role-only
        // opening delta, and counting that as the first token would understate TTFT. Reasoning
        // counts too - it is decode time the listener waits through either way.
        metrics
            .time_to_first_token
            .get_or_insert_with(|| started.elapsed());
        sink.push_str(text);
    }

    // llama-server reports final timings on the closing frame, under `timings` at the top level
    // and (depending on build) mirrored into `usage`.
    let timings = event
        .get("timings")
        .or_else(|| event.get("usage").and_then(|usage| usage.get("timings")));
    if let Some(timings) = timings {
        let read_u64 = |key: &str| timings.get(key).and_then(Value::as_u64);
        let read_f64 = |key: &str| timings.get(key).and_then(Value::as_f64);
        metrics.prompt_tokens = read_u64("prompt_n").or(metrics.prompt_tokens);
        metrics.prompt_ms = read_f64("prompt_ms").or(metrics.prompt_ms);
        metrics.predicted_tokens = read_u64("predicted_n").or(metrics.predicted_tokens);
        metrics.predicted_ms = read_f64("predicted_ms").or(metrics.predicted_ms);
        metrics.draft_tokens = read_u64("draft_n").or(metrics.draft_tokens);
        metrics.draft_accepted = read_u64("draft_n_accepted").or(metrics.draft_accepted);
    }

    if let Some(usage) = event.get("usage") {
        if metrics.prompt_tokens.is_none() {
            metrics.prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
        }
        if metrics.predicted_tokens.is_none() {
            metrics.predicted_tokens = usage.get("completion_tokens").and_then(Value::as_u64);
        }
    }
}

fn parse_slot(value: Value) -> Result<NativeSlotStatus, EngineError> {
    let id = value
        .get("id_slot")
        .or_else(|| value.get("id"))
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::Protocol("slot response has no numeric id".into()))?;
    let slot = SlotId::from_index(id as usize)
        .map_err(|error| EngineError::Protocol(error.to_string()))?;
    let context_capacity = value
        .get("n_ctx")
        .or_else(|| value.get("n_ctx_slot"))
        .or_else(|| value.get("context_tokens"))
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::Protocol("slot response has no context capacity".into()))?;
    let context_capacity = usize::try_from(context_capacity)
        .map_err(|_| EngineError::Protocol("slot context capacity exceeds usize".into()))?;
    Ok(NativeSlotStatus {
        slot,
        context_capacity,
        is_processing: value
            .get("is_processing")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn bounded_text(bytes: &[u8], limit: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= limit {
        text.into_owned()
    } else {
        let mut truncated: String = text.chars().take(limit).collect();
        truncated.push_str(" [truncated]");
        truncated
    }
}

// ============================================================================================
// Runtime
// ============================================================================================

/// What the server looked like once it came up.
#[derive(Debug, Clone)]
pub struct StartReport {
    pub pid: u32,
    pub startup: Duration,
    pub slots: Vec<NativeSlotStatus>,
    pub per_slot_context: Option<usize>,
    pub model_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlamaEngine {
    config: LlamaConfig,
    process: ProcessOwner,
    client: LlamaClient,
}

impl LlamaEngine {
    pub fn new(config: LlamaConfig) -> Result<Self, EngineError> {
        config.validate()?;
        let client = LlamaClient::new(&config)?;
        let process = ProcessOwner::new(config.clone());
        Ok(Self {
            config,
            process,
            client,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn client(&self) -> &LlamaClient {
        &self.client
    }

    pub fn process(&self) -> &ProcessOwner {
        &self.process
    }

    /// Spawns the server and blocks until it answers, or explains why it never did.
    pub async fn start(&self) -> Result<StartReport, EngineError> {
        // Refuse to adopt a server this runtime does not own: it may be running an entirely
        // different model or context size, and every measurement taken against it would be a
        // quiet lie.
        if self.client.healthy().await {
            return Err(EngineError::EndpointOccupied(self.config.base_url()));
        }

        let started = Instant::now();
        let pid = self.process.spawn().await?;

        loop {
            if self.client.healthy().await {
                break;
            }
            if !self.process.is_alive().await? {
                return Err(EngineError::Protocol(format!(
                    "llama-server exited during startup. Recent errors:\n{}",
                    self.failure_detail()
                )));
            }
            if started.elapsed() > self.config.startup_timeout {
                let _ = self.process.terminate().await;
                return Err(EngineError::Timeout(format!(
                    "llama-server did not become healthy within {:?}. Recent errors:\n{}",
                    self.config.startup_timeout,
                    self.failure_detail()
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        let startup = started.elapsed();
        let properties = self.client.properties().await?;
        let slots = self.client.slots().await?;

        if slots.len() != PARALLEL_SLOTS {
            return Err(EngineError::Protocol(format!(
                "expected {PARALLEL_SLOTS} slots but llama-server reported {}",
                slots.len()
            )));
        }

        Ok(StartReport {
            pid,
            startup,
            slots,
            per_slot_context: properties.per_slot_context,
            model_path: properties.model_path,
        })
    }

    pub async fn stop(&self) -> Result<bool, EngineError> {
        self.process.terminate().await
    }

    fn failure_detail(&self) -> &'static str {
        "native output is disabled; verify model assets and available graphics memory"
    }
}

// ============================================================================================
// Tests
// ============================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tokenizer_budget_evicts_complete_exchanges_and_rejects_oversized_current_input() {
        use axum::{routing::post, Json, Router};
        let app=Router::new()
            .route("/apply-template",post(|Json(value):Json<Value>|async move {Json(json!({"prompt":value["messages"].to_string()}))}))
            .route("/tokenize",post(|Json(value):Json<Value>|async move {
                let text=value["content"].as_str().unwrap();
                Json(json!({"tokens":vec![1; if text.contains("OLDER")||text.contains("TOO_LARGE") {9000}else{10}]}))
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut config = config();
        config.port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = LlamaClient::new(&config).unwrap();
        let messages = vec![
            ("system", "instructions".into()),
            ("user", "OLDER".into()),
            ("assistant", "old answer".into()),
            ("user", "latest".into()),
        ];
        let fitted = client.fit_messages(&messages, 1024).await.unwrap();
        assert_eq!(
            fitted,
            vec![("system", "instructions".into()), ("user", "latest".into())]
        );
        assert!(client
            .fit_messages(
                &[("system", "TOO_LARGE".into()), ("user", "latest".into())],
                1024
            )
            .await
            .is_err());
        task.abort();
    }

    #[test]
    fn streaming_parser_handles_split_utf8_and_requires_a_terminal_event() {
        let wire="data: {\"choices\":[{\"delta\":{\"content\":\"héllo\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let mut decoder = CompletionDecoder::default();
        let mut text = String::new();
        for byte in wire.as_bytes() {
            for token in decoder.push(&[*byte]).unwrap() {
                text.push_str(&token);
            }
        }
        assert_eq!(text, "héllo");
        decoder.finish().unwrap();
        let mut truncated = CompletionDecoder::default();
        truncated
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
            .unwrap();
        assert!(truncated.finish().is_err());
    }

    #[test]
    fn malformed_and_oversized_stream_events_are_errors() {
        assert!(CompletionDecoder::default()
            .push(b"data: nope\n\n")
            .is_err());
        assert!(CompletionDecoder::default()
            .push(b"data: {\"error\":\"failed\"}\n\n")
            .is_err());
        assert!(CompletionDecoder::default()
            .push(&vec![b'x'; 262_145])
            .is_err());
    }

    #[tokio::test]
    async fn actual_http_stream_delivers_tokens_before_returning_completion() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut config = config();
        config.port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            for (path, mime, body) in [
                ("/apply-template", "application/json", r#"{"prompt":"hi"}"#),
                ("/tokenize", "application/json", r#"{"tokens":[1,2]}"#),
                ("/v1/chat/completions", "text/event-stream", "data: {\"choices\":[{\"delta\":{\"content\":\"Hello.\"}}]}\n\ndata: [DONE]\n\n"),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 16384];
                let n = socket.read(&mut request).await.unwrap();
                assert!(String::from_utf8_lossy(&request[..n]).starts_with(&format!("POST {path} ")));
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
            }
        });
        let client = LlamaClient::new(&config).unwrap();
        let mut events = Vec::new();
        let full = client
            .stream_completion_with(SlotKind::Talker, &[("user", "hi".into())], 8, 0.1, |t| {
                events.push(t);
                true
            })
            .await
            .unwrap();
        events.push("DONE".into());
        assert_eq!(events, vec!["Hello.", "DONE"]);
        assert_eq!(full.text, "Hello.");
        assert!(!full.truncated);
        server.await.unwrap();
    }

    fn config() -> LlamaConfig {
        LlamaConfig::from_zen_root(r"C:\zen-ai")
    }

    fn args_of(config: &LlamaConfig) -> Vec<String> {
        config
            .launch_args()
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn value_after(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|index| args.get(index + 1))
            .cloned()
    }

    // --- slots -------------------------------------------------------------------------------

    #[test]
    fn slot_kind_and_parallel_slots_agree() {
        // If PARALLEL_SLOTS changes without SlotKind gaining a variant, SlotId::kind() starts
        // mapping every extra slot to Talker and nothing else notices.
        assert_eq!(SlotKind::ALL.len(), PARALLEL_SLOTS);
    }

    #[test]
    fn roles_map_to_fixed_slots_both_ways() {
        assert_eq!(SlotKind::Filter.slot(), SlotId::FILTER);
        assert_eq!(SlotKind::Talker.slot(), SlotId::TALKER);
        for kind in SlotKind::ALL {
            assert_eq!(kind.slot().kind(), kind, "{kind:?} must round-trip");
        }
    }

    #[test]
    fn the_two_slots_run_opposite_kv_policies() {
        assert_eq!(SlotKind::Filter.kv_policy(), KvPolicy::PerRequest);
        assert_eq!(SlotKind::Talker.kv_policy(), KvPolicy::StablePrefix);
        assert!(SlotKind::Talker.kv_policy().prefix_stability_matters());
        assert!(!SlotKind::Filter.kv_policy().prefix_stability_matters());
    }

    #[test]
    fn out_of_range_slots_cannot_be_constructed() {
        assert!(SlotId::from_index(0).is_ok());
        assert!(SlotId::from_index(1).is_ok());
        assert_eq!(SlotId::from_index(2), Err(SlotIdError { index: 2 }));
        assert!(SlotId::from_index(usize::MAX).is_err());
    }

    #[test]
    fn deserialization_cannot_smuggle_in_a_bad_slot() {
        // serde goes through from_index, so a malformed /slots response cannot bypass the check.
        assert!(serde_json::from_str::<SlotId>("0").is_ok());
        assert!(serde_json::from_str::<SlotId>("7").is_err());
    }

    #[test]
    fn display_names_the_slot_and_its_role() {
        assert_eq!(SlotId::FILTER.to_string(), "slot0(filter)");
        assert_eq!(SlotId::TALKER.to_string(), "slot1(talker)");
    }

    // --- launch configuration ----------------------------------------------------------------

    #[test]
    fn two_slots_of_eight_k_make_a_sixteen_k_context() {
        let config = config();
        assert_eq!(config.tokens_per_slot, 8_192);
        assert_eq!(config.total_context_tokens(), 16_384);
        assert_eq!(
            value_after(&args_of(&config), "--ctx-size").as_deref(),
            Some("16384")
        );
        assert_eq!(
            value_after(&args_of(&config), "--parallel").as_deref(),
            Some("2")
        );
    }

    #[test]
    fn all_layers_are_pinned_and_auto_fit_cannot_walk_that_back() {
        // These two travel together. `--fit on` may reduce the offloaded layer count, so leaving
        // it enabled would make the -ngl pin advisory rather than binding.
        let args = args_of(&config());
        assert_eq!(value_after(&args, "--n-gpu-layers").as_deref(), Some("99"));
        assert_eq!(value_after(&args, "--fit").as_deref(), Some("off"));
    }

    #[test]
    fn swa_full_is_present_because_prefix_reuse_depends_on_it() {
        assert!(args_of(&config()).iter().any(|arg| arg == "--swa-full"));
    }

    #[test]
    fn swa_full_can_be_disabled_for_a_vram_comparison() {
        let mut config = config();
        config.swa_full = false;
        assert!(!args_of(&config).iter().any(|arg| arg == "--swa-full"));
    }

    #[test]
    fn thinking_is_off_at_the_server_by_default() {
        // Enforced here rather than per-request so one caller forgetting the flag cannot
        // reintroduce reasoning tokens into a live conversation.
        let args = args_of(&config());
        assert_eq!(value_after(&args, "--reasoning").as_deref(), Some("off"));
        assert_eq!(
            value_after(&args, "--reasoning-budget").as_deref(),
            Some("0")
        );
    }

    #[test]
    fn enabling_thinking_drops_the_zero_budget_that_would_cancel_it() {
        let mut config = config();
        config.thinking = true;
        let args = args_of(&config);
        assert_eq!(value_after(&args, "--reasoning").as_deref(), Some("on"));
        assert!(!args.iter().any(|arg| arg == "--reasoning-budget"));
    }

    #[test]
    fn the_mtp_draft_is_wired_as_a_draft_mtp_spec_model() {
        let args = args_of(&config());
        assert_eq!(
            value_after(&args, "--spec-type").as_deref(),
            Some("draft-mtp")
        );
        assert_eq!(
            value_after(&args, "--spec-draft-ngl").as_deref(),
            Some("99")
        );
        assert!(value_after(&args, "--spec-draft-model")
            .is_some_and(|path| path.ends_with("mtp-gemma-4-E2B-it.gguf")));
    }

    #[test]
    fn dropping_the_draft_model_removes_every_spec_flag() {
        // An A/B run without MTP must not leave a dangling --spec-* flag behind.
        let mut config = config();
        config.draft_model = None;
        assert!(!args_of(&config)
            .iter()
            .any(|arg| arg.starts_with("--spec-")));
    }

    #[test]
    fn private_runtime_does_not_enable_disk_slot_snapshots() {
        assert!(!args_of(&config())
            .iter()
            .any(|arg| arg == "--slot-save-path"));
    }

    #[test]
    fn a_remote_bind_address_is_rejected() {
        let mut config = config();
        config.host = "0.0.0.0".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn a_micro_batch_larger_than_the_batch_is_rejected() {
        let mut config = config();
        config.micro_batch_size = config.batch_size + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn zero_gpu_layers_is_rejected_rather_than_silently_running_on_cpu() {
        let mut config = config();
        config.gpu_layers = 0;
        assert!(config.validate().is_err());
    }

    // --- protocol ----------------------------------------------------------------------------

    #[test]
    fn slots_outside_the_configured_range_are_rejected() {
        // A server launched with more parallel slots than this build expects must be caught here
        // rather than producing a SlotId that lies about which role owns the cache.
        assert!(parse_slot(json!({"id_slot": 0, "n_ctx": 8192})).is_ok());
        assert!(parse_slot(json!({"id_slot": 5, "n_ctx": 8192})).is_err());
    }

    #[test]
    fn a_slot_without_a_context_capacity_is_a_protocol_error() {
        assert!(parse_slot(json!({"id_slot": 0})).is_err());
    }

    #[test]
    fn first_content_token_sets_ttft_but_an_empty_role_delta_does_not() {
        let started = Instant::now();
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            started,
        );
        assert!(metrics.time_to_first_token.is_none());
        apply_event(
            &mut metrics,
            &json!({"choices": [{"delta": {"content": "hi"}}]}),
            started,
        );
        assert!(metrics.time_to_first_token.is_some());
        assert_eq!(metrics.content, "hi");
    }

    #[test]
    fn reasoning_only_output_still_registers_as_tokens_received() {
        // A reasoning model can spend the whole budget in reasoning_content and leave content
        // empty. Reporting that as "no tokens received" sent us chasing a decode bug that was
        // really just thinking being enabled.
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"choices": [{"delta": {"reasoning_content": "hmm"}}]}),
            Instant::now(),
        );
        assert!(metrics.time_to_first_token.is_some());
        assert_eq!(metrics.reasoning, "hmm");
        assert!(metrics.content.is_empty());
    }

    #[test]
    fn content_and_reasoning_accumulate_into_separate_buffers() {
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"choices": [{"delta": {"reasoning_content": "think", "content": "say"}}]}),
            Instant::now(),
        );
        assert_eq!(metrics.reasoning, "think");
        assert_eq!(metrics.content, "say");
    }

    #[test]
    fn draft_acceptance_is_computed_from_reported_timings() {
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"timings": {"draft_n": 100, "draft_n_accepted": 37}}),
            Instant::now(),
        );
        let rate = metrics.draft_acceptance_rate().expect("rate");
        assert!((rate - 0.37).abs() < 1e-9);
    }

    #[test]
    fn acceptance_rate_is_absent_rather_than_zero_when_mtp_is_off() {
        // Reporting 0% for a run with no draft model would read as "MTP is failing" instead of
        // "MTP was never enabled".
        let metrics = GenerationMetrics::default();
        assert!(metrics.draft_acceptance_rate().is_none());
    }

    #[test]
    fn decode_rate_is_derived_from_predicted_tokens_and_time() {
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"timings": {"predicted_n": 50, "predicted_ms": 1000.0}}),
            Instant::now(),
        );
        assert_eq!(metrics.decode_tokens_per_second(), Some(50.0));
    }

    #[test]
    fn a_zero_duration_does_not_produce_an_infinite_rate() {
        let mut metrics = GenerationMetrics::default();
        apply_event(
            &mut metrics,
            &json!({"timings": {"predicted_n": 50, "predicted_ms": 0.0}}),
            Instant::now(),
        );
        assert_eq!(metrics.decode_tokens_per_second(), None);
    }
}
