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
//! - The talker keeps a byte-identical prompt prefix so its cached conversation is reused turn
//!   to turn; the filter carries nothing from one request to the next.
//! - Every layer is pinned to the device; startup fails loudly rather than falling back to CPU.
//! - Thinking is off at the server, not per request.

use std::{
    collections::VecDeque,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tokio::{
    io::AsyncBufReadExt,
    process::{Child, Command},
    sync::Mutex as AsyncMutex,
    task::JoinHandle,
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
    /// Stateless. Repairs one transcript, and nothing about a request depends on the one before.
    /// Its fixed instruction is still a prefix, and is still cached.
    Filter,
    /// Stateful. Holds the rolling conversation and speaks to the user. Its prompt prefix has to
    /// stay byte-identical across turns for the server to reuse the cached conversation: one
    /// changed byte near the front re-prefills all of it, heard as a stall before Zen speaks.
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

// ============================================================================================
// Errors
// ============================================================================================

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid engine configuration: {0}")]
    InvalidConfig(String),
    #[error("configured llama-server endpoint is already responding but is not owned by this runtime: {0}")]
    EndpointOccupied(String),
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
pub(crate) const WINDOWS_CREATE_NO_WINDOW: u32 = 0x0800_0000;

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
    startup: Arc<Mutex<StartupLog>>,
    /// Reads the server's error output for as long as it runs, so the pipe never fills.
    reader: AsyncMutex<Option<JoinHandle<()>>>,
}

/// The last lines llama-server printed while starting, kept only until it is up.
///
/// Its output is otherwise thrown away: once a conversation is running, a server log is no
/// place for anything of the user's to end up. While it starts, that output is the only
/// account of why a launch failed - a model file it could not read, graphics memory it could
/// not get - and without it the failure could only be described in general terms.
#[derive(Debug, Default)]
struct StartupLog {
    lines: VecDeque<String>,
    closed: bool,
}

impl StartupLog {
    const LINES: usize = 12;
    const LINE_CHARS: usize = 200;

    fn push(&mut self, line: &str) {
        let line = line.trim();
        if self.closed || line.is_empty() {
            return;
        }
        if self.lines.len() == Self::LINES {
            self.lines.pop_front();
        }
        self.lines
            .push_back(line.chars().take(Self::LINE_CHARS).collect());
    }

    /// Startup is over: forget what was printed, and keep nothing from here on.
    fn close(&mut self) {
        self.closed = true;
        self.lines.clear();
    }

    /// The line that best says what went wrong: the last one reporting an error, or failing
    /// that, the last one printed.
    fn reason(&self) -> Option<&str> {
        let failed = |line: &&String| {
            let line = line.to_ascii_lowercase();
            ["error", "failed", "unable", "cannot", "out of memory"]
                .iter()
                .any(|word| line.contains(word))
        };
        self.lines
            .iter()
            .rev()
            .find(failed)
            .or(self.lines.back())
            .map(String::as_str)
    }
}

impl ProcessOwner {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            inner: Arc::new(ProcessInner {
                config,
                child: AsyncMutex::new(None),
                startup: Arc::default(),
                reader: AsyncMutex::new(None),
            }),
        }
    }

    pub async fn spawn(&self) -> Result<(), EngineError> {
        let mut child_guard = self.inner.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            if child.try_wait()?.is_none() {
                return Ok(());
            }
            *child_guard = None;
        }

        let mut command = Command::new(&self.inner.config.executable);
        command
            .args(self.inner.config.launch_args())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command
                .as_std_mut()
                .creation_flags(WINDOWS_CREATE_NO_WINDOW);
        }

        let mut child = command.spawn()?;
        // `kill_on_drop` covers an orderly exit. This covers the rest: a crash or an "End
        // task" would otherwise leave llama-server holding the GPU and port 8740 against
        // the next launch.
        #[cfg(windows)]
        if let Some(handle) = child.raw_handle() {
            crate::job::adopt(handle);
        }
        *self.inner.startup.lock().unwrap_or_else(|e| e.into_inner()) = StartupLog::default();
        if let Some(stderr) = child.stderr.take() {
            let log = Arc::clone(&self.inner.startup);
            *self.inner.reader.lock().await = Some(tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(stderr);
                let mut line = Vec::new();
                // Read to the end whatever arrives, including invalid UTF-8: a pipe nobody
                // empties fills up, and the server then blocks on its next log line.
                loop {
                    line.clear();
                    match reader.read_until(b'\n', &mut line).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => log
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(&String::from_utf8_lossy(&line)),
                    }
                }
            }));
        }
        *child_guard = Some(child);
        Ok(())
    }

    /// The server is up. Nothing it prints from here on is kept.
    fn started(&self) {
        self.inner
            .startup
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .close();
    }

    /// Why startup failed, in the server's own words, once it has stopped printing.
    async fn startup_failure(&self) -> Option<String> {
        if let Some(reader) = self.inner.reader.lock().await.take() {
            // The process is gone by now, so the pipe is closing; this only waits for its last
            // lines to be read.
            let _ = tokio::time::timeout(Duration::from_secs(1), reader).await;
        }
        let log = self.inner.startup.lock().unwrap_or_else(|e| e.into_inner());
        log.reason().map(str::to_owned)
    }

    pub async fn is_alive(&self) -> Result<bool, EngineError> {
        let mut child = self.inner.child.lock().await;
        let alive = match child.as_mut() {
            Some(child) => child.try_wait()?.is_none(),
            None => false,
        };
        if !alive {
            *child = None;
        }
        Ok(alive)
    }

    pub async fn terminate(&self) -> Result<bool, EngineError> {
        let child = self.inner.child.lock().await.take();
        let Some(mut child) = child else {
            return Ok(false);
        };
        let _ = child.start_kill();
        let waited = tokio::time::timeout(self.inner.config.shutdown_timeout, child.wait()).await;
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
}

/// What llama-server says it is actually running, after its own clamping.
#[derive(Debug, Clone)]
pub struct ServerProperties {
    /// The PER-SLOT window. `--ctx-size` is divided across `--parallel` slots, so this is the
    /// total divided by [`PARALLEL_SLOTS`], not the value passed on the command line.
    pub per_slot_context: Option<usize>,
}

impl LlamaClient {
    pub fn new(config: &LlamaConfig) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .read_timeout(Duration::from_secs(15))
            // Drop an idle connection before the server does, rather than racing it.
            //
            // llama-server is built on cpp-httplib, which closes a keep-alive connection after
            // about five seconds of silence. A turn easily leaves a longer gap than that -
            // recognition and synthesis both take longer - so the next request goes out on a
            // socket the other end has already dropped and fails as it is sent. Retiring them
            // here first costs one loopback handshake, which is not measurable next to the work
            // either side of it.
            .pool_idle_timeout(Duration::from_secs(2))
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

    /// Posts to one of the server's pure endpoints - rendering a template, counting tokens.
    ///
    /// Retried once if the request never reached the server. Connections are pooled, and the
    /// server closes an idle one on its own schedule, so a request can be written into a socket
    /// the other end has just dropped. It surfaces as a send failure rather than a status, and
    /// on the reference machine it took down a whole turn about one run in two. Nothing here
    /// changes state on the server, so sending it again is safe; a request that did arrive and
    /// came back with a status is not retried, because that is an answer.
    async fn post_json(&self, path: &str, body: Value) -> Result<Value, EngineError> {
        let send = || {
            self.client
                .post(format!("{}{path}", self.base_url))
                .timeout(Duration::from_secs(5))
                .json(&body)
                .send()
        };
        let response = match send().await {
            Ok(response) => response,
            Err(error) if error.is_request() || error.is_connect() => send().await?,
            Err(error) => return Err(error.into()),
        };
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
        // Counted by the tokenizer that will actually read this, every time, rather than
        // estimated from the length of the text.
        //
        // There used to be a character-count estimate here that skipped the real count whenever
        // it looked comfortably small. Characters per token is not a constant: it depends on the
        // script and on the content. Three bytes of Devanagari or CJK is one character, digits
        // and identifiers tokenise far denser than prose, and the estimate was a single divisor
        // for all of it. Whenever it read low the conversation went to the server over budget,
        // and what that costs is the oldest turns being dropped by the server instead of by the
        // rule here - silently, and differently depending on what language someone spoke.
        //
        // The reason it existed was cost: two round trips, one of which answers with a JSON
        // array holding every token id. That is worth paying on every turn rather than being
        // wrong about the budget in the cases hardest to notice.
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
        Ok(ServerProperties { per_slot_context })
    }

    pub async fn slots(&self) -> Result<Vec<NativeSlotStatus>, EngineError> {
        let value = self.get_json("/slots", Duration::from_secs(10)).await?;
        let entries = value
            .as_array()
            .ok_or_else(|| EngineError::Protocol("/slots did not return an array".into()))?;
        entries.iter().cloned().map(parse_slot).collect()
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

    /// Spawns the server and waits until it answers with the shape this runtime asked for, or
    /// explains why it never did.
    pub async fn start(&self) -> Result<(), EngineError> {
        // Refuse to adopt a server this runtime does not own: it may be running an entirely
        // different model or context size, and every measurement taken against it would be a
        // quiet lie.
        if self.client.healthy().await {
            return Err(EngineError::EndpointOccupied(self.config.base_url()));
        }

        let started = Instant::now();
        self.process.spawn().await?;

        loop {
            if self.client.healthy().await {
                break;
            }
            if !self.process.is_alive().await? {
                return Err(EngineError::Protocol(
                    self.failure("the model server stopped while starting")
                        .await,
                ));
            }
            if started.elapsed() > self.config.startup_timeout {
                let _ = self.process.terminate().await;
                return Err(EngineError::Timeout(
                    self.failure(&format!(
                        "the model server was not ready within {} s",
                        self.config.startup_timeout.as_secs()
                    ))
                    .await,
                ));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        self.process.started();

        let slots = self.client.slots().await?;
        if slots.len() != PARALLEL_SLOTS {
            return Err(EngineError::Protocol(format!(
                "expected {PARALLEL_SLOTS} slots but llama-server reported {}",
                slots.len()
            )));
        }
        // The conversation is fitted to the window this runtime asked for. A server that quietly
        // gave each slot less would truncate replies on its own terms instead.
        let window = self.client.properties().await?.per_slot_context;
        if window.is_some_and(|tokens| tokens != self.config.tokens_per_slot) {
            return Err(EngineError::Protocol(format!(
                "llama-server gave each slot {} tokens instead of {}",
                window.unwrap_or_default(),
                self.config.tokens_per_slot
            )));
        }
        Ok(())
    }

    pub async fn stop(&self) -> Result<bool, EngineError> {
        self.process.terminate().await
    }

    /// A startup failure, with the server's own last word on it when it gave one.
    async fn failure(&self, what: &str) -> String {
        match self.process.startup_failure().await {
            Some(reason) => format!("{what}: {reason}"),
            None => {
                format!("{what} without saying why; check the model files and free graphics memory")
            }
        }
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
        // Long enough that the cheap estimate cannot rule the limit out, so the exact count
        // runs. Short conversations deliberately skip both round trips.
        let bulky = format!("OLDER {}", "a lot of words to say ".repeat(1_200));
        let messages = vec![
            ("system", "instructions".into()),
            ("user", bulky),
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
                &[
                    (
                        "system",
                        format!("TOO_LARGE {}", "and more words ".repeat(2_000))
                    ),
                    ("user", "latest".into())
                ],
                1024
            )
            .await
            .is_err());

        // And the shortcut itself: an ordinary conversation is fitted without asking the
        // server anything, which is why this client can answer with the server torn down.
        task.abort();
        let ordinary = vec![("system", "instructions".into()), ("user", "latest".into())];
        assert_eq!(
            client.fit_messages(&ordinary, 1024).await.unwrap(),
            ordinary,
            "a short conversation must not need the server to be fitted"
        );
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
            // Every conversation is rendered and counted before it is sent, however short it
            // looks, so the exchange is template, tokenize, then the completion itself.
            for (path, mime, body) in [
                ("/apply-template", "application/json", "{\"prompt\":\"hi\"}"),
                ("/tokenize", "application/json", "{\"tokens\":[1,2,3,4]}"),
                (
                    "/v1/chat/completions",
                    "text/event-stream",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello.\"}}]}\n\ndata: [DONE]\n\n",
                ),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 16384];
                let n = socket.read(&mut request).await.unwrap();
                assert!(
                    String::from_utf8_lossy(&request[..n]).starts_with(&format!("POST {path} "))
                );
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
    fn a_failed_startup_is_explained_by_its_last_error_and_nothing_is_kept_after() {
        let mut log = StartupLog::default();
        assert_eq!(log.reason(), None);
        log.push("load_backend: loaded CUDA backend");
        log.push("   ");
        assert_eq!(log.reason(), Some("load_backend: loaded CUDA backend"));
        log.push("ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory: out of memory\r\n");
        log.push("main: exiting due to model loading error");
        assert_eq!(
            log.reason(),
            Some("main: exiting due to model loading error")
        );
        for n in 0..50 {
            log.push(&format!("line {n} {}", "x".repeat(500)));
        }
        assert_eq!(log.lines.len(), StartupLog::LINES);
        assert!(log
            .lines
            .iter()
            .all(|l| l.chars().count() <= StartupLog::LINE_CHARS));
        log.close();
        log.push("error: something about a request");
        assert_eq!(log.reason(), None, "a running server's output is not kept");
    }

    #[test]
    fn a_slot_without_a_context_capacity_is_a_protocol_error() {
        assert!(parse_slot(json!({"id_slot": 0})).is_err());
    }
}
