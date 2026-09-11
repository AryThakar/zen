// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Separate native libraries by process. Framed PCM uses authenticated loopback sockets;
//! Native output is discarded for privacy. A worker exits when its owner disconnects.
use crate::{
    asr::{AsrEngine, AsrError, Recognizer, Transcript},
    tts::{Synthesizer, TtsEngine, TtsError},
};
use std::{
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

const HELLO: u8 = 1;
const READY: u8 = 2;
const ASR: u8 = 3;
const TTS: u8 = 4;
const CANCEL: u8 = 5;
const AUDIO: u8 = 6;
const TEXT: u8 = 7;
const DONE: u8 = 8;
const ERROR: u8 = 9;
const MAX_PACKET: usize = 16 * 1024 * 1024;
struct Packet {
    kind: u8,
    data: Vec<u8>,
}
fn write_packet(stream: &mut impl Write, kind: u8, data: &[u8]) -> io::Result<()> {
    if data.len() > MAX_PACKET {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native packet too large",
        ));
    }
    stream.write_all(&[kind])?;
    stream.write_all(&(data.len() as u32).to_le_bytes())?;
    stream.write_all(data)?;
    stream.flush()
}
fn read_packet(stream: &mut impl Read) -> io::Result<Packet> {
    let mut header = [0; 5];
    stream.read_exact(&mut header)?;
    let size = u32::from_le_bytes(header[1..].try_into().unwrap()) as usize;
    if size > MAX_PACKET {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native packet too large",
        ));
    }
    let mut data = vec![0; size];
    stream.read_exact(&mut data)?;
    Ok(Packet {
        kind: header[0],
        data,
    })
}
fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}
fn pcm_samples(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unaligned PCM"));
    }
    let samples: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "non-finite PCM"));
    }
    Ok(samples)
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Process {
    child: OwnedChild,
    socket: TcpStream,
    rx: Receiver<io::Result<Packet>>,
    reader: Option<thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}
impl Process {
    fn spawn(kind: &str, root: &Path) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let mut nonce = [0u8; 32];
        getrandom::fill(&mut nonce).map_err(io::Error::other)?;
        let nonce_hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("--native-worker")
            .arg(kind)
            .env("ZEN_WORKER_ADDR", listener.local_addr()?.to_string())
            .env("ZEN_WORKER_NONCE", &nonce_hex)
            .env("ZEN_WORKER_ROOT", root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let spawned = command.spawn()?;
        // These already exit when the command socket closes, but a worker wedged inside a
        // DLL loader never reaches the code that would notice.
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            crate::job::adopt(spawned.as_raw_handle());
        }
        let mut child = OwnedChild(spawned);
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut socket = loop {
            if let Some(status) = child.0.try_wait()? {
                return Err(io::Error::other(format!("{kind} worker exited: {status}")));
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("{kind} worker startup timed out"),
                ));
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false)?;
                    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                    if read_packet(&mut stream)
                        .is_ok_and(|p| p.kind == HELLO && p.data == nonce_hex.as_bytes())
                    {
                        break stream;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10))
                }
                Err(e) => return Err(e),
            }
        };
        socket.set_nodelay(true)?;
        socket.set_read_timeout(Some(
            deadline
                .saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1)),
        ))?;
        let ready = read_packet(&mut socket)?;
        if ready.kind != READY {
            return Err(io::Error::other(format!(
                "{kind} initialization failed: {}",
                String::from_utf8_lossy(&ready.data)
            )));
        }
        socket.set_read_timeout(None)?;
        socket.set_write_timeout(Some(Duration::from_secs(3)))?;
        let mut input = socket.try_clone()?;
        let (tx, rx) = mpsc::sync_channel(8);
        let stop = Arc::new(AtomicBool::new(false));
        let quit = Arc::clone(&stop);
        let reader = thread::Builder::new()
            .name(format!("zen-{kind}-ipc"))
            .spawn(move || {
                while !quit.load(Ordering::Acquire) {
                    let mut packet = read_packet(&mut input);
                    let failed = packet.is_err();
                    loop {
                        match tx.try_send(packet) {
                            Ok(()) => break,
                            Err(TrySendError::Full(value)) => packet = value,
                            Err(TrySendError::Disconnected(_)) => return,
                        }
                        if quit.load(Ordering::Acquire) {
                            return;
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                    if failed {
                        break;
                    }
                }
            })?;
        Ok(Self {
            child,
            socket,
            rx,
            reader: Some(reader),
            stop,
        })
    }
    fn request(
        &mut self,
        kind: u8,
        data: &[u8],
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(Packet) -> io::Result<bool>,
    ) -> io::Result<bool> {
        Self::exchange(&mut self.socket, &self.rx, kind, data, cancelled, callback)
    }

    fn exchange(
        socket: &mut TcpStream,
        rx: &Receiver<io::Result<Packet>>,
        kind: u8,
        data: &[u8],
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(Packet) -> io::Result<bool>,
    ) -> io::Result<bool> {
        write_packet(socket, kind, data)?;
        let mut progress = Instant::now();
        let mut cancelling = None;
        loop {
            if cancelled.load(Ordering::Acquire) && cancelling.is_none() {
                write_packet(socket, CANCEL, &[])?;
                cancelling = Some(Instant::now());
            }
            if cancelling.is_some_and(|t: Instant| t.elapsed() > Duration::from_secs(2)) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "native cancellation deadline exceeded",
                ));
            }
            if progress.elapsed() > Duration::from_secs(30) {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "native worker stopped making progress",
                ));
            }
            match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(Ok(packet)) => {
                    progress = Instant::now();
                    if packet.kind == DONE {
                        return Ok(cancelling.is_none());
                    }
                    if packet.kind == ERROR {
                        return Err(io::Error::other(
                            String::from_utf8_lossy(&packet.data).into_owned(),
                        ));
                    }
                    if cancelling.is_none() && !callback(packet)? {
                        write_packet(socket, CANCEL, &[])?;
                        cancelling = Some(Instant::now());
                    }
                }
                Ok(Err(e)) => return Err(e),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "native worker disconnected",
                    ))
                }
            }
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.socket.shutdown(std::net::Shutdown::Both);
        let _ = self.child.0.kill();
        let _ = self.child.0.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

pub struct NativeEngine {
    kind: &'static str,
    root: PathBuf,
    process: Mutex<Option<Process>>,
}
impl NativeEngine {
    pub fn load(kind: &'static str, root: &Path) -> io::Result<Self> {
        Ok(Self {
            kind,
            root: root.to_path_buf(),
            process: Mutex::new(Some(Process::spawn(kind, root)?)),
        })
    }
    fn request(
        &self,
        kind: u8,
        data: &[u8],
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(Packet) -> io::Result<bool>,
    ) -> io::Result<()> {
        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
        }
        let mut process = self
            .process
            .lock()
            .map_err(|_| io::Error::other("native process lock poisoned"))?;
        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
        }
        if process.is_none() {
            *process = Some(Process::spawn(self.kind, &self.root)?);
        }
        let result = process
            .as_mut()
            .unwrap()
            .request(kind, data, cancelled, callback);
        // Destroy an uncertain worker; the next request recreates it in a clean process.
        match result {
            Ok(true) => Ok(()),
            Ok(false) => Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled")),
            Err(error) => {
                process.take();
                Err(error)
            }
        }
    }
}
impl Recognizer for NativeEngine {
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
        self.transcribe_cancellable(samples, &AtomicBool::new(false))
    }
    fn transcribe_cancellable(
        &self,
        samples: &[f32],
        cancelled: &AtomicBool,
    ) -> Result<Transcript, AsrError> {
        let start = Instant::now();
        let mut text = None;
        self.request(ASR, &pcm_bytes(samples), cancelled, &mut |packet| {
            if packet.kind != TEXT {
                return Err(io::Error::other("unexpected ASR packet"));
            }
            text = Some(String::from_utf8(packet.data).map_err(io::Error::other)?);
            Ok(true)
        })
        .map_err(|e| AsrError::Load(format!("native ASR: {e}")))?;
        Ok(Transcript {
            text: text.ok_or(AsrError::NoResult)?,
            audio_ms: samples.len() * 1000 / 16000,
            latency_ms: start.elapsed().as_secs_f32() * 1000.0,
        })
    }
}
impl Synthesizer for NativeEngine {
    fn synthesize_cancellable(
        &self,
        text: &str,
        cancelled: &AtomicBool,
        callback: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), TtsError> {
        self.request(TTS, text.as_bytes(), cancelled, &mut |packet| {
            if packet.kind != AUDIO {
                return Err(io::Error::other("unexpected TTS packet"));
            }
            Ok(callback(&pcm_samples(&packet.data)?))
        })
        .map_err(|e| {
            if cancelled.load(Ordering::Acquire) {
                TtsError::Cancelled
            } else {
                TtsError::Synthesis(format!("native TTS: {e}"))
            }
        })
    }
}

/// Internal entry point. It never opens microphone/output devices or launches another worker.
pub fn worker_entry(kind: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let address = std::env::var("ZEN_WORKER_ADDR")?.parse::<std::net::SocketAddr>()?;
    if !address.ip().is_loopback() {
        return Err("worker address must be loopback".into());
    }
    let nonce = std::env::var("ZEN_WORKER_NONCE")?;
    let root = PathBuf::from(std::env::var_os("ZEN_WORKER_ROOT").ok_or("missing worker root")?);
    let mut socket = TcpStream::connect_timeout(&address, Duration::from_secs(5))?;
    socket.set_nodelay(true)?;
    socket.set_write_timeout(Some(Duration::from_secs(3)))?;
    write_packet(&mut socket, HELLO, nonce.as_bytes())?;
    let mut commands = socket.try_clone()?;
    let (tx, rx) = mpsc::sync_channel(1);
    let active = Arc::new(Mutex::new(Arc::new(AtomicBool::new(false))));
    let cancel = Arc::clone(&active);
    // Establish parent-death handling before loading any native DLL. Even a loader deadlock
    // cannot leave a worker behind after its parent closes the connection.
    thread::spawn(move || loop {
        let packet = match read_packet(&mut commands) {
            Ok(packet) => packet,
            Err(_) => std::process::exit(0),
        };
        if packet.kind == CANCEL {
            cancel.lock().unwrap().store(true, Ordering::Release);
            continue;
        }
        if !matches!(packet.kind, ASR | TTS) {
            std::process::exit(2);
        }
        let token = Arc::new(AtomicBool::new(false));
        *cancel.lock().unwrap() = Arc::clone(&token);
        if tx.send((packet, token)).is_err() {
            return;
        }
    });
    enum Engine {
        Asr(AsrEngine),
        Tts(TtsEngine),
    }
    eprintln!("Loading {kind} native runtime...");
    let engine = match kind {
        "asr" => AsrEngine::load(root.join("model/Qwen ASR"), AsrEngine::default_threads())
            .map(Engine::Asr)
            .map_err(|e| e.to_string()),
        "tts" => TtsEngine::load(root.join("lib/qwen.dll"), root.join("model/Qwen TTS"))
            .map(Engine::Tts)
            .map_err(|e| e.to_string()),
        _ => Err("unknown native worker".into()),
    };
    let engine = match engine {
        Ok(engine) => engine,
        Err(e) => {
            write_packet(&mut socket, ERROR, e.as_bytes())?;
            return Err(e.into());
        }
    };
    write_packet(&mut socket, READY, &[])?;
    eprintln!("{kind} native runtime ready");
    while let Ok((packet, token)) = rx.recv() {
        let result: Result<(), String> = match &engine {
            Engine::Asr(engine) if packet.kind == ASR => (|| {
                let samples = pcm_samples(&packet.data).map_err(|e| e.to_string())?;
                let transcript = engine.transcribe(&samples).map_err(|e| e.to_string())?;
                write_packet(&mut socket, TEXT, transcript.text.as_bytes())
                    .map_err(|e| e.to_string())
            })(),
            Engine::Tts(engine) if packet.kind == TTS => (|| {
                let text = String::from_utf8(packet.data).map_err(|e| e.to_string())?;
                engine
                    .synthesize_cancellable(&text, &token, &mut |samples| {
                        samples.chunks(6000).all(|chunk| {
                            write_packet(&mut socket, AUDIO, &pcm_bytes(chunk)).is_ok()
                        })
                    })
                    .map_err(|e| e.to_string())
            })(),
            _ => Err("request type does not match worker".into()),
        };
        if let Err(e) = result {
            if !token.load(Ordering::Acquire) {
                write_packet(&mut socket, ERROR, e.as_bytes())?;
            }
        }
        write_packet(&mut socket, DONE, &[])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_pcm_round_trips_and_rejects_invalid_lengths() {
        let pcm = vec![-0.5, 0.0, 0.25];
        let mut wire = Vec::new();
        write_packet(&mut wire, AUDIO, &pcm_bytes(&pcm)).unwrap();
        let packet = read_packet(&mut &wire[..]).unwrap();
        assert_eq!(pcm_samples(&packet.data).unwrap(), pcm);
        assert!(pcm_samples(&[1, 2, 3]).is_err());
        assert!(pcm_samples(&f32::NAN.to_le_bytes()).is_err());
    }
    #[test]
    fn packet_size_is_validated_before_allocation() {
        let mut wire = vec![AUDIO];
        wire.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_packet(&mut &wire[..]).is_err());
    }

    fn socket_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        (client, server)
    }

    #[test]
    fn cancelled_audio_is_drained_before_reusing_the_connection() {
        let (mut client, mut server) = socket_pair();
        let (tx, rx) = mpsc::channel();
        // Audio can already be queued when cancellation occurs. DONE bounds that request.
        for (kind, data) in [
            (AUDIO, "old"),
            (AUDIO, "stale"),
            (DONE, ""),
            (AUDIO, "new"),
            (DONE, ""),
        ] {
            tx.send(Ok(Packet {
                kind,
                data: data.as_bytes().to_vec(),
            }))
            .unwrap();
        }
        let token = AtomicBool::new(false);
        let mut delivered = Vec::new();
        assert!(
            !Process::exchange(&mut client, &rx, TTS, b"first", &token, &mut |p| {
                delivered.push(p.data);
                Ok(false)
            })
            .unwrap()
        );
        assert!(
            Process::exchange(&mut client, &rx, TTS, b"second", &token, &mut |p| {
                delivered.push(p.data);
                Ok(true)
            })
            .unwrap()
        );
        assert_eq!(delivered, [b"old".to_vec(), b"new".to_vec()]);
        assert_eq!(read_packet(&mut server).unwrap().kind, TTS);
        assert_eq!(read_packet(&mut server).unwrap().kind, CANCEL);
        assert_eq!(read_packet(&mut server).unwrap().data, b"second");
    }

    #[test]
    fn disconnected_worker_never_reports_success() {
        let (mut client, _server) = socket_pair();
        let (tx, rx) = mpsc::channel();
        tx.send(Ok(Packet {
            kind: AUDIO,
            data: Vec::new(),
        }))
        .unwrap();
        drop(tx);
        let error = Process::exchange(
            &mut client,
            &rx,
            TTS,
            b"hello",
            &AtomicBool::new(false),
            &mut |_| Ok(true),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn cancelled_job_does_not_launch_a_replacement_worker() {
        let engine = NativeEngine {
            kind: "tts",
            root: PathBuf::from("missing-test-root"),
            process: Mutex::new(None),
        };
        let error = engine
            .request(TTS, b"hello", &AtomicBool::new(true), &mut |_| Ok(true))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(engine.process.lock().unwrap().is_none());
    }
}
