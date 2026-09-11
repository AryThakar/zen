// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Cancellable synthesis on a worker thread. Playback and its acknowledgements belong to
//! the webview, which owns the output device.
use crate::{session::Generation, tts::Synthesizer};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub enum VoiceEvent {
    Begin(Generation, String),
    Audio(Generation, Vec<f32>),
    End(Generation, String),
    Failed(Generation, String),
}
struct Job {
    generation: Generation,
    text: String,
    cancelled: Arc<AtomicBool>,
}

pub struct VoiceWorker {
    jobs: Option<SyncSender<Job>>,
    pub events: Receiver<VoiceEvent>,
    handle: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    progress: Arc<AtomicU64>,
    epoch: Instant,
    queued: Arc<AtomicUsize>,
}

fn send_cancelled<T>(
    sender: &SyncSender<T>,
    mut value: T,
    cancelled: &AtomicBool,
    shutdown: &AtomicBool,
) -> bool {
    while !cancelled.load(Ordering::Acquire) && !shutdown.load(Ordering::Acquire) {
        match sender.try_send(value) {
            Ok(()) => return true,
            Err(TrySendError::Full(v)) => value = v,
            Err(TrySendError::Disconnected(_)) => return false,
        }
        thread::sleep(Duration::from_millis(2));
    }
    false
}

impl VoiceWorker {
    pub fn spawn(engine: Arc<dyn Synthesizer>) -> Result<Self, std::io::Error> {
        let (jobs, rx) = mpsc::sync_channel::<Job>(16);
        let (tx, events) = mpsc::sync_channel(8);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&shutdown);
        let progress = Arc::new(AtomicU64::new(0));
        let clock = Arc::clone(&progress);
        let epoch = Instant::now();
        let queued = Arc::new(AtomicUsize::new(0));
        let queue_count = queued.clone();
        let handle = thread::Builder::new()
            .name("zen-voice".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    queue_count.fetch_sub(1, Ordering::AcqRel);
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if job.cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    clock.store(epoch.elapsed().as_millis() as u64 + 1, Ordering::Release);
                    if !send_cancelled(
                        &tx,
                        VoiceEvent::Begin(job.generation, job.text.clone()),
                        &job.cancelled,
                        &stop,
                    ) {
                        clock.store(0, Ordering::Release);
                        continue;
                    }
                    let mut produced = 0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        engine.synthesize_cancellable(&job.text, &job.cancelled, &mut |samples| {
                            // Bound allocations even if an incompatible native build returns bad lengths.
                            if samples.len() > crate::tts::TTS_SAMPLE_RATE as usize * 2 {
                                return false;
                            }
                            for block in samples.chunks(crate::tts::TTS_SAMPLE_RATE as usize / 4) {
                                if !send_cancelled(
                                    &tx,
                                    VoiceEvent::Audio(job.generation, block.to_vec()),
                                    &job.cancelled,
                                    &stop,
                                ) {
                                    return false;
                                }
                                produced += block.len();
                                clock.store(
                                    epoch.elapsed().as_millis() as u64 + 1,
                                    Ordering::Release,
                                );
                            }
                            true
                        })
                    }));
                    let event = match result {
                        Ok(Ok(())) if produced > 0 => VoiceEvent::End(job.generation, job.text),
                        Ok(Ok(())) => {
                            VoiceEvent::Failed(job.generation, "synthesis produced no audio".into())
                        }
                        Ok(Err(e)) => VoiceEvent::Failed(job.generation, e.to_string()),
                        Err(_) => {
                            VoiceEvent::Failed(job.generation, "synthesis worker panicked".into())
                        }
                    };
                    send_cancelled(&tx, event, &job.cancelled, &stop);
                    clock.store(0, Ordering::Release);
                }
            })?;
        Ok(Self {
            jobs: Some(jobs),
            events,
            handle: Some(handle),
            shutdown,
            progress,
            epoch,
            queued,
        })
    }
    pub fn submit(
        &self,
        generation: Generation,
        text: String,
        cancelled: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let sender = self.jobs.as_ref().ok_or("synthesis worker is closed")?;
        self.queued.fetch_add(1, Ordering::AcqRel);
        sender
            .try_send(Job {
                generation,
                text,
                cancelled,
            })
            .map_err(|e| {
                self.queued.fetch_sub(1, Ordering::AcqRel);
                match e {
                    TrySendError::Full(_) => "synthesis queue reached its limit",
                    TrySendError::Disconnected(_) => "synthesis worker disconnected",
                }
                .into()
            })
    }
    pub fn can_submit(&self) -> bool {
        self.jobs.is_some() && self.queued.load(Ordering::Acquire) < 16
    }
    pub fn stalled(&self) -> bool {
        let progress = self.progress.load(Ordering::Acquire);
        progress > 0
            && (self.epoch.elapsed().as_millis() as u64 + 1).saturating_sub(progress) > 45_000
    }
    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }
    pub fn shutdown(&mut self, timeout: Duration) -> bool {
        self.shutdown.store(true, Ordering::Release);
        self.jobs.take();
        let deadline = Instant::now() + timeout;
        while !self.is_finished() && Instant::now() < deadline {
            while self.events.try_recv().is_ok() {}
            thread::sleep(Duration::from_millis(5));
        }
        if self.is_finished() {
            return self.handle.take().is_none_or(|h| h.join().is_ok());
        }
        false
    }
}

impl Drop for VoiceWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.jobs.take();
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_unblocks_a_full_native_audio_channel() {
        let (tx, _rx) = mpsc::sync_channel(1);
        tx.send(1).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let token = Arc::clone(&cancelled);
        let h = thread::spawn(move || send_cancelled(&tx, 2, &token, &AtomicBool::new(false)));
        cancelled.store(true, Ordering::Release);
        assert!(!h.join().unwrap());
    }
}
