// SPDX-License-Identifier: Apache-2.0
//! Stateful, band-limited rate conversion with an explicit tail drain.
use crate::audio::AudioError;
use rubato::{audioadapter_buffers::direct::InterleavedSlice, Resampler};

pub struct StreamResampler {
    inner: rubato::Fft<f32>,
    from: u32,
    to: u32,
    pending: Vec<f32>,
    scratch: Vec<f32>,
    trim: usize,
    input: usize,
    output: usize,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Result<Self, AudioError> {
        if !(8_000..=384_000).contains(&from) || !(8_000..=384_000).contains(&to) {
            return Err(AudioError::Resample(
                "rates must be between 8000 and 384000 Hz".into(),
            ));
        }
        let inner = rubato::Fft::new(
            from as usize,
            to as usize,
            (from as usize / 100).max(64),
            1,
            rubato::FixedSync::Input,
        )
        .map_err(|e| AudioError::Resample(e.to_string()))?;
        let trim = inner.output_delay();
        let capacity = inner.output_frames_max();
        Ok(Self {
            inner,
            from,
            to,
            pending: Vec::new(),
            scratch: vec![0.0; capacity],
            trim,
            input: 0,
            output: 0,
        })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.trim = self.inner.output_delay();
        self.pending.clear();
        self.input = 0;
        self.output = 0;
    }

    pub fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) -> Result<(), AudioError> {
        self.input += samples.len();
        self.pending.extend_from_slice(samples);
        self.process(out)
    }

    fn process(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let mut consumed = 0;
        while self.pending.len() - consumed >= self.inner.input_frames_next() {
            let count = self.inner.input_frames_next();
            let input = InterleavedSlice::new(&self.pending[consumed..consumed + count], 1, count)
                .map_err(|e| AudioError::Resample(e.to_string()))?;
            let capacity = self.scratch.len();
            let mut output = InterleavedSlice::new_mut(&mut self.scratch, 1, capacity)
                .map_err(|e| AudioError::Resample(e.to_string()))?;
            let (used, made) = self
                .inner
                .process_into_buffer(&input, &mut output, None)
                .map_err(|e| AudioError::Resample(e.to_string()))?;
            let skip = self.trim.min(made);
            self.trim -= skip;
            out.extend_from_slice(&self.scratch[skip..made]);
            self.output += made - skip;
            consumed += used;
        }
        self.pending.drain(..consumed);
        Ok(())
    }

    pub fn finish(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let target = ((self.input as u64 * self.to as u64 + self.from as u64 / 2)
            / self.from as u64) as usize;
        let needed = target.saturating_sub(self.output);
        let start = out.len();
        while self.output < target {
            self.pending.resize(self.inner.input_frames_next(), 0.0);
            self.process(out)?;
        }
        out.truncate(start + needed);
        self.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_device_rates_preserve_duration_and_stream_chunk_boundaries() {
        for (from, to) in [
            (24000, 44100),
            (24000, 48000),
            (48000, 16000),
            (44100, 48000),
        ] {
            let input: Vec<_> = (0..from)
                .map(|i| (i as f32 * 1000.0 * std::f32::consts::TAU / from as f32).sin() * 0.2)
                .collect();
            let convert = |chunks: usize| {
                let mut r = StreamResampler::new(from, to).unwrap();
                let mut out = Vec::new();
                for chunk in input.chunks(chunks) {
                    r.push(chunk, &mut out).unwrap();
                }
                r.finish(&mut out).unwrap();
                out
            };
            let a = convert(137);
            let b = convert(input.len());
            assert_eq!(a.len(), to as usize);
            assert_eq!(a, b);
            let rms = (a.iter().map(|s| s * s).sum::<f32>() / a.len() as f32).sqrt();
            assert!((rms - 0.1414).abs() < 0.005, "signal level changed: {rms}");
        }
    }
    #[test]
    fn even_sub_frame_audio_and_last_consonant_survive_the_tail_drain() {
        let mut r = StreamResampler::new(24000, 44100).unwrap();
        let mut out = Vec::new();
        r.push(&[0.2; 120], &mut out).unwrap();
        r.finish(&mut out).unwrap();
        assert_eq!(out.len(), 221);
        assert!(out.iter().any(|s| s.abs() > 0.1));
    }
}
