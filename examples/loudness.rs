//! Measures Zen's voice the way a listener gets it: how big the pieces are that synthesis hands
//! over, how loud the result is by ITU-R BS.1770-4, and how hard the output limiter has to work
//! at a given makeup gain. Local tool.
//!
//!   loudness render <dir>          synthesise the passages, raw, and log each delivered piece
//!   loudness measure <dir> [gain…] loudness and limiter activity at each gain
use std::{path::Path, sync::atomic::AtomicBool, time::Instant};
use zen::{
    audio::Loudness,
    reply::to_speakable,
    tts::{TtsEngine, TTS_SAMPLE_RATE},
};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const PASSAGES: &[&str] = &[
    "That sounds like a really common struggle, and it is not the planning itself that causes it. When every hour of the week is already spoken for, a single change breaks the whole shape of the day. Leaving some of the time open is what lets a plan survive an ordinary week.",
    "Sure. Boil the water first, then add the pasta and stir it once so it does not stick. Give it about ten minutes. Taste a piece before you drain it, because the packet time is only a guess.",
    "Honestly? I think you should take the offer. The pay is better, the commute is shorter, and you said yourself that the team felt friendly. What is actually holding you back?",
    "The Moon is about 384,000 km away, which is roughly thirty Earths lined up in a row. Light covers that in just over a second. The Apollo crews took three days.",
    "Okay, here is the short version. Sleep matters more than any supplement. Keep the room cool and dark, put the phone somewhere else, and try to wake at the same time every day, even on weekends. It feels boring, but it works.",
    "Hmm, I am not sure that is right. A spider is not an insect at all. Insects have six legs and three body parts, while spiders have eight legs and two. They are closer relatives of scorpions than of ants.",
    "Good morning! It is going to be a busy one. You have the dentist at 9:30, lunch with Sam at noon, and the report is due by five. Do you want me to go through the report first?",
    // As long as the chunker ever lets a phrase grow: ninety words.
    "Here is how I would think about it. Start with what you already know you want, which is more time in the mornings and less of the evening spent on email. Then look at the week as it really runs, not as the calendar says it runs, and notice where the time actually goes. Most people find two or three habits that quietly eat an hour a day. Pick one of them, change only that, and give it a fortnight before you judge it.",
];

fn read(path: &Path) -> R<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    Ok(reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?)
}

fn write(path: &Path, samples: &[f32]) -> R<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TTS_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// One biquad, direct form I, in f64 so the filter itself adds nothing to the measurement.
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
    x: [f64; 2],
    y: [f64; 2],
}

impl Biquad {
    fn run(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[1] * self.y[0]
            - self.a[2] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// The K-weighting of BS.1770-4, solved for `rate` from the standard's analogue prototype (the
/// same derivation pyloudnorm uses), since the tabulated coefficients are for 48 kHz only.
fn k_weighting(rate: f64) -> [Biquad; 2] {
    let shelf = {
        let (gain, q, fc) = (
            3.999_843_853_973_347,
            0.707_175_236_955_419_3,
            1_681.974_450_955_533,
        );
        let a = 10f64.powf(gain / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let (c, s) = (w0.cos(), a.sqrt());
        let a0 = (a + 1.0) - (a - 1.0) * c + 2.0 * s * alpha;
        Biquad {
            b: [
                a * ((a + 1.0) + (a - 1.0) * c + 2.0 * s * alpha) / a0,
                -2.0 * a * ((a - 1.0) + (a + 1.0) * c) / a0,
                a * ((a + 1.0) + (a - 1.0) * c - 2.0 * s * alpha) / a0,
            ],
            a: [
                1.0,
                2.0 * ((a - 1.0) - (a + 1.0) * c) / a0,
                ((a + 1.0) - (a - 1.0) * c - 2.0 * s * alpha) / a0,
            ],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    };
    let high = {
        let (q, fc) = (0.500_327_037_323_877_3, 38.135_470_876_024_44);
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let c = w0.cos();
        let a0 = 1.0 + alpha;
        Biquad {
            b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0],
            a: [1.0, -2.0 * c / a0, (1.0 - alpha) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    };
    [shelf, high]
}

/// Gated integrated loudness, LUFS: 400 ms blocks overlapping by 75 %, an absolute gate at
/// -70 LUFS and a relative gate 10 LU under the level of what passed it.
fn integrated(samples: &[f32]) -> f64 {
    let rate = TTS_SAMPLE_RATE as f64;
    let [mut shelf, mut high] = k_weighting(rate);
    let weighted: Vec<f64> = samples
        .iter()
        .map(|&s| high.run(shelf.run(s as f64)))
        .collect();
    let block = (0.4 * rate) as usize;
    let step = block / 4;
    let mut powers = Vec::new();
    let mut start = 0;
    while start + block <= weighted.len() {
        powers.push(
            weighted[start..start + block]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                / block as f64,
        );
        start += step;
    }
    let lufs = |p: f64| -0.691 + 10.0 * p.log10();
    let gated = |threshold: f64| -> Vec<f64> {
        powers
            .iter()
            .copied()
            .filter(|&p| lufs(p) > threshold)
            .collect()
    };
    let absolute = gated(-70.0);
    if absolute.is_empty() {
        return f64::NEG_INFINITY;
    }
    let relative = lufs(absolute.iter().sum::<f64>() / absolute.len() as f64) - 10.0;
    let kept = gated(relative);
    lufs(kept.iter().sum::<f64>() / kept.len() as f64)
}

/// True peak, measured independently of the limiter it checks: sixteen steps between samples
/// through a Hann-windowed sinc thirty-two taps wide.
fn true_peak(samples: &[f32]) -> f32 {
    const PHASES: usize = 16;
    const HALF: isize = 16;
    let mut peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    for i in 0..samples.len() as isize {
        for phase in 1..PHASES {
            let t = phase as f64 / PHASES as f64;
            let mut v = 0.0f64;
            for k in -HALF + 1..=HALF {
                let at = i + k;
                if at < 0 || at >= samples.len() as isize {
                    continue;
                }
                let x = k as f64 - t;
                let sinc = if x == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let r = x / HALF as f64;
                let window = if r.abs() >= 1.0 {
                    0.0
                } else {
                    0.5 + 0.5 * (std::f64::consts::PI * r).cos()
                };
                v += samples[at as usize] as f64 * sinc * window;
            }
            peak = peak.max(v.abs() as f32);
        }
    }
    peak
}

fn render(dir: &Path) -> R<()> {
    std::fs::create_dir_all(dir)?;
    let root = zen::runtime::discover_root();
    let engine = TtsEngine::load(root.join("lib/qwen.dll"), root.join("model/Qwen TTS"))?;
    let never = AtomicBool::new(false);
    let mut largest = 0usize;
    for (index, passage) in PASSAGES.iter().enumerate() {
        let text = to_speakable(passage);
        let mut samples = Vec::new();
        let mut pieces = Vec::new();
        let started = Instant::now();
        let mut first = None;
        engine.synthesize_cancellable(&text, &never, &mut |piece| {
            first.get_or_insert(started.elapsed());
            pieces.push(piece.len());
            samples.extend_from_slice(piece);
            true
        })?;
        let took = started.elapsed();
        largest = largest.max(pieces.iter().copied().max().unwrap_or(0));
        write(&dir.join(format!("raw-{index}.wav")), &samples)?;
        let ms = |n: usize| n * 1000 / TTS_SAMPLE_RATE as usize;
        let audio = samples.len() as f64 / TTS_SAMPLE_RATE as f64;
        println!(
            "raw-{index}: {:.1} s audio in {:.1} s (x{:.2} real time), first piece after {} ms, {} words\n  pieces (ms): {:?}",
            audio,
            took.as_secs_f64(),
            audio / took.as_secs_f64(),
            first.map_or(0, |f| f.as_millis()),
            text.split_whitespace().count(),
            pieces.iter().map(|&n| ms(n)).collect::<Vec<_>>()
        );
    }
    println!(
        "largest single piece: {} ms",
        largest * 1000 / TTS_SAMPLE_RATE as usize
    );
    Ok(())
}

fn measure(dir: &Path, gains: &[f32]) -> R<()> {
    let mut passages = Vec::new();
    for index in 0.. {
        let path = dir.join(format!("raw-{index}.wav"));
        if !path.exists() {
            break;
        }
        passages.push(read(&path)?);
    }
    let all: Vec<f32> = passages.concat();
    println!(
        "raw: {:.2} LUFS integrated, true peak {:.3} ({:.2} dBTP), {} passages, {:.1} s",
        integrated(&all),
        true_peak(&all),
        20.0 * true_peak(&all).log10(),
        passages.len(),
        all.len() as f64 / TTS_SAMPLE_RATE as f64
    );
    for &gain in gains {
        let mut out = Vec::new();
        // How far under the plain gain the limiter held each sample, with its delay removed.
        let mut reductions = Vec::new();
        for passage in &passages {
            let mut limiter = Loudness::new(gain);
            let mut samples = passage.clone();
            limiter.process(&mut samples);
            samples.extend(limiter.finish());
            let delay = Loudness::LATENCY;
            for (i, &raw) in passage.iter().enumerate() {
                let wanted = raw * gain;
                if wanted.abs() > 0.05 {
                    reductions.push((samples[i + delay] / wanted).clamp(0.0, 1.0));
                }
            }
            out.extend_from_slice(&samples);
        }
        let limited = |db: f32| {
            let floor = 10f32.powf(-db / 20.0);
            100.0 * reductions.iter().filter(|&&r| r < floor).count() as f32
                / reductions.len() as f32
        };
        let deepest = reductions.iter().copied().fold(1.0f32, f32::min);
        println!(
            "gain {gain:.2}: {:.2} LUFS, true peak {:.2} dBTP, limiter engaged on {:.2} % of voiced samples (>0.5 dB on {:.2} %, >1 dB on {:.2} %, >3 dB on {:.2} %), deepest {:.1} dB",
            integrated(&out),
            20.0 * true_peak(&out).log10(),
            limited(0.01),
            limited(0.5),
            limited(1.0),
            limited(3.0),
            20.0 * deepest.log10()
        );
        write(&dir.join(format!("gain-{gain:.2}.wav")), &out)?;
    }
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("render") if args.len() == 2 => render(Path::new(&args[1])),
        Some("measure") if args.len() >= 2 => {
            let gains: Vec<f32> = if args.len() > 2 {
                args[2..]
                    .iter()
                    .map(|g| g.parse())
                    .collect::<Result<_, _>>()?
            } else {
                vec![1.65, 1.85, Loudness::DEFAULT_GAIN, 2.2, 2.4]
            };
            measure(Path::new(&args[1]), &gains)
        }
        _ => Err("usage: loudness render <dir> | loudness measure <dir> [gain…]".into()),
    }
}
