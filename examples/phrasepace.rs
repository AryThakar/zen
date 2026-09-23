//! Renders reply-shaped passages so the pace of speech inside one phrase can be measured:
//! where each sentence actually ends in the audio, against where its share of the text says it
//! should. That decides how much of an interrupted phrase Zen may claim was heard.
//! Local tool.
use std::path::PathBuf;
use zen::{
    reply::to_speakable,
    tts::{TtsEngine, TTS_SAMPLE_RATE},
};

const PASSAGES: &[&str] = &[
    "That sounds like a really common struggle, and it is not the planning itself that causes it. When every hour of the week is already spoken for, a single change breaks the whole shape of the day. Leaving some of the time open is what lets a plan survive an ordinary week.",
    "Sure. Boil the water first, then add the pasta and stir it once so it does not stick. Give it about ten minutes. Taste a piece before you drain it, because the packet time is only a guess.",
    "Honestly? I think you should take the offer. The pay is better, the commute is shorter, and you said yourself that the team felt friendly. What is actually holding you back?",
    "The Moon is about 384,000 km away, which is roughly thirty Earths lined up in a row. Light covers that in just over a second. The Apollo crews took three days.",
    "Okay, here is the short version. Sleep matters more than any supplement. Keep the room cool and dark, put the phone somewhere else, and try to wake at the same time every day, even on weekends. It feels boring, but it works.",
    "Hmm, I am not sure that is right. A spider is not an insect at all. Insects have six legs and three body parts, while spiders have eight legs and two. They are closer relatives of scorpions than of ants.",
    "Good morning! It is going to be a busy one. You have the dentist at 9:30, lunch with Sam at noon, and the report is due by five. Do you want me to go through the report first?",
    "Yes, you can. Just keep it simple. Open the settings, choose the microphone you want, and Zen will switch to it straight away. If you unplug it later, it falls back to the default one.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: phrasepace <out dir>")?,
    );
    std::fs::create_dir_all(&out)?;
    let root = zen::runtime::discover_root();
    let engine = TtsEngine::load(root.join("lib/qwen.dll"), root.join("model/Qwen TTS"))?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TTS_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    for (index, passage) in PASSAGES.iter().enumerate() {
        let text = to_speakable(passage);
        let samples = engine.synthesize(&text)?;
        let mut writer = hound::WavWriter::create(out.join(format!("pace-{index}.wav")), spec)?;
        for sample in &samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
        writer.finalize()?;
        std::fs::write(out.join(format!("pace-{index}.txt")), &text)?;
        println!(
            "pace-{index}: {:.1} s, {} chars",
            samples.len() as f32 / TTS_SAMPLE_RATE as f32,
            text.chars().count()
        );
    }
    Ok(())
}
