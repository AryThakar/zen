//! Local tool: runs transcripts through the filter exactly as a live turn does - the same rules,
//! slot, budget and temperature - and prints each verdict, so a change to `prompts/filter.txt`
//! can be checked on the real model before it ships.
//!
//!   cargo run --release --example filterprobe -- "transcript one" "transcript two" ...
use zen::{
    engine::{LlamaConfig, LlamaEngine, SlotKind},
    reply::{parse_filter_verdict, FilterVerdict},
};

const RULES: &str = include_str!("../src/prompts/filter.txt");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let transcripts: Vec<String> = std::env::args().skip(1).collect();
    if transcripts.is_empty() {
        return Err("usage: filterprobe TRANSCRIPT...".into());
    }
    let llama = LlamaEngine::new(LlamaConfig::from_zen_root(zen::runtime::discover_root()))?;
    llama.start().await?;
    for transcript in &transcripts {
        let messages = vec![
            ("system", RULES.to_string()),
            (
                "user",
                serde_json::json!({ "transcript": transcript }).to_string(),
            ),
        ];
        // `remote::filter_budget`, which is crate-private.
        let budget = (transcript.chars().count() + 64).clamp(128, 512);
        let reply = llama
            .client()
            .stream_completion_with(SlotKind::Filter, &messages, budget, 0.1, |_| true)
            .await?;
        let verdict = match parse_filter_verdict(&reply.text) {
            FilterVerdict::Clean(text) => format!("CLEAN {text}"),
            FilterVerdict::Ask(question) => format!("ASK   {question}"),
        };
        println!("{transcript:55} -> {verdict}");
    }
    llama.stop().await?;
    Ok(())
}
