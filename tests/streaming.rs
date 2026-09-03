//! Streaming regression test, on the model, in its own process.
//!
//! ```text
//! QTB_MODEL_DIR=/path/to/model cargo test --release --test streaming -- --ignored
//! ```
//!
//! It lives in its own test binary on purpose: after three other tests have
//! each loaded and dropped an engine in the same process, the pool pages they
//! left behind push generation below realtime, the lead buffer grows to cover
//! the whole utterance, and this test fails for a reason that has nothing to
//! do with streaming. Loading several engines into one process is not a
//! supported pattern.

use std::path::PathBuf;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::sampling::SamplerCfg;

const SEED: u64 = 77;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> String {
    std::env::var("QTB_MODEL_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
        .expect("streaming: set QTB_MODEL_DIR to the 1.7B Base model directory")
}

#[test]
#[ignore = "needs QTB_MODEL_DIR and a Vulkan GPU"]
fn streaming_hands_back_audio_before_the_utterance_is_finished() {
    let engine = qwen3_tts_burn::load_vulkan(&model_dir()).expect("load model");
    engine.warmup().expect("warmup");
    let ref_wav = root().join("samples").join("reference_en.wav");
    let ref_text = std::fs::read_to_string(root().join("samples").join("reference_en.txt"))
        .expect("reference text");
    let prompt = engine
        .build_clone_prompt(
            ref_wav.to_str().unwrap(),
            ref_wav.to_str().unwrap(),
            ref_text.trim(),
            Language::English,
        )
        .expect("prompt");
    let text = qwen3_tts_burn::engine::damp_ending(
        "The train to the coast leaves at seven, stops twice, and reaches the harbour a little after nine.",
    );
    let mut cfg = SamplerCfg::app();
    cfg.seed = SEED;
    let mut chunks: Vec<usize> = Vec::new();
    let post = PostProcess {
        damp_ending: false,
        ..PostProcess::app_default()
    };
    let frames = engine
        .synthesize_streaming(&text, &prompt, cfg, 400, true, post, |c| {
            chunks.push(c.len())
        })
        .expect("stream");
    let total: usize = chunks.iter().sum();
    // The lead buffer used to swallow the whole utterance before the first
    // chunk (a 3 s line waited 6.5 s); with the rate measured from the first
    // frame, the first chunk must be a fraction of the total.
    assert!(
        chunks.len() >= 2,
        "one chunk only ({} frames): the lead buffer ate the utterance",
        frames.len()
    );
    assert!(
        chunks[0] * 2 < total,
        "first chunk is {} of {} samples; streaming is not streaming",
        chunks[0],
        total
    );
}
