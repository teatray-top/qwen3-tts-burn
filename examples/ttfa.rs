//! Time to first audio on the streaming path.
//!
//! ```text
//! cargo run --release --example ttfa -- <model-dir> <reference.wav> "<transcript>" "<text>" [lang]
//! ```
//!
//! Prints, for the streaming call the relay app uses: milliseconds until the
//! first chunk is handed back, total wall time, and audio seconds produced.
//! `full_lead` is the app's setting (buffer enough that realtime playback
//! cannot overtake generation); the second run measures with the lead off,
//! which is the earliest the pipeline can hand anything back.

use std::time::Instant;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::sampling::SamplerCfg;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!("usage: ttfa <model-dir> <reference.wav> <transcript> <text> [lang]");
        std::process::exit(2);
    }
    let lang = args
        .get(4)
        .and_then(|s| Language::from_code(s))
        .unwrap_or_default();
    let engine = qwen3_tts_burn::load_vulkan(&args[0]).expect("load");
    engine.warmup().expect("warmup");
    let prompt = engine
        .build_clone_prompt(&args[1], &args[1], &args[2], lang)
        .expect("prompt");
    let text = qwen3_tts_burn::engine::damp_ending(&args[3]);
    let post = PostProcess {
        damp_ending: false,
        ..PostProcess::app_default()
    };

    for (label, full_lead) in [("app lead", true), ("no lead", false)] {
        let mut cfg = SamplerCfg::app();
        cfg.seed = 77;
        let t0 = Instant::now();
        let mut first_ms: Option<f64> = None;
        let mut samples = 0usize;
        let frames = engine
            .synthesize_streaming(&text, &prompt, cfg, 400, full_lead, post, |chunk| {
                if first_ms.is_none() && !chunk.is_empty() {
                    first_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
                samples += chunk.len();
            })
            .expect("stream");
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "{label}: first audio {:.0} ms | total {wall:.2} s | audio {:.2} s ({} frames) | {:.2}x realtime",
            first_ms.unwrap_or(f64::NAN),
            samples as f64 / 24_000.0,
            frames.len(),
            samples as f64 / 24_000.0 / wall
        );
    }
}
