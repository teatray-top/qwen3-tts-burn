//! Where the VRAM goes.
//!
//! Weights are 3.86 GB of bf16 plus a 651 MB codec, but the process peaks
//! around 5.5 GB. This holds after each stage long enough for an external
//! sampler to read a steady value, so the difference can be attributed rather
//! than guessed.
//!
//! ```text
//! cargo run --release --example mem_probe -- <model-dir> <reference.wav> <transcript>
//! ```

use std::thread::sleep;
use std::time::Duration;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;

const HOLD: Duration = Duration::from_secs(5);

fn stage(name: &str) {
    // stderr is unbuffered, so the marker lands when the stage does.
    eprintln!("STAGE {name}");
    sleep(HOLD);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: mem_probe <model-dir> <reference.wav> <transcript>");
        std::process::exit(2);
    }
    let (model_dir, reference, ref_text) = (&args[0], &args[1], &args[2]);

    stage("baseline");
    let engine = qwen3_tts_burn::load_vulkan(model_dir).expect("load model");
    stage("loaded");

    engine.warmup().expect("warmup");
    stage("warmed");

    let prompt = engine
        .build_clone_prompt(reference, reference, ref_text, Language::Korean)
        .expect("build prompt");
    stage("prompt");

    let cfg = qwen3_tts_burn::sampling::SamplerCfg::app();
    let audio = engine
        .synthesize_with(
            "로그인 후 마이결제 화면에서 보실 수 있습니다.",
            &prompt,
            cfg,
            400,
            PostProcess::app_default(),
        )
        .expect("synthesize");
    println!("({:.2}s of audio)", audio.len() as f64 / 24_000.0);
    stage("generated");
}
