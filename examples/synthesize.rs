//! Clone a voice from a reference clip and speak a line in it.
//!
//! ```text
//! cargo run --release --example synthesize -- \
//!     <model-dir> <reference.wav> "<reference transcript>" "<text>" [lang] [out.wav]
//! ```
//!
//! `lang` is the codec language token; `auto` leaves the choice to the model,
//! which is what the official implementation does by default. It has to
//! match the text being spoken. The reference transcript has to be what the
//! reference clip actually says, since the model is shown that pair as an
//! example before it is asked for new speech.
//!
//! By default the model runs as published: `SamplerCfg::default()` (the
//! official generation config) and `PostProcess::none()`. `--post` switches on
//! the relay app's heuristics and filters, `--profile app|greedy` picks the
//! app sampler or a fully greedy one, `--xvector` skips the in-context
//! example, and `--temp`/`--seed`/`--rep`/`--ras`/`--max-frames` override
//! single knobs.

use std::time::Instant;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;

fn opt<T: std::str::FromStr>(args: &mut Vec<String>, name: &str) -> Option<T> {
    let i = args.iter().position(|a| a == name)?;
    let v = args.get(i + 1)?.parse().ok();
    args.drain(i..i + 2);
    v
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let post = match args.iter().position(|a| a == "--post") {
        Some(i) => {
            args.remove(i);
            PostProcess::app_default()
        }
        None => PostProcess::none(),
    };
    let profile: Option<String> = opt(&mut args, "--profile");
    let xvector = match args.iter().position(|a| a == "--xvector") {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };
    let rep: Option<f64> = opt(&mut args, "--rep");
    let ras: Option<i64> = opt(&mut args, "--ras");
    let temp: Option<f64> = opt(&mut args, "--temp");
    let seed: Option<u64> = opt(&mut args, "--seed");
    let max_frames: usize = opt(&mut args, "--max-frames").unwrap_or(400);
    if args.len() < 4 {
        eprintln!(
            "usage: synthesize <model-dir> <reference.wav> <reference transcript> \
             <text> [lang] [out.wav]"
        );
        std::process::exit(2);
    }
    let (model_dir, reference, ref_text, text) = (&args[0], &args[1], &args[2], &args[3]);
    let lang = args
        .get(4)
        .map(|s| {
            Language::from_code(s).unwrap_or_else(|| {
                eprintln!(
                    "unknown language {s:?}; use one of {}",
                    Language::codes().join(", ")
                );
                std::process::exit(2);
            })
        })
        .unwrap_or_default();
    let out = args.get(5).map(String::as_str).unwrap_or("out.wav");

    let t = Instant::now();
    let engine = qwen3_tts_burn::load_vulkan(model_dir).expect("load model");
    println!("load {:.1}s", t.elapsed().as_secs_f64());

    // Compiles the GPU kernels the model will use. Skipping it is allowed; the
    // same cost then lands on the first sentence.
    let t = Instant::now();
    engine.warmup().expect("warmup");
    println!("warmup {:.1}s", t.elapsed().as_secs_f64());

    // The same clip serves as speaker vector and in-context example here; they
    // can be different files.
    let t = Instant::now();
    let prompt = if xvector {
        engine
            .build_xvector_prompt(reference, lang)
            .expect("build prompt")
    } else {
        engine
            .build_clone_prompt(reference, reference, ref_text, lang)
            .expect("build prompt")
    };
    println!(
        "prompt {:.1}s ({} reference frames, lang {lang:?})",
        t.elapsed().as_secs_f64(),
        prompt.ref_codes.len()
    );

    let t = Instant::now();
    let mut cfg = match profile.as_deref() {
        None | Some("official") => qwen3_tts_burn::sampling::SamplerCfg::default(),
        Some("app") => qwen3_tts_burn::sampling::SamplerCfg::app(),
        Some("greedy") => qwen3_tts_burn::sampling::SamplerCfg::greedy(),
        Some(other) => {
            eprintln!("unknown profile {other:?}; use official, app or greedy");
            std::process::exit(2);
        }
    };
    if let Some(v) = temp {
        cfg.temperature = v;
    }
    if let Some(v) = seed {
        cfg.seed = v;
    }
    if let Some(v) = rep {
        cfg.repetition_penalty = v as f32;
    }
    if let Some(v) = ras {
        cfg.ras_window = if v <= 0 { None } else { Some(v as usize) };
    }
    let audio = engine
        .synthesize_with(text, &prompt, cfg, max_frames, post)
        .expect("synthesize");
    let wall = t.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / 24_000.0;
    println!(
        "synthesize {secs:.2}s of audio in {wall:.1}s ({:.2}x realtime)          [{} temp {:.2} top_p {:.2} rep {:.2} ras {} cp_sample {} seed {} max_frames {max_frames}]",
        secs / wall,
        profile.as_deref().unwrap_or("official"),
        cfg.temperature,
        cfg.top_p,
        cfg.repetition_penalty,
        cfg.ras_window.is_some(),
        cfg.cp_sample,
        cfg.seed
    );

    qwen3_tts_burn::audio::write_wav_24k(out, &audio).expect("write wav");
    println!("wrote {out}");
}
