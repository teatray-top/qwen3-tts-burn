//! Parity with the official implementation, measured two ways.
//!
//! ```text
//! cargo run --release --example parity -- <model-dir>
//! ```
//!
//! Greedy frames: the same reference clip, sentence and greedy rule as the
//! official `qwen_tts` package run with `do_sample=False,
//! subtalker_dosample=False` (no random draw anywhere, repetition penalty 1.05
//! before the argmax, stop at the first end-of-speech token, no text rewrite,
//! no attention bias). The official frames (bf16 and f32), reference codes and
//! speaker embedding are checked in under `eval/parity/`; the engine runs on
//! its own reference codes and speaker embedding and on the official ones, and
//! writes both frame sequences next to them.
//!
//! Logits: greedy paths part at the first near-tie and never meet again, so
//! the frame comparison says little about numerics. The second check feeds the
//! official frame sequence to the engine (teacher forcing, official inputs)
//! and compares the talker's logits at every step with the logits the official
//! model produced at that same step — the same history on both sides. The
//! example exits non-zero unless the engine picks the official top token at
//! every step of both paths with a cosine of at least 0.999, and its first-step
//! logits sit no further from the official f32 logits than the official bf16
//! logits do. The code predictor's own logits are not compared; that is the
//! one part of a frame this check does not cover.

use std::path::{Path, PathBuf};

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::pipeline::ClonePrompt;
use qwen3_tts_burn::sampling::SamplerCfg;

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog.";

fn load_frames(path: &Path) -> Vec<Vec<u32>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse().expect("integer frame entry"))
                .collect()
        })
        .collect()
}

/// A little-endian float32 array in NumPy's `.npy` format, version 1 or 2,
/// C order, one- or two-dimensional. Returns the shape and the flat data.
fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        bytes.len() > 10 && &bytes[..6] == b"\x93NUMPY",
        "{}: not .npy",
        path.display()
    );
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    let header = String::from_utf8_lossy(&bytes[start..start + header_len]).into_owned();
    assert!(
        header.contains("'<f4'") && header.contains("'fortran_order': False"),
        "{}: expected little-endian float32 C order, got {header}",
        path.display()
    );
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .unwrap_or_else(|| panic!("{}: no shape in header", path.display()));
    let shape: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("shape entry"))
        .collect();
    let data: Vec<f32> = bytes[start + header_len..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{}: data does not match shape {shape:?}",
        path.display()
    );
    (shape, data)
}

fn load_logits(path: &Path) -> Vec<Vec<f32>> {
    let (shape, data) = load_npy_f32(path);
    assert_eq!(shape.len(), 2, "{}: expected steps x vocab", path.display());
    data.chunks_exact(shape[1]).map(|r| r.to_vec()).collect()
}

fn write_frames(path: &Path, frames: &[Vec<u32>]) {
    let mut text = String::new();
    for f in frames {
        let line: Vec<String> = f.iter().map(u32::to_string).collect();
        text.push_str(&line.join(" "));
        text.push('\n');
    }
    std::fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// Prints one frame comparison line and returns the semantic-column agreement
/// in percent.
fn report_frames(name: &str, a: &[Vec<u32>], b: &[Vec<u32>]) -> f64 {
    let n = a.len().min(b.len());
    let mut first = n;
    let mut semantic = 0usize;
    let mut whole = 0usize;
    for i in 0..n {
        let differs = a[i] != b[i];
        if differs && first == n {
            first = i;
        }
        if a[i].first() == b[i].first() {
            semantic += 1;
        }
        if !differs {
            whole += 1;
        }
    }
    let pct = |k: usize| {
        if n == 0 {
            0.0
        } else {
            100.0 * k as f64 / n as f64
        }
    };
    println!(
        "{name:46} {:3} vs {:3} frames | first difference at frame {first:3} | \
         semantic column agrees {:5.1}% | whole frames agree {:5.1}%",
        a.len(),
        b.len(),
        pct(semantic),
        pct(whole)
    );
    pct(semantic)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-12)
}

/// 1-based rank of `index` in `v` (1 = the largest entry).
fn rank_of(v: &[f32], index: usize) -> usize {
    1 + v.iter().filter(|&&x| x > v[index]).count()
}

struct PathStats {
    steps: usize,
    top1_agree: usize,
    cosine_mean: f64,
    cosine_min: f64,
    rank_mean: f64,
}

/// Compare the engine's logits with the official ones step by step along one
/// official path; both sides saw the same frames before each step.
fn compare_logits(name: &str, engine: &[Vec<f32>], official: &[Vec<f32>]) -> PathStats {
    let steps = engine.len().min(official.len());
    let mut top1_agree = 0usize;
    let mut cos_sum = 0.0f64;
    let mut cos_min = 1.0f64;
    let mut rank_sum = 0usize;
    let mut rank_max = 0usize;
    for i in 0..steps {
        assert_eq!(
            engine[i].len(),
            official[i].len(),
            "vocab size differs: engine {} vs official {}",
            engine[i].len(),
            official[i].len()
        );
        let official_top = argmax(&official[i]);
        if argmax(&engine[i]) == official_top {
            top1_agree += 1;
        }
        let c = cosine(&engine[i], &official[i]);
        cos_sum += c;
        cos_min = cos_min.min(c);
        let r = rank_of(&engine[i], official_top);
        rank_sum += r;
        rank_max = rank_max.max(r);
    }
    let stats = PathStats {
        steps,
        top1_agree,
        cosine_mean: cos_sum / steps.max(1) as f64,
        cosine_min: cos_min,
        rank_mean: rank_sum as f64 / steps.max(1) as f64,
    };
    println!(
        "{name:46} {steps:3} steps | top-1 agrees at {top1_agree:3} | cosine mean {:.4} min {:.4} | official top-1 ranks {:.2} in the engine's logits on average, {rank_max} at worst",
        stats.cosine_mean, stats.cosine_min, stats.rank_mean
    );
    stats
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: parity <model-dir>");
        std::process::exit(2);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let parity = root.join("eval").join("parity");
    let ref_wav = root.join("samples").join("reference_en.wav");
    let ref_text = std::fs::read_to_string(root.join("samples").join("reference_en.txt"))
        .expect("reference text");

    let engine = qwen3_tts_burn::load_vulkan(&args[0]).expect("load");
    engine.warmup().expect("warmup");
    let own = engine
        .build_clone_prompt(
            ref_wav.to_str().unwrap(),
            ref_wav.to_str().unwrap(),
            ref_text.trim(),
            Language::English,
        )
        .expect("prompt");
    let (spk_shape, spk) = load_npy_f32(&parity.join("official_spk.npy"));
    assert_eq!(spk_shape, vec![2048], "official speaker vector shape");
    let official_inputs = ClonePrompt {
        language: Language::English,
        speaker_embedding: spk,
        ref_codes: load_frames(&parity.join("official_ref_codes.txt")),
        ref_text_ids: own.ref_text_ids.clone(),
    };
    println!(
        "reference: {} frames from this engine's encoder, {} from the official one",
        own.ref_codes.len(),
        official_inputs.ref_codes.len()
    );

    println!("\n== greedy frames");
    let run = |prompt: &ClonePrompt| {
        engine
            .synthesize_frames(
                SENTENCE,
                prompt,
                SamplerCfg::greedy(),
                400,
                PostProcess::none(),
            )
            .expect("frames")
    };
    let from_own = run(&own);
    let from_official = run(&official_inputs);
    write_frames(&parity.join("engine_frames_own_inputs.txt"), &from_own);
    write_frames(
        &parity.join("engine_frames_official_inputs.txt"),
        &from_official,
    );
    let bf16 = load_frames(&parity.join("official_frames_bf16.txt"));
    let f32_frames = load_frames(&parity.join("official_frames_f32.txt"));
    report_frames("official bf16 vs official f32", &bf16, &f32_frames);
    report_frames("engine, own inputs vs official bf16", &from_own, &bf16);
    report_frames("engine, own inputs vs official f32", &from_own, &f32_frames);
    report_frames(
        "engine, official inputs vs official bf16",
        &from_official,
        &bf16,
    );
    report_frames(
        "engine, official inputs vs official f32",
        &from_official,
        &f32_frames,
    );

    println!("\n== logits on the same history (official inputs, official frames fed back)");
    let official_logits_f32 = load_logits(&parity.join("official_logits_f32.npy"));
    let official_logits_bf16 = load_logits(&parity.join("official_logits_bf16.npy"));
    let engine_along_f32 = engine
        .forced_logits(SENTENCE, &official_inputs, &f32_frames)
        .expect("forced logits");
    let engine_along_bf16 = engine
        .forced_logits(SENTENCE, &official_inputs, &bf16)
        .expect("forced logits");
    let reference_gap = compare_logits(
        "official bf16 vs official f32, step 0 only",
        &official_logits_bf16[..1],
        &official_logits_f32[..1],
    );
    let along_f32 = compare_logits(
        "engine f16 vs official f32, along the f32 path",
        &engine_along_f32,
        &official_logits_f32,
    );
    let along_bf16 = compare_logits(
        "engine f16 vs official bf16, along the bf16 path",
        &engine_along_bf16,
        &official_logits_bf16,
    );
    let engine_own_step0 = engine
        .forced_logits(SENTENCE, &own, &[])
        .expect("forced logits");
    compare_logits(
        "engine f16 on its own inputs vs official f32, step 0",
        &engine_own_step0[..1],
        &official_logits_f32[..1],
    );

    // The gate. Every step of both official paths must pick the same top token
    // as the official model, at a cosine no worse than MIN_COSINE — measured at
    // 0.9999 on both paths, so the bar catches a real numerical regression
    // without tripping on the last digit.
    const MIN_COSINE: f64 = 0.999;
    let mut failures: Vec<String> = Vec::new();
    for (path, stats) in [("f32", &along_f32), ("bf16", &along_bf16)] {
        if stats.top1_agree < stats.steps {
            failures.push(format!(
                "along the {path} path the engine's top-1 differs at {} of {} steps",
                stats.steps - stats.top1_agree,
                stats.steps
            ));
        }
        if stats.cosine_min < MIN_COSINE {
            failures.push(format!(
                "along the {path} path the worst-step cosine is {:.4}, below {MIN_COSINE}",
                stats.cosine_min
            ));
        }
    }
    let engine_step0 = cosine(&engine_along_f32[0], &official_logits_f32[0]);
    if engine_step0 < reference_gap.cosine_mean {
        failures.push(format!(
            "at step 0 the engine's logits sit further from the official f32 logits (cosine {engine_step0:.4}) than the official bf16 logits do ({:.4})",
            reference_gap.cosine_mean
        ));
    }
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("parity: {f}");
        }
        std::process::exit(1);
    }
    println!(
        "\nstep 0: engine f16 to official f32 cosine {engine_step0:.4}, official bf16 to f32 {:.4}; along the f32 path the engine's top-1 matched at {}/{} steps",
        reference_gap.cosine_mean, along_f32.top1_agree, along_f32.steps
    );
}
