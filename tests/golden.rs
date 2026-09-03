//! Golden regression tests for the talker + code predictor, on the model.
//!
//! Ignored by default because they need the weights and a Vulkan GPU:
//!
//! ```text
//! QTB_MODEL_DIR=/path/to/model cargo test --release --test golden -- --ignored --test-threads=1
//! ```
//!
//! One thread: each test loads the model, and four engines on one GPU at once
//! is not the condition anything here is measured under.
//!
//! ```text
//! (the same command)
//! ```
//!
//! The golden is the frame sequence, not audio. Frames are the integer codec
//! codes (16 per frame: the talker's token plus 15 code-predictor codes) that
//! the sampler draws for a fixed seed, so on one machine they are reproducible
//! bit for bit. Decoded audio would not be: it is f16 GPU floating point and it
//! changes with every post-processing knob that has nothing to do with the
//! model.
//!
//! Two profiles are pinned: the official sampling config with no heuristics,
//! and the relay app's profile with all of them. Whether the frames are
//! identical across GPUs or drivers is not guaranteed — f16 reduction order can
//! flip a near-tie in the logits — so treat a golden file as a per-machine
//! guard: on new hardware delete it and let the test rewrite it before
//! suspecting a regression.

use std::path::PathBuf;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::sampling::SamplerCfg;
use qwen3_tts_burn::TtsError;

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog.";
const SEED: u64 = 77;
const MAX_FRAMES: usize = 400;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn render(frames: &[Vec<u32>]) -> String {
    let mut s = String::new();
    for f in frames {
        let line: Vec<String> = f.iter().map(u32::to_string).collect();
        s.push_str(&line.join(" "));
        s.push('\n');
    }
    s
}

fn parse(text: &str) -> Vec<Vec<u32>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<u32>().expect("golden file: not an integer"))
                .collect()
        })
        .collect()
}

fn model_dir() -> String {
    // Loud on purpose: this only runs when asked for (`--ignored`), and a
    // silent return would report "ok" for a check that never happened.
    std::env::var("QTB_MODEL_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
        .expect("golden: set QTB_MODEL_DIR to the 1.7B Base model directory")
}

fn check_golden(name: &str, cfg: SamplerCfg, post: PostProcess) {
    let engine = qwen3_tts_burn::load_vulkan(&model_dir()).expect("load model");
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
    let mut cfg = cfg;
    cfg.seed = SEED;
    let frames = engine
        .synthesize_frames(SENTENCE, &prompt, cfg, MAX_FRAMES, post)
        .expect("frames");
    assert!(!frames.is_empty(), "no frames generated");
    assert!(
        frames.iter().all(|f| f.len() == 16),
        "every frame has 16 codes"
    );

    let path = root()
        .join("tests")
        .join("golden")
        .join(format!("{name}.frames"));
    let rendered = render(&frames);
    if !path.exists() {
        std::fs::write(&path, &rendered).expect("write golden");
        panic!("golden written, rerun: {}", path.display());
    }
    let expected = parse(&std::fs::read_to_string(&path).expect("read golden"));
    if expected != frames {
        let first = expected
            .iter()
            .zip(&frames)
            .position(|(a, b)| a != b)
            .unwrap_or(expected.len().min(frames.len()));
        panic!(
            "{name}: frames differ from {} (expected {} frames, got {}, first difference at frame {first})",
            path.display(),
            expected.len(),
            frames.len()
        );
    }
}

#[test]
#[ignore = "needs QTB_MODEL_DIR and a Vulkan GPU"]
fn en_seed77_official_profile() {
    check_golden(
        "en_seed77_official",
        SamplerCfg::default(),
        PostProcess::none(),
    );
}

#[test]
#[ignore = "needs QTB_MODEL_DIR and a Vulkan GPU"]
fn en_seed77_app_profile() {
    check_golden(
        "en_seed77_app",
        SamplerCfg::app(),
        PostProcess::app_default(),
    );
}

#[test]
#[ignore = "needs QTB_MODEL_DIR and a Vulkan GPU"]
fn bad_input_is_an_error_not_a_panic() {
    let engine = qwen3_tts_burn::load_vulkan(&model_dir()).expect("load model");
    let ref_wav = root().join("samples").join("reference_en.wav");
    let ref_text = std::fs::read_to_string(root().join("samples").join("reference_en.txt"))
        .expect("reference text");
    let prompt = engine
        .build_xvector_prompt(ref_wav.to_str().unwrap(), Language::English)
        .expect("prompt");

    assert!(matches!(
        engine.synthesize("", &prompt, SamplerCfg::default(), 40),
        Err(TtsError::EmptyText)
    ));
    assert!(matches!(
        engine.synthesize("   \n", &prompt, SamplerCfg::default(), 40),
        Err(TtsError::EmptyText)
    ));

    let mut bad = engine
        .build_xvector_prompt(ref_wav.to_str().unwrap(), Language::English)
        .expect("prompt");
    bad.speaker_embedding.truncate(10);
    assert!(matches!(
        engine.synthesize("hello", &bad, SamplerCfg::default(), 40),
        Err(TtsError::InvalidPrompt(_))
    ));

    let mut bad = engine
        .build_clone_prompt(
            ref_wav.to_str().unwrap(),
            ref_wav.to_str().unwrap(),
            ref_text.trim(),
            Language::English,
        )
        .expect("prompt");
    bad.ref_codes[0].pop();
    assert!(matches!(
        engine.synthesize("hello", &bad, SamplerCfg::default(), 40),
        Err(TtsError::InvalidPrompt(_))
    ));

    assert!(matches!(
        engine.decode(&[vec![0u32; 15]]),
        Err(TtsError::InvalidFrames(_))
    ));
    assert!(matches!(
        engine.forced_logits("hello", &prompt, &[vec![0u32; 15]]),
        Err(TtsError::InvalidFrames(_))
    ));
    assert!(matches!(
        engine.forced_logits("hello", &prompt, &[vec![9999u32; 16]]),
        Err(TtsError::InvalidFrames(_))
    ));
    assert!(matches!(
        engine.build_clone_prompt(
            ref_wav.to_str().unwrap(),
            ref_wav.to_str().unwrap(),
            "  ",
            Language::English
        ),
        Err(TtsError::EmptyReferenceText)
    ));
    assert!(matches!(
        engine.build_xvector_prompt("does-not-exist.wav", Language::English),
        Err(TtsError::Io { .. })
    ));
}
