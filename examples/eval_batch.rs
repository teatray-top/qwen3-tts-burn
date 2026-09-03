//! Synthesize a list of lines in one process and record per-line timings.
//!
//! ```text
//! cargo run --release --example eval_batch -- \
//!     <model-dir> <reference.wav> "<transcript>" <lang> <lines.txt> <out-dir>
//! ```
//!
//! One line of text per input line. Each becomes `<out-dir>/<n>.wav` with the
//! app's post-processing, and a `timings.tsv` records generation seconds,
//! audio seconds and frames for each. The first line is generated twice and
//! the first pass is discarded, so the numbers are warm. A seventh argument
//! `keep-tail` leaves the trailing silence untrimmed, for clips that will be
//! joined with a pause anyway.

use std::fmt::Write as _;
use std::time::Instant;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::sampling::SamplerCfg;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        eprintln!("usage: eval_batch <model-dir> <reference.wav> <transcript> <lang> <lines.txt> <out-dir>");
        std::process::exit(2);
    }
    let lang = Language::from_code(&args[3]).unwrap_or_else(|| {
        eprintln!("languages: {}", Language::codes().join(", "));
        eprintln!("unknown language {:?}", args[3]);
        std::process::exit(2);
    });
    let lines: Vec<String> = std::fs::read_to_string(&args[4])
        .expect("read lines")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    std::fs::create_dir_all(&args[5]).expect("out dir");
    let keep_tail = args.get(6).map(|a| a == "keep-tail").unwrap_or(false);

    let t = Instant::now();
    let engine = qwen3_tts_burn::load_vulkan(&args[0]).expect("load");
    let load_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    engine.warmup().expect("warmup");
    let warm_s = t.elapsed().as_secs_f64();
    let prompt = engine
        .build_clone_prompt(&args[1], &args[1], &args[2], lang)
        .expect("prompt");
    println!(
        "load {load_s:.1}s warmup {warm_s:.1}s ref_frames {}",
        prompt.ref_codes.len()
    );

    let mut cfg = SamplerCfg::app();
    cfg.seed = 77;
    let post = PostProcess::app_default();
    let _ = engine.synthesize_with(&lines[0], &prompt, cfg, 600, post);

    let mut tsv = String::from("n\tgen_s\taudio_s\tframes\ttext\n");
    for (i, line) in lines.iter().enumerate() {
        let t = Instant::now();
        let frames = engine
            .synthesize_frames(line, &prompt, cfg, 600, post)
            .expect("frames");
        let raw = engine
            .decode_after_reference(&prompt, &frames)
            .expect("decode");
        let gen_s = t.elapsed().as_secs_f64();
        let wav = {
            let sr = 24000u32;
            let start = qwen3_tts_burn::postproc::leading_trim(&raw, sr).min(raw.len());
            let end = if keep_tail {
                raw.len()
            } else {
                qwen3_tts_burn::postproc::trailing_trim(&raw, sr).max(start)
            };
            let mut w = raw[start..end].to_vec();
            qwen3_tts_burn::lowpass::ButterworthLp::new(10500.0, sr as f64, 6)
                .process_buffer(&mut w);
            qwen3_tts_burn::deesser::Deesser::new(sr as f64, 12.0).process_buffer(&mut w);
            w
        };
        let audio_s = wav.len() as f64 / 24_000.0;
        let path = format!("{}/{:02}.wav", args[5], i + 1);
        qwen3_tts_burn::audio::write_wav_24k(&path, &wav).expect("write");
        println!(
            "{:02} gen {gen_s:.2}s audio {audio_s:.2}s ({} frames) {:.2}x",
            i + 1,
            frames.len(),
            audio_s / gen_s
        );
        let _ = writeln!(
            tsv,
            "{}\t{gen_s:.3}\t{audio_s:.3}\t{}\t{line}",
            i + 1,
            frames.len()
        );
    }
    std::fs::write(format!("{}/timings.tsv", args[5]), tsv).expect("tsv");
}
