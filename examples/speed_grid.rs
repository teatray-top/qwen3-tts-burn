//! Generation time as a function of reference length and output length.
//!
//! ```text
//! cargo run --release --example speed_grid -- <model-dir> <reference.wav> "<transcript>" <lang> <out.tsv>
//! ```
//!
//! The reference clip is repeated and cut to several durations, and lines of
//! several lengths are synthesized against each (app profile, seed 77, one
//! warm-up per prompt). Each row records the reference frames, the generated
//! frames, the decoder's padded frame count and the wall time of generation
//! and of decoding separately, so the cost can be fitted rather than quoted
//! as one number.

use std::fmt::Write as _;
use std::time::Instant;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;
use qwen3_tts_burn::sampling::SamplerCfg;

const REF_SECONDS: [f64; 5] = [2.0, 4.0, 7.0, 10.0, 15.0];
const LINES_EN: [&str; 4] = [
    "Good morning.",
    "The train leaves at half past seven every morning except Sunday.",
    "The train to the coast leaves at seven, stops twice, and reaches the harbour a little after nine. If the morning is clear, you can see the islands from the last carriage.",
    "The train to the coast leaves at seven, stops twice, and reaches the harbour a little after nine. If the morning is clear, you can see the islands from the last carriage, and on a very clear day the lighthouse on the far point as well. We usually take the early service, buy coffee at the kiosk, and sit on the left so the sea stays in view for the whole ride.",
];
const LINES_KO: [&str; 4] = [
    "안녕하세요.",
    "기차는 일요일을 빼고 매일 아침 일곱 시 반에 출발합니다.",
    "해안으로 가는 기차는 일곱 시에 출발해 두 번 서고 아홉 시 조금 지나 항구에 도착합니다. 아침이 맑으면 마지막 칸에서 섬들이 보입니다.",
    "해안으로 가는 기차는 일곱 시에 출발해 두 번 서고 아홉 시 조금 지나 항구에 도착합니다. 아침이 맑으면 마지막 칸에서 섬들이 보이고, 아주 맑은 날에는 먼 곶의 등대까지 보입니다. 우리는 보통 이른 차를 타고 매점에서 커피를 사서 바다가 계속 보이도록 왼쪽에 앉습니다.",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        eprintln!("usage: speed_grid <model-dir> <reference.wav> <transcript> <lang> <out.tsv>");
        std::process::exit(2);
    }
    let lang = Language::from_code(&args[3]).expect("language");
    let lines: &[&str] = if lang == Language::Korean {
        &LINES_KO
    } else {
        &LINES_EN
    };
    let engine = qwen3_tts_burn::load_vulkan(&args[0]).expect("load");
    engine.warmup().expect("warmup");

    let (raw, sr) = qwen3_tts_burn::audio::load_wav(&args[1]).expect("reference");
    let tmp = std::env::temp_dir().join("qtb_speed_grid_ref.wav");
    let mut tsv = String::from(
        "ref_s	ref_frames	line	chars	max_frames	frames	padded	gen_s	decode_s	audio_s
",
    );
    let mut cfg = SamplerCfg::app();
    cfg.seed = 77;
    let post = PostProcess::app_default();

    for &secs in &REF_SECONDS {
        let want = (secs * sr as f64) as usize;
        let mut clip: Vec<f32> = Vec::with_capacity(want);
        while clip.len() < want {
            let take = (want - clip.len()).min(raw.len());
            clip.extend_from_slice(&raw[..take]);
        }
        // The transcript is only the text the model is shown; for timing the
        // pairing with a repeated clip does not matter.
        let transcript =
            std::iter::repeat_n(args[2].as_str(), ((secs / 7.6).ceil() as usize).max(1))
                .collect::<Vec<_>>()
                .join(" ");
        let rs = qwen3_tts_burn::resample::resample(&clip, sr, 24000).expect("resample");
        qwen3_tts_burn::audio::write_wav_24k(tmp.to_str().unwrap(), &rs).expect("tmp wav");
        let prompt = engine
            .build_clone_prompt(
                tmp.to_str().unwrap(),
                tmp.to_str().unwrap(),
                &transcript,
                lang,
            )
            .expect("prompt");
        let _ = engine.synthesize_with(lines[0], &prompt, cfg, 600, post);

        // Each line once in the 1024-position bucket; line 2 also in the 448
        // and 2048 buckets so the per-frame cost can be fitted per bucket.
        let mut jobs: Vec<(usize, usize)> = (0..lines.len()).map(|i| (i, 600)).collect();
        jobs.push((1, 100));
        jobs.push((1, 1500));
        for (i, max_frames) in jobs {
            let line = lines[i];
            let t = Instant::now();
            let frames = engine
                .synthesize_frames(line, &prompt, cfg, max_frames, post)
                .expect("frames");
            let gen_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let raw_audio = engine
                .decode_after_reference(&prompt, &frames)
                .expect("decode");
            let decode_s = t.elapsed().as_secs_f64();
            let total = prompt.ref_codes.len() + frames.len();
            let padded = total.next_power_of_two().max(32);
            let _ = writeln!(
                tsv,
                "{secs}\t{}\t{}\t{}\t{max_frames}\t{}\t{padded}\t{gen_s:.3}\t{decode_s:.3}\t{:.3}",
                prompt.ref_codes.len(),
                i + 1,
                line.chars().count(),
                frames.len(),
                raw_audio.len() as f64 / 24000.0
            );
            println!(
                "ref {secs:>4}s ({:>3} fr) line {} max {max_frames:>4} {:>3} fr pad {padded:>4}: gen {gen_s:5.2}s decode {decode_s:5.2}s",
                prompt.ref_codes.len(),
                i + 1,
                frames.len()
            );
        }
    }
    std::fs::write(&args[4], tsv).expect("write tsv");
    let _ = std::fs::remove_file(&tmp);
}
