//! `deesser::Deesser` only acts when the >4.5 kHz band spikes relative to its
//! own running average. A steady 1 kHz tone has no such spike, so the tone must
//! come through at its input level: within 1 dB over the whole buffer, and
//! unchanged once the running average has settled.

use qwen3_tts_burn::deesser::Deesser;

const SR: f64 = 24_000.0;

fn tone(hz: f64, secs: f64, amp: f32) -> Vec<f32> {
    let n = (SR * secs) as usize;
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / SR).sin() as f32)
        .collect()
}

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

fn db(a: &[f32], b: &[f32]) -> f64 {
    20.0 * (rms(a) / rms(b)).log10()
}

#[test]
fn leaves_1khz_tone_within_1db() {
    let x = tone(1_000.0, 2.0, 0.5);
    let mut y = x.clone();
    Deesser::new(SR, 12.0).process_buffer(&mut y);
    let d = db(&y, &x);
    assert!(d.abs() < 1.0, "whole-buffer gain {d:.3} dB");
}

#[test]
fn settled_1khz_tone_is_unchanged() {
    let x = tone(1_000.0, 2.0, 0.5);
    let mut y = x.clone();
    Deesser::new(SR, 12.0).process_buffer(&mut y);
    let half = x.len() / 2;
    let d = db(&y[half..], &x[half..]);
    assert!(d.abs() < 0.05, "settled gain {d:.4} dB");
    let worst = x[half..]
        .iter()
        .zip(&y[half..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "settled max sample deviation {worst:e}");
}

#[test]
fn ceiling_bounds_the_startup_dip() {
    let x = tone(1_000.0, 0.5, 0.5);
    let mut y = x.clone();
    Deesser::new(SR, 12.0).process_buffer(&mut y);
    let first = (SR * 0.1) as usize;
    let d = db(&y[..first], &x[..first]);
    assert!(d < 0.5 && d > -12.0, "first 100 ms gain {d:.3} dB");
}

#[test]
fn silence_stays_silent() {
    let mut z = vec![0.0f32; 24_000];
    Deesser::new(SR, 12.0).process_buffer(&mut z);
    assert!(z.iter().all(|&s| s == 0.0));
}
