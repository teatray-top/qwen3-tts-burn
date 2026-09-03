//! CPU-only checks of the silence detectors in `postproc` on synthetic
//! signals: exact zeros around a 440 Hz tone at 24 kHz, so every 20 ms window
//! (480 samples) is either fully silent or fully active and the expected
//! boundaries are exact multiples of the window.

use qwen3_tts_burn::postproc::{leading_trim, speech_bounds, trailing_trim, trim_silence};

const SR: u32 = 24_000;
const WIN: usize = 480;

fn tone(secs: f32, amp: f32) -> Vec<f32> {
    let n = (SR as f32 * secs) as usize;
    (0..n)
        .map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
        .collect()
}

fn silence(secs: f32) -> Vec<f32> {
    vec![0.0; (SR as f32 * secs) as usize]
}

fn concat(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// 1 s silence, 1 s tone, 1 s silence: the tone occupies windows 50..=99.
fn sandwich() -> Vec<f32> {
    concat(&[silence(1.0), tone(1.0, 0.5), silence(1.0)])
}

#[test]
fn speech_bounds_keeps_40ms_margin_each_side() {
    let x = sandwich();
    let (a, b) = speech_bounds(&x, SR);
    assert_eq!(a, (50 - 2) * WIN);
    assert_eq!(b, (99 + 1 + 2) * WIN);
}

#[test]
fn trim_silence_matches_speech_bounds() {
    let x = sandwich();
    let (a, b) = speech_bounds(&x, SR);
    let y = trim_silence(&x, SR);
    assert_eq!(y.len(), b - a);
    assert_eq!(y, x[a..b].to_vec());
}

#[test]
fn speech_bounds_on_pure_silence_returns_whole_clip() {
    let x = silence(0.5);
    assert_eq!(speech_bounds(&x, SR), (0, x.len()));
    assert_eq!(trim_silence(&x, SR).len(), x.len());
}

#[test]
fn speech_bounds_on_empty_and_sub_window_input() {
    assert_eq!(speech_bounds(&[], SR), (0, 0));
    let short = vec![0.3; WIN - 1];
    assert_eq!(speech_bounds(&short, SR), (0, short.len()));
}

#[test]
fn speech_bounds_clamps_end_to_length() {
    let x = concat(&[silence(1.0), tone(1.0, 0.5)]);
    let (a, b) = speech_bounds(&x, SR);
    assert_eq!(a, (50 - 2) * WIN);
    assert_eq!(b, x.len());
}

#[test]
fn leading_trim_keeps_60ms_preroll() {
    let x = sandwich();
    assert_eq!(leading_trim(&x, SR), (50 - 3) * WIN);
}

#[test]
fn leading_trim_ignores_a_blip_shorter_than_200ms() {
    let x = concat(&[
        silence(1.0),
        tone(0.1, 0.5),
        silence(1.0),
        tone(1.0, 0.5),
        silence(1.0),
    ]);
    assert_eq!(leading_trim(&x, SR), (105 - 3) * WIN);
}

#[test]
fn leading_trim_is_zero_for_silence_or_immediate_speech() {
    assert_eq!(leading_trim(&silence(1.0), SR), 0);
    assert_eq!(leading_trim(&tone(1.0, 0.5), SR), 0);
    assert_eq!(leading_trim(&[], SR), 0);
}

#[test]
fn trims_ignore_sub_threshold_noise() {
    let x = tone(1.0, 0.004);
    assert_eq!(leading_trim(&x, SR), 0);
    assert_eq!(trailing_trim(&x, SR), x.len());
}

#[test]
fn trailing_trim_keeps_160ms_tail() {
    let x = sandwich();
    assert_eq!(trailing_trim(&x, SR), (99 + 1 + 8) * WIN);
}

#[test]
fn trailing_trim_returns_length_when_nothing_to_cut() {
    let s = silence(1.0);
    assert_eq!(trailing_trim(&s, SR), s.len());
    let t = tone(1.0, 0.5);
    assert_eq!(trailing_trim(&t, SR), t.len());
    assert_eq!(trailing_trim(&[], SR), 0);
}

#[test]
fn leading_and_trailing_trim_compose() {
    let x = sandwich();
    let a = leading_trim(&x, SR);
    let b = trailing_trim(&x, SR);
    assert!(a < b);
    let cut = &x[a..b];
    assert_eq!(cut.len(), (108 - 47) * WIN);
    assert_eq!(trailing_trim(cut, SR), cut.len());
}
