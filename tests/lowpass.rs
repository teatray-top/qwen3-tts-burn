//! `lowpass::ButterworthLp` at the relay app's setting (10.5 kHz, order 6),
//! run at 48 kHz so a 20 kHz tone sits below Nyquist. Levels are measured on
//! the second half of a one-second tone, after the filter has settled.

use qwen3_tts_burn::lowpass::ButterworthLp;

const SR: f64 = 48_000.0;

fn tone(hz: f64, secs: f64, amp: f32) -> Vec<f32> {
    let n = (SR * secs) as usize;
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / SR).sin() as f32)
        .collect()
}

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

fn gain_db(hz: f64, order: usize) -> f64 {
    let x = tone(hz, 1.0, 0.5);
    let mut y = x.clone();
    ButterworthLp::new(10_500.0, SR, order).process_buffer(&mut y);
    let half = x.len() / 2;
    20.0 * (rms(&y[half..]) / rms(&x[half..])).log10()
}

#[test]
fn passes_1khz_within_a_tenth_of_a_db() {
    let db = gain_db(1_000.0, 6);
    assert!(db.abs() < 0.1, "1 kHz gain {db:.3} dB");
}

#[test]
fn attenuates_20khz_by_more_than_40db() {
    let db = gain_db(20_000.0, 6);
    assert!(db < -40.0, "20 kHz gain {db:.2} dB");
}

#[test]
fn higher_order_attenuates_more() {
    let o2 = gain_db(20_000.0, 2);
    let o6 = gain_db(20_000.0, 6);
    assert!(o2 < -6.0, "order 2 gain {o2:.2} dB");
    assert!(o6 < o2 - 10.0, "order 6 {o6:.2} dB vs order 2 {o2:.2} dB");
}

#[test]
fn process1_matches_process_buffer() {
    let x = tone(3_000.0, 0.05, 0.4);
    let mut a = x.clone();
    ButterworthLp::new(10_500.0, SR, 6).process_buffer(&mut a);
    let mut f = ButterworthLp::new(10_500.0, SR, 6);
    let b: Vec<f32> = x.iter().map(|&s| f.process1(s)).collect();
    assert_eq!(a, b);
}

#[test]
fn silence_stays_silent() {
    let mut z = vec![0.0f32; 4_800];
    ButterworthLp::new(10_500.0, SR, 6).process_buffer(&mut z);
    assert!(z.iter().all(|&s| s == 0.0));
}
