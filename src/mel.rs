use crate::error::TtsError;

const N_FFT: usize = 1024;
pub const HOP: usize = 256;
const N_MELS: usize = 128;
pub const SR: f64 = 24000.0;
pub const FRAME_MS: f64 = HOP as f64 * 1000.0 / SR;
// Reflect padding mirrors pad = (N_FFT - HOP) / 2 samples on each side, so the
// mirrored index j = pad must exist: n >= pad + 1.
pub const MIN_SAMPLES: usize = (N_FFT - HOP) / 2 + 1;

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if f < 1000.0 {
        f / f_sp
    } else {
        let min_log_mel = 15.0;
        let logstep = (6.4f64).ln() / 27.0;
        min_log_mel + (f / 1000.0).ln() / logstep
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if m < 15.0 {
        m * f_sp
    } else {
        let logstep = (6.4f64).ln() / 27.0;
        1000.0 * ((m - 15.0) * logstep).exp()
    }
}

fn filterbank() -> Vec<Vec<f32>> {
    let n_bins = N_FFT / 2 + 1;
    let fmin = 0.0;
    let fmax = SR / 2.0;
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let hz: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let bin_hz: Vec<f64> = (0..n_bins).map(|i| i as f64 * SR / N_FFT as f64).collect();
    (0..N_MELS)
        .map(|m| {
            let (lo, ctr, hi) = (hz[m], hz[m + 1], hz[m + 2]);
            let norm = 2.0 / (hi - lo);
            bin_hz
                .iter()
                .map(|&f| {
                    let up = (f - lo) / (ctr - lo);
                    let down = (hi - f) / (hi - ctr);
                    (up.min(down).max(0.0) * norm) as f32
                })
                .collect()
        })
        .collect()
}

/// Log-mel exactly matching the candle port's `compute_for_speaker_encoder`:
/// periodic Hann, reflect pad (n_fft-hop)/2, magnitude spectrum with 1e-9 in
/// the sqrt, Slaney filterbank with area norm, natural log floored at 1e-5.
/// Returns [n_mels][frames].
pub fn log_mel(samples: &[f32]) -> Result<Vec<Vec<f32>>, TtsError> {
    let pad = (N_FFT - HOP) / 2;
    let n = samples.len();
    if n < MIN_SAMPLES {
        return Err(TtsError::ReferenceTooShort {
            samples_ms: n as f64 * 1000.0 / SR,
            min_ms: MIN_SAMPLES as f64 * 1000.0 / SR,
        });
    }
    let padded: Vec<f64> = (0..n + 2 * pad)
        .map(|i| {
            let idx = i as isize - pad as isize;
            let j = if idx < 0 {
                (-idx) as usize
            } else if idx as usize >= n {
                2 * (n - 1) - idx as usize
            } else {
                idx as usize
            };
            samples[j] as f64
        })
        .collect();

    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()))
        .collect();
    let n_frames = (padded.len() - N_FFT) / HOP + 1;
    let fb = filterbank();
    let n_bins = N_FFT / 2 + 1;

    let mut mags = vec![vec![0f32; n_bins]; n_frames];
    for (fi, mag_row) in mags.iter_mut().enumerate() {
        let start = fi * HOP;
        let mut re: Vec<f64> = (0..N_FFT).map(|i| padded[start + i] * window[i]).collect();
        let mut im = vec![0f64; N_FFT];
        fft(&mut re, &mut im);
        for b in 0..n_bins {
            mag_row[b] = ((re[b] * re[b] + im[b] * im[b] + 1e-9) as f32).sqrt();
        }
    }

    Ok((0..N_MELS)
        .map(|m| {
            (0..n_frames)
                .map(|fi| {
                    let e: f32 = fb[m].iter().zip(&mags[fi]).map(|(w, x)| w * x).sum();
                    e.max(1e-5).ln()
                })
                .collect()
        })
        .collect())
}
