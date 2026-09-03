use std::path::PathBuf;

use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    audioadapter::Adapter, Async, FixedAsync, Resampler as _, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};

use crate::error::TtsError;

/// Sinc resampler with the exact parameters the candle port uses for reference
/// audio (sinc_len 128, cutoff 0.95, linear interp, oversampling 128,
/// BlackmanHarris2, 1024-sample chunks, zero-padded tail).
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, TtsError> {
    for rate in [from_rate, to_rate] {
        if rate == 0 {
            return Err(TtsError::BadSampleRate {
                path: PathBuf::new(),
                rate,
            });
        }
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let chunk = 1024usize;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let numeric = |e: &dyn std::fmt::Display| TtsError::Numeric(format!("resample: {e}"));
    let mut rs = Async::<f32>::new_sinc(ratio, 1.0, &params, chunk, 1, FixedAsync::Input)
        .map_err(|e| numeric(&e))?;

    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + chunk).min(samples.len());
        let mut data = samples[pos..end].to_vec();
        data.resize(chunk, 0.0);
        let input_vecs = vec![data];
        let input = SequentialSliceOfVecs::new(&input_vecs, 1, chunk).map_err(|e| numeric(&e))?;
        let result = rs.process(&input, 0, None).map_err(|e| numeric(&e))?;
        let frames = result.frames();
        for i in 0..frames {
            out.push(result.read_sample(0, i).unwrap_or(0.0));
        }
        pos += chunk;
    }
    Ok(out)
}
