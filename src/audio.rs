use std::path::{Path, PathBuf};

use crate::error::TtsError;

pub const MAX_SAMPLE_RATE: u32 = 384_000;

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn u16_at(d: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([d[p], d[p + 1]])
}

fn u32_at(d: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([d[p], d[p + 1], d[p + 2], d[p + 3]])
}

struct Fmt {
    format: u16,
    channels: usize,
    rate: u32,
    bits: usize,
}

fn parse_fmt(path: &Path, body: &[u8]) -> Result<Fmt, TtsError> {
    let bad = |detail: &str| TtsError::BadWav {
        path: path.to_path_buf(),
        detail: detail.into(),
    };
    if body.len() < 16 {
        return Err(bad("truncated fmt chunk"));
    }
    let mut format = u16_at(body, 0);
    if format == WAVE_FORMAT_EXTENSIBLE {
        if body.len() < 26 {
            return Err(bad("truncated extensible fmt chunk"));
        }
        format = u16_at(body, 24);
    }
    Ok(Fmt {
        format,
        channels: u16_at(body, 2) as usize,
        rate: u32_at(body, 4),
        bits: u16_at(body, 14) as usize,
    })
}

/// Minimal WAV reader (PCM 16/24/32-bit int, 32-bit float), mono-averaged.
pub fn load_wav(path: &str) -> Result<(Vec<f32>, u32), TtsError> {
    let p = Path::new(path);
    let bad = |detail: &str| TtsError::BadWav {
        path: p.to_path_buf(),
        detail: detail.into(),
    };
    let d = std::fs::read(p).map_err(|e| TtsError::Io {
        path: p.to_path_buf(),
        source: e,
    })?;
    if d.len() < 12 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut fmt: Option<Fmt> = None;
    let mut dat: Option<(usize, usize)> = None;
    while pos + 8 <= d.len() {
        let sz = u32_at(&d, pos + 4) as usize;
        let b = pos + 8;
        match &d[pos..pos + 4] {
            b"fmt " => {
                if b + sz > d.len() {
                    return Err(bad("truncated fmt chunk"));
                }
                fmt = Some(parse_fmt(p, &d[b..b + sz])?);
            }
            b"data" => {
                if b + sz > d.len() {
                    return Err(bad("truncated data chunk"));
                }
                dat = Some((b, sz));
            }
            _ => {}
        }
        pos = b + sz + (sz & 1);
    }
    let Fmt {
        format,
        channels: ch,
        rate,
        bits,
    } = fmt.ok_or_else(|| bad("no fmt chunk"))?;
    let (start, len) = dat.ok_or_else(|| bad("no data chunk"))?;
    if ch == 0 {
        return Err(bad("zero channels"));
    }
    let supported = matches!(
        (format, bits),
        (WAVE_FORMAT_PCM, 16)
            | (WAVE_FORMAT_PCM, 24)
            | (WAVE_FORMAT_PCM, 32)
            | (WAVE_FORMAT_IEEE_FLOAT, 32)
    );
    if !supported {
        return Err(TtsError::UnsupportedWav {
            path: p.to_path_buf(),
            detail: format!("format tag {format}, {bits}-bit"),
        });
    }
    if rate == 0 || rate > MAX_SAMPLE_RATE {
        return Err(TtsError::BadSampleRate {
            path: p.to_path_buf(),
            rate,
        });
    }
    let bytes = bits / 8;
    let n = len / (bytes * ch);
    if n == 0 {
        return Err(bad("no samples"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = 0f64;
        for c in 0..ch {
            let s = start + (i * ch + c) * bytes;
            acc += match (format, bits) {
                (WAVE_FORMAT_PCM, 16) => u16_at(&d, s) as i16 as f64 / 32768.0,
                (WAVE_FORMAT_PCM, 24) => {
                    (i32::from_le_bytes([0, d[s], d[s + 1], d[s + 2]]) >> 8) as f64 / 8388608.0
                }
                (WAVE_FORMAT_PCM, 32) => u32_at(&d, s) as i32 as f64 / 2147483648.0,
                _ => f32::from_bits(u32_at(&d, s)) as f64,
            };
        }
        out.push((acc / ch as f64) as f32);
    }
    Ok((out, rate))
}

pub fn write_wav_24k(path: &str, samples: &[f32]) -> Result<(), TtsError> {
    let mut b: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    let dl = (samples.len() * 2) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + dl).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24000u32.to_le_bytes());
    b.extend_from_slice(&48000u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&dl.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| TtsError::Io {
        path: PathBuf::from(path),
        source: e,
    })
}
