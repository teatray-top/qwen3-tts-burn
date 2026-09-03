//! CPU-only checks that bad input reaches the caller as a `TtsError` instead
//! of a panic: hand-built WAV files, a too-short mel input, a git-lfs pointer
//! where a safetensors file should be, and the Display strings.

use std::path::{Path, PathBuf};

use qwen3_tts_burn::audio::load_wav;
use qwen3_tts_burn::mel;
use qwen3_tts_burn::resample::resample;
use qwen3_tts_burn::weights::WeightFile;
use qwen3_tts_burn::TtsError;

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("qtb-errors-{}-{name}", std::process::id()))
}

fn with_file<T>(name: &str, bytes: &[u8], f: impl FnOnce(&str) -> T) -> T {
    let path = temp(name);
    std::fs::write(&path, bytes).expect("write fixture");
    let out = f(path.to_str().expect("utf-8 temp path"));
    let _ = std::fs::remove_file(&path);
    out
}

/// RIFF/WAVE with one `fmt ` chunk (16 bytes) and one `data` chunk.
fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&format.to_le_bytes());
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * block as u32).to_le_bytes());
    b.extend_from_slice(&block.to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data.len() as u32).to_le_bytes());
    b.extend_from_slice(data);
    b
}

fn pcm16(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn load_wav_rejects_non_riff() {
    let err = with_file("junk.wav", b"this is not a wav file at all", |p| {
        load_wav(p).unwrap_err()
    });
    assert!(matches!(err, TtsError::BadWav { .. }), "{err:?}");
}

#[test]
fn load_wav_rejects_truncated_fmt_chunk() {
    let full = wav(1, 1, 24_000, 16, &pcm16(&[0, 1, 2, 3]));
    let err = with_file("trunc_fmt.wav", &full[..24], |p| load_wav(p).unwrap_err());
    match err {
        TtsError::BadWav { detail, .. } => assert!(detail.contains("fmt"), "{detail}"),
        other => panic!("expected BadWav, got {other:?}"),
    }
}

#[test]
fn load_wav_rejects_truncated_data_chunk() {
    let mut full = wav(1, 1, 24_000, 16, &pcm16(&[0, 1, 2, 3]));
    full.truncate(full.len() - 3);
    let err = with_file("trunc_data.wav", &full, |p| load_wav(p).unwrap_err());
    match err {
        TtsError::BadWav { detail, .. } => assert!(detail.contains("data"), "{detail}"),
        other => panic!("expected BadWav, got {other:?}"),
    }
}

#[test]
fn load_wav_rejects_8_bit() {
    let bytes = wav(1, 1, 24_000, 8, &[128, 255, 0, 64]);
    let err = with_file("8bit.wav", &bytes, |p| load_wav(p).unwrap_err());
    assert!(matches!(err, TtsError::UnsupportedWav { .. }), "{err:?}");
}

#[test]
fn load_wav_rejects_zero_sample_rate() {
    let bytes = wav(1, 1, 0, 16, &pcm16(&[1, 2, 3, 4]));
    let err = with_file("rate0.wav", &bytes, |p| load_wav(p).unwrap_err());
    assert!(
        matches!(err, TtsError::BadSampleRate { rate: 0, .. }),
        "{err:?}"
    );
}

#[test]
fn load_wav_rejects_absurd_sample_rate() {
    let bytes = wav(1, 1, 10_000_000, 16, &pcm16(&[1, 2, 3, 4]));
    let err = with_file("rate_huge.wav", &bytes, |p| load_wav(p).unwrap_err());
    assert!(
        matches!(
            err,
            TtsError::BadSampleRate {
                rate: 10_000_000,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn load_wav_rejects_empty_data() {
    let bytes = wav(1, 1, 24_000, 16, &[]);
    let err = with_file("empty.wav", &bytes, |p| load_wav(p).unwrap_err());
    match err {
        TtsError::BadWav { detail, .. } => assert!(detail.contains("no samples"), "{detail}"),
        other => panic!("expected BadWav, got {other:?}"),
    }
}

#[test]
fn load_wav_reads_24_bit_pcm() {
    let data: Vec<u8> = [0x7F_FF_FFi32, -0x80_00_00, 0, 0x40_00_00]
        .iter()
        .flat_map(|v| v.to_le_bytes()[..3].to_vec())
        .collect();
    let bytes = wav(1, 1, 48_000, 24, &data);
    let (samples, rate) = with_file("24bit.wav", &bytes, |p| load_wav(p).expect("24-bit wav"));
    assert_eq!(rate, 48_000);
    assert_eq!(samples.len(), 4);
    assert!(
        (samples[0] - 8388607.0 / 8388608.0).abs() < 1e-6,
        "{}",
        samples[0]
    );
    assert_eq!(samples[1], -1.0);
    assert_eq!(samples[2], 0.0);
    assert_eq!(samples[3], 0.5);
}

#[test]
fn load_wav_reads_float32_and_downmixes_stereo() {
    let data: Vec<u8> = [0.5f32, -0.5, -0.25, 0.75]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bytes = wav(3, 2, 24_000, 32, &data);
    let (samples, rate) = with_file("f32.wav", &bytes, |p| load_wav(p).expect("float wav"));
    assert_eq!(rate, 24_000);
    assert_eq!(samples, vec![0.0, 0.25]);
}

#[test]
fn load_wav_reads_16_bit_pcm() {
    let bytes = wav(1, 1, 16_000, 16, &pcm16(&[i16::MIN, 0, 16384]));
    let (samples, rate) = with_file("16bit.wav", &bytes, |p| load_wav(p).expect("16-bit wav"));
    assert_eq!(rate, 16_000);
    assert_eq!(samples, vec![-1.0, 0.0, 0.5]);
}

#[test]
fn load_wav_reports_missing_file_with_path() {
    let path = temp("does-not-exist.wav");
    let err = load_wav(path.to_str().unwrap()).unwrap_err();
    match &err {
        TtsError::Io { path: p, .. } => assert_eq!(p, &path),
        other => panic!("expected Io, got {other:?}"),
    }
    assert!(err.to_string().contains(path.to_str().unwrap()));
}

#[test]
fn log_mel_rejects_too_short_input() {
    let err = mel::log_mel(&[0.0; 100]).unwrap_err();
    match err {
        TtsError::ReferenceTooShort { samples_ms, min_ms } => {
            assert!(samples_ms < min_ms, "{samples_ms} vs {min_ms}");
            assert!((samples_ms - 100.0 * 1000.0 / 24000.0).abs() < 1e-9);
        }
        other => panic!("expected ReferenceTooShort, got {other:?}"),
    }
}

#[test]
fn log_mel_accepts_min_samples() {
    let mel = mel::log_mel(&vec![0.01; mel::MIN_SAMPLES]).expect("MIN_SAMPLES is enough");
    assert_eq!(mel.len(), 128);
    assert!(!mel[0].is_empty());
    assert!(mel::log_mel(&vec![0.01; mel::MIN_SAMPLES - 1]).is_err());
}

#[test]
fn resample_rejects_zero_rates() {
    let zero =
        |r: Result<Vec<f32>, TtsError>| matches!(r, Err(TtsError::BadSampleRate { rate: 0, .. }));
    assert!(zero(resample(&[0.0; 10], 0, 24_000)));
    assert!(zero(resample(&[0.0; 10], 16_000, 0)));
    assert_eq!(
        resample(&[1.0, 2.0], 24_000, 24_000).unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn weight_file_rejects_git_lfs_pointer() {
    let pointer = b"version https://git-lfs.github.com/spec/v1\n\
                    oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n\
                    size 1234567890\n";
    let path = temp("model.safetensors");
    std::fs::write(&path, pointer).expect("write");
    let err = WeightFile::open(&path)
        .err()
        .expect("lfs pointer must be rejected");
    match &err {
        TtsError::BadSafetensors { path: p, .. } => assert_eq!(p, &path),
        other => panic!("expected BadSafetensors, got {other:?}"),
    }
    assert!(err.to_string().contains(path.to_str().unwrap()));
    let _ = std::fs::remove_file(path);
}

#[test]
fn display_strings_name_the_path() {
    let path = Path::new("C:/voices/ref.wav").to_path_buf();
    let cases = vec![
        TtsError::Io {
            path: path.clone(),
            source: std::io::Error::other("boom"),
        },
        TtsError::ModelFileMissing { path: path.clone() },
        TtsError::BadSafetensors {
            path: path.clone(),
            detail: "x".into(),
        },
        TtsError::BadWav {
            path: path.clone(),
            detail: "x".into(),
        },
        TtsError::UnsupportedWav {
            path: path.clone(),
            detail: "x".into(),
        },
        TtsError::BadSampleRate {
            path: path.clone(),
            rate: 7,
        },
    ];
    for e in &cases {
        let s = e.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("C:/voices/ref.wav"), "{s}");
        assert!(!s.contains('\n'), "{s}");
    }
    let rest = vec![
        TtsError::MissingTensor { name: "t.w".into() },
        TtsError::TensorShape {
            name: "t.w".into(),
            expected: "2D".into(),
            got: vec![3],
        },
        TtsError::TensorDtype {
            name: "t.w".into(),
            dtype: "I8".into(),
        },
        TtsError::Tokenizer("x".into()),
        TtsError::Gpu("x".into()),
        TtsError::ReferenceTooShort {
            samples_ms: 1.0,
            min_ms: 2.0,
        },
        TtsError::ReferenceTooLong { frames: 3, max: 2 },
        TtsError::EmptyText,
        TtsError::EmptyReferenceText,
        TtsError::InvalidPrompt("x".into()),
        TtsError::InvalidFrames("x".into()),
        TtsError::InvalidConfig("x".into()),
        TtsError::Numeric("x".into()),
    ];
    for e in rest {
        let s: String = e.into();
        assert!(!s.is_empty());
        assert!(!s.contains('\n'), "{s}");
    }
}
