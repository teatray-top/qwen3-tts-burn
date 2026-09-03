//! `weights::WeightFile` on a tiny hand-built safetensors file.
//!
//! `HostTable` has no public constructor: its `data` field is private and the
//! only way to build one is `WeightFile::rows_f16` (src/weights.rs), which is
//! what these tests go through. `HostTable::gather` itself is not called: it
//! returns a burn `Tensor<B, 2>` and the crate enables only the wgpu/Vulkan
//! backend (Cargo.toml), so calling it needs a GPU device. It stays untested
//! here rather than adding a CPU backend feature to the crate.

use std::path::PathBuf;

use qwen3_tts_burn::weights::WeightFile;
use qwen3_tts_burn::TtsError;

/// Layout per the safetensors format: u64 LE header length, JSON header
/// (space-padded to a multiple of 8), then the tensor bytes in offset order.
fn write_fixture(name: &str) -> PathBuf {
    let t: Vec<u8> = (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let u: Vec<u8> = [0x3C00u16, 0x4000, 0x4200, 0x4400]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let b: Vec<u8> = [1.0f32, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (o1, o2, o3) = (t.len(), t.len() + u.len(), t.len() + u.len() + b.len());
    let mut header = format!(
        "{{\"t\":{{\"dtype\":\"F32\",\"shape\":[3,4],\"data_offsets\":[0,{o1}]}},\
         \"u\":{{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[{o1},{o2}]}},\
         \"b\":{{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[{o2},{o3}]}}}}"
    );
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&t);
    bytes.extend_from_slice(&u);
    bytes.extend_from_slice(&b);

    let path = std::env::temp_dir().join(format!(
        "qtb-weights-{}-{name}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

#[test]
fn open_reports_tensor_names() {
    let path = write_fixture("names");
    let wf = WeightFile::open(&path).expect("open");
    assert!(wf.has("t"));
    assert!(wf.has("u"));
    assert!(wf.has("b"));
    assert!(!wf.has("missing"));
    assert!(!wf.has(""));
    drop(wf);
    let _ = std::fs::remove_file(path);
}

#[test]
fn rows_f16_reports_shape_for_f32_and_f16_sources() {
    let path = write_fixture("shape");
    let wf = WeightFile::open(&path).expect("open");
    let t = wf.rows_f16("t").expect("t");
    assert_eq!((t.rows, t.cols), (3, 4));
    let u = wf.rows_f16("u").expect("u");
    assert_eq!((u.rows, u.cols), (2, 2));
    drop(wf);
    let _ = std::fs::remove_file(path);
}

#[test]
fn rows_f16_rejects_1d_tensor() {
    let path = write_fixture("1d");
    let wf = WeightFile::open(&path).expect("open");
    let err = wf.rows_f16("b").err().expect("1D tensor must be rejected");
    match &err {
        TtsError::TensorShape {
            name,
            expected,
            got,
        } => {
            assert_eq!(name, "b");
            assert_eq!(expected, "2D");
            assert_eq!(got, &[3]);
        }
        other => panic!("expected TensorShape, got {other:?}"),
    }
    assert!(err.to_string().contains("b"));
    drop(wf);
    let _ = std::fs::remove_file(path);
}

#[test]
fn rows_f16_reports_missing_name() {
    let path = write_fixture("missing");
    let wf = WeightFile::open(&path).expect("open");
    let err = wf
        .rows_f16("nope")
        .err()
        .expect("missing tensor must be rejected");
    match &err {
        TtsError::MissingTensor { name } => assert_eq!(name, "nope"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
    assert!(err.to_string().contains("nope"));
    drop(wf);
    let _ = std::fs::remove_file(path);
}

#[test]
fn open_fails_on_missing_file() {
    let path = std::env::temp_dir().join("qtb-weights-does-not-exist.safetensors");
    match WeightFile::open(&path) {
        Err(TtsError::ModelFileMissing { path: p }) => assert_eq!(p, path),
        Err(other) => panic!("expected ModelFileMissing, got {other:?}"),
        Ok(_) => panic!("open must fail on a missing file"),
    }
}

#[test]
fn open_rejects_a_non_safetensors_file() {
    let path = std::env::temp_dir().join(format!(
        "qtb-weights-{}-junk.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, b"not a safetensors file").expect("write");
    match WeightFile::open(&path) {
        Err(TtsError::BadSafetensors { path: p, detail }) => {
            assert_eq!(p, path);
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("expected BadSafetensors, got {other:?}"),
        Ok(_) => panic!("open must reject a non-safetensors file"),
    }
    let _ = std::fs::remove_file(path);
}
