use qwen3_tts_burn::{EngineOptions, KernelCache, TtsError};

#[test]
fn missing_model_dir_is_reported_before_any_gpu_work() {
    let err = qwen3_tts_burn::load_vulkan_with("definitely/not/a/model", EngineOptions::default())
        .err()
        .expect("must fail");
    match err {
        TtsError::ModelFileMissing { path } => {
            assert!(
                path.to_string_lossy().contains("model.safetensors"),
                "{path:?}"
            );
        }
        other => panic!("expected ModelFileMissing, got {other}"),
    }
}

#[test]
fn incomplete_model_dir_names_the_missing_file() {
    let dir = std::env::temp_dir().join(format!("qtb-load-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
    std::fs::write(dir.join("vocab.json"), b"{}").unwrap();
    let err = qwen3_tts_burn::load_vulkan_with(dir.to_str().unwrap(), EngineOptions::default())
        .err()
        .expect("must fail");
    let text = err.to_string();
    assert!(text.contains("merges.txt"), "{text}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn default_options_touch_nothing_global() {
    let o = EngineOptions::default();
    assert!(o.memory.is_none());
    assert_eq!(o.kernel_cache, KernelCache::Inherit);
    assert!(!o.power_hint);
    let a = EngineOptions::app();
    assert!(a.memory.is_some());
    assert_eq!(a.kernel_cache, KernelCache::Global);
}
