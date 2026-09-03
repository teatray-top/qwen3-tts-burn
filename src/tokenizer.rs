use std::path::Path;

use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::error::TtsError;

const PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct TextTokenizer {
    inner: Tokenizer,
}

impl TextTokenizer {
    /// Mirrors qwen3-tts-rs `TextTokenizer::from_vocab_and_merges` exactly
    /// (Qwen2 byte-level BPE: NFC + isolated-split regex + ByteLevel, special
    /// tokens from tokenizer_config.json's added_tokens_decoder).
    pub fn from_dir(dir: &Path) -> Result<Self, TtsError> {
        let tk_err =
            |what: &str, e: &dyn std::fmt::Display| TtsError::Tokenizer(format!("{what}: {e}"));
        for name in ["vocab.json", "merges.txt"] {
            let p = dir.join(name);
            if !p.is_file() {
                return Err(TtsError::ModelFileMissing { path: p });
            }
        }
        let bpe = BPE::from_file(
            &dir.join("vocab.json").to_string_lossy(),
            &dir.join("merges.txt").to_string_lossy(),
        )
        .unk_token("<|endoftext|>".to_string())
        .byte_fallback(false)
        .build()
        .map_err(|e| tk_err("bpe", &e))?;

        let mut tk = Tokenizer::new(bpe);
        tk.with_normalizer(Some(NFC));
        let split = Split::new(PRETOKENIZE_REGEX, SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| tk_err("split", &e))?;
        let byte_level = ByteLevel::new(false, false, false);
        tk.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), byte_level.into()])));
        tk.with_post_processor(Some(ByteLevel::new(false, false, false)));
        tk.with_decoder(Some(ByteLevel::new(false, false, false)));

        let cfg_path = dir.join("tokenizer_config.json");
        if cfg_path.exists() {
            let content = std::fs::read_to_string(&cfg_path).map_err(|e| TtsError::Io {
                path: cfg_path.clone(),
                source: e,
            })?;
            let cfg: serde_json::Value =
                serde_json::from_str(&content).map_err(|e| tk_err("tokenizer_config.json", &e))?;
            if let Some(added) = cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                let mut specials = Vec::new();
                for (_, info) in added {
                    let Some(content) = info.get("content").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if info
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        specials.push(AddedToken::from(content.to_string(), true));
                    }
                }
                tk.add_special_tokens(&specials);
            }
        }
        Ok(Self { inner: tk })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::Tokenizer(format!("encode: {e}")))?
            .get_ids()
            .to_vec())
    }
}
