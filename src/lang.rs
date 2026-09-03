//! Codec language tokens.
//!
//! The talker's prefill carries a language token, which the model uses to pick
//! the phonology it generates in. The ten ids below are `codec_language_id`
//! from the model's config.json.

/// Languages the codec has a prefix token for, plus `Auto`, the official
/// default, which sends no language token and lets the model infer one from
/// the text.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Language {
    #[default]
    Auto,
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Italian,
    Portuguese,
    Spanish,
}

impl Language {
    /// The codec language token; `None` for `Auto`.
    pub fn token_id(self) -> Option<u32> {
        Some(match self {
            Language::Auto => return None,
            Language::Chinese => 2055,
            Language::English => 2050,
            Language::Japanese => 2058,
            Language::Korean => 2064,
            Language::German => 2053,
            Language::French => 2061,
            Language::Russian => 2069,
            Language::Italian => 2070,
            Language::Portuguese => 2071,
            Language::Spanish => 2054,
        })
    }

    /// The codec prefix the talker's prefill starts with, before the speaker
    /// slot: think tokens around a language id, or the three-token "no
    /// think" prefix when the language is left to the model.
    pub fn codec_prefix(self) -> Vec<u32> {
        const THINK: u32 = 2154;
        const NOTHINK: u32 = 2155;
        const THINK_BOS: u32 = 2156;
        const THINK_EOS: u32 = 2157;
        match self.token_id() {
            Some(id) => vec![THINK, THINK_BOS, id, THINK_EOS],
            None => vec![NOTHINK, THINK_BOS, THINK_EOS],
        }
    }

    /// The codes `from_code` accepts, in the order they are listed to users.
    pub fn codes() -> &'static [&'static str] {
        &[
            "auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "it", "pt", "es",
        ]
    }

    pub fn from_code(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Language::Auto),
            "zh" | "chinese" => Some(Language::Chinese),
            "en" | "english" => Some(Language::English),
            "ja" | "japanese" => Some(Language::Japanese),
            "ko" | "korean" => Some(Language::Korean),
            "de" | "german" => Some(Language::German),
            "fr" | "french" => Some(Language::French),
            "ru" | "russian" => Some(Language::Russian),
            "it" | "italian" => Some(Language::Italian),
            "pt" | "portuguese" => Some(Language::Portuguese),
            "es" | "spanish" => Some(Language::Spanish),
            _ => None,
        }
    }
}
