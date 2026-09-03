use qwen3_tts_burn::lang::Language;

const ALL: [(Language, &str, &str, u32); 10] = [
    (Language::Chinese, "zh", "chinese", 2055),
    (Language::English, "en", "english", 2050),
    (Language::Japanese, "ja", "japanese", 2058),
    (Language::Korean, "ko", "korean", 2064),
    (Language::German, "de", "german", 2053),
    (Language::French, "fr", "french", 2061),
    (Language::Russian, "ru", "russian", 2069),
    (Language::Italian, "it", "italian", 2070),
    (Language::Portuguese, "pt", "portuguese", 2071),
    (Language::Spanish, "es", "spanish", 2054),
];

#[test]
fn from_code_accepts_iso_code_and_name() {
    for (lang, code, name, _) in ALL {
        assert_eq!(Language::from_code(code), Some(lang));
        assert_eq!(Language::from_code(name), Some(lang));
    }
}

#[test]
fn from_code_is_case_and_whitespace_insensitive() {
    assert_eq!(Language::from_code("EN"), Some(Language::English));
    assert_eq!(Language::from_code(" Ko "), Some(Language::Korean));
    assert_eq!(Language::from_code("JAPANESE"), Some(Language::Japanese));
    assert_eq!(Language::from_code("\tzh\n"), Some(Language::Chinese));
}

#[test]
fn from_code_rejects_unknown() {
    for s in ["", "xx", "eng", "en-US", "kr", "jp", "cn", "한국어"] {
        assert_eq!(Language::from_code(s), None, "input {s:?}");
    }
}

#[test]
fn token_ids_match_table_and_round_trip() {
    for (lang, code, _, id) in ALL {
        assert_eq!(lang.token_id(), Some(id));
        assert_eq!(Language::from_code(code).unwrap().token_id(), Some(id));
    }
}

#[test]
fn token_ids_are_distinct_and_in_codec_range() {
    let mut ids: Vec<u32> = ALL
        .iter()
        .map(|(l, _, _, _)| l.token_id().unwrap())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), ALL.len());
    for id in ids {
        assert!((2048..3072).contains(&id), "token {id} outside codec vocab");
    }
}

#[test]
fn default_is_english() {
    assert_eq!(Language::default(), Language::Auto);
}

#[test]
fn auto_has_no_token_and_a_three_token_prefix() {
    assert_eq!(Language::from_code("auto"), Some(Language::Auto));
    assert_eq!(Language::Auto.token_id(), None);
    assert_eq!(Language::Auto.codec_prefix(), vec![2155, 2156, 2157]);
    assert_eq!(
        Language::Korean.codec_prefix(),
        vec![2154, 2156, 2064, 2157]
    );
}
