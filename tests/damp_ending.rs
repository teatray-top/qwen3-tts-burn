//! `engine::damp_ending` swaps sentence-final punctuation for a comma. The set
//! it strips is `.`, `?`, `!`, the full-width `。`, `？`, `！`, the ellipsis
//! `…`, and spaces (src/engine.rs).

use qwen3_tts_burn::engine::damp_ending;

#[test]
fn each_terminal_mark_becomes_a_comma() {
    for mark in [".", "?", "!", "。", "？", "！", "…"] {
        assert_eq!(
            damp_ending(&format!("hello{mark}")),
            "hello,",
            "mark {mark:?}"
        );
    }
}

#[test]
fn repeated_and_mixed_marks_collapse_to_one_comma() {
    assert_eq!(damp_ending("really?!"), "really,");
    assert_eq!(damp_ending("wait..."), "wait,");
    assert_eq!(damp_ending("정말？！"), "정말,");
    assert_eq!(damp_ending("hm… ?"), "hm,");
}

#[test]
fn surrounding_whitespace_is_dropped() {
    assert_eq!(damp_ending("  hello.  "), "hello,");
    assert_eq!(damp_ending("hello . "), "hello,");
    assert_eq!(damp_ending("\thello?\n"), "hello,");
}

#[test]
fn text_without_terminal_mark_still_gets_a_comma() {
    assert_eq!(damp_ending("hello"), "hello,");
    assert_eq!(damp_ending("들리나요"), "들리나요,");
}

#[test]
fn interior_punctuation_is_untouched() {
    assert_eq!(damp_ending("one. two? three!"), "one. two? three,");
    assert_eq!(damp_ending("3.14 is pi."), "3.14 is pi,");
}

#[test]
fn only_punctuation_or_empty_is_returned_unchanged() {
    for s in ["", "   ", ".", "...", "?!", "。？！…", " . "] {
        assert_eq!(damp_ending(s), s, "input {s:?}");
    }
}

#[test]
fn other_terminal_characters_are_not_stripped() {
    assert_eq!(damp_ending("hello,"), "hello,,");
    assert_eq!(damp_ending("hello;"), "hello;,");
    assert_eq!(damp_ending("quote.\""), "quote.\",");
}
