/// Locate where real speech begins so the ICL lead-in artifact can be dropped.
/// Ported verbatim from the relay app's `leading_trim`: 20 ms RMS windows,
/// threshold 0.005, bridge gaps <=80 ms, first sustained run >=200 ms is
/// speech, keep 60 ms pre-roll. Returns the sample index to start from (0 = no
/// trim). ICL conditioning prepends a short non-speech artifact that this
/// removes; without it the very start of the clip has a click/buzz.
pub fn leading_trim(samples: &[f32], sr: u32) -> usize {
    let win = (sr as usize * 20 / 1000).max(1);
    let nwin = samples.len() / win;
    if nwin == 0 {
        return 0;
    }
    const THRESHOLD: f32 = 0.005;
    const GAP_WIN: usize = 4;
    const MIN_WIN: usize = 10;
    const PREROLL_WIN: usize = 3;

    let active: Vec<bool> = (0..nwin)
        .map(|w| {
            let s = &samples[w * win..(w + 1) * win];
            let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
            rms > THRESHOLD
        })
        .collect();

    let mut start: Option<usize> = None;
    let mut last_active = 0usize;
    let mut gap = 0usize;
    for (w, &on) in active.iter().enumerate() {
        if on {
            if start.is_none() {
                start = Some(w);
            }
            last_active = w;
            gap = 0;
        } else if let Some(s) = start {
            gap += 1;
            if gap > GAP_WIN {
                if last_active - s + 1 >= MIN_WIN {
                    return s.saturating_sub(PREROLL_WIN) * win;
                }
                start = None;
                gap = 0;
            }
        }
    }
    if let Some(s) = start {
        if last_active - s + 1 >= MIN_WIN {
            return s.saturating_sub(PREROLL_WIN) * win;
        }
    }
    0
}

/// Sample index to cut AFTER, dropping trailing silence/artifact past the last
/// sustained speech. f16 generation tends to emit several seconds of near-silent
/// frames before EOS; this trims them. Keeps a 120 ms tail so releases aren't
/// clipped. Returns samples.len() if no trailing silence is found.
pub fn trailing_trim(samples: &[f32], sr: u32) -> usize {
    let win = (sr as usize * 20 / 1000).max(1);
    let nwin = samples.len() / win;
    if nwin == 0 {
        return samples.len();
    }
    const THRESHOLD: f32 = 0.005;
    const TAIL_WIN: usize = 8; // keep 160 ms after the last active window (final-phoneme decay)

    let mut last_active = None;
    for w in 0..nwin {
        let s = &samples[w * win..(w + 1) * win];
        let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
        if rms > THRESHOLD {
            last_active = Some(w);
        }
    }
    match last_active {
        Some(w) => ((w + 1 + TAIL_WIN) * win).min(samples.len()),
        None => samples.len(),
    }
}

/// Tight end trim for the ENDING (not silence removal). The model, releasing the
/// last vowel toward the dropped damping word across the pause, gives a long slow
/// taper (~220 ms down to silence) that drags ("질질 끌림"). Keep the phoneme plus a
/// natural ~60 ms release by cutting where the level falls below 45% of the
/// utterance's peak, rather than all the way down to the noise floor.
pub fn release_trim(samples: &[f32], sr: u32) -> usize {
    let win = (sr as usize * 20 / 1000).max(1);
    let nwin = samples.len() / win;
    if nwin == 0 {
        return samples.len();
    }
    let rms = |w: usize| -> f32 {
        let s = &samples[w * win..(w + 1) * win];
        (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    };
    let peak = (0..nwin).map(rms).fold(0f32, f32::max);
    let thr = (peak * 0.45).max(0.006);
    let last = (0..nwin).rev().find(|&w| rms(w) > thr).unwrap_or(0);
    ((last + 1 + 3) * win).min(samples.len())
}

/// Bounds of the speech in a reference clip, as a sample range.
///
/// A reference is shown to the model as an example of how an utterance is
/// delivered, silence included: a clip carrying a second of trailing room tone
/// teaches the model to fade out instead of articulating its last syllable, and
/// the same silence dilutes the speaker vector. Both ends are cut back to a
/// short margin.
pub fn speech_bounds(samples: &[f32], sr: u32) -> (usize, usize) {
    let win = (sr as usize * 20 / 1000).max(1);
    let nwin = samples.len() / win;
    if nwin == 0 {
        return (0, samples.len());
    }
    let rms = |w: usize| -> f32 {
        let s = &samples[w * win..(w + 1) * win];
        (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    };
    let peak = (0..nwin).map(rms).fold(0f32, f32::max);
    if peak <= 0.0 {
        return (0, samples.len());
    }
    let thr = (peak * 0.03).max(1e-4);
    let first = (0..nwin).find(|&w| rms(w) > thr).unwrap_or(0);
    let last = (0..nwin).rev().find(|&w| rms(w) > thr).unwrap_or(nwin - 1);
    const MARGIN: usize = 2; // 40 ms
    let start = first.saturating_sub(MARGIN) * win;
    let end = ((last + 1 + MARGIN) * win).min(samples.len());
    (start, end.max(start))
}

/// [`speech_bounds`] applied.
pub fn trim_silence(samples: &[f32], sr: u32) -> Vec<f32> {
    let (a, b) = speech_bounds(samples, sr);
    samples[a..b].to_vec()
}
