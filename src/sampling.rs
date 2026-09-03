use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

use crate::error::TtsError;

pub const CODEC_EOS: u32 = 2150;
pub const SEMANTIC_VOCAB: usize = 2048;
pub const CODEC_VOCAB: usize = 3072;

/// How the talker's semantic token and the code predictor's fifteen acoustic
/// codes are drawn.
///
/// `Default` is the official model's `generation_config.json` (sampled talker
/// at 0.9 / top-k 50 / no nucleus / repetition 1.05, sampled acoustic codes at
/// the same temperature and top-k). [`SamplerCfg::app`] is what the relay app
/// tuned for itself — a colder talker, nucleus 0.9, a stronger penalty, a
/// repetition-aware redraw and greedy acoustic codes — and it is not the
/// official algorithm. [`SamplerCfg::greedy`] removes every random draw and is
/// what the parity test against the official implementation uses.
#[derive(Clone, Copy, Debug)]
pub struct SamplerCfg {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f32,
    /// Fish-speech style redraw when the drawn token appeared in the last
    /// `n` tokens. Not part of the official model.
    pub ras_window: Option<usize>,
    pub ras_temperature: f64,
    pub ras_top_p: f64,
    pub min_new_tokens: usize,
    /// Sample the fifteen acoustic codes; `false` takes the argmax of each.
    pub cp_sample: bool,
    pub cp_temperature: f64,
    pub cp_top_k: usize,
    pub cp_top_p: f64,
    pub seed: u64,
}

impl Default for SamplerCfg {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            ras_window: None,
            ras_temperature: 1.0,
            ras_top_p: 0.9,
            min_new_tokens: 2,
            cp_sample: true,
            cp_temperature: 0.9,
            cp_top_k: 50,
            cp_top_p: 1.0,
            seed: 0,
        }
    }
}

impl SamplerCfg {
    /// The official `generation_config.json` values. Same as `Default`.
    pub fn official() -> Self {
        Self::default()
    }

    /// The relay app's tuning: colder, nucleus-clipped, penalised harder, with
    /// a repetition-aware redraw and greedy acoustic codes.
    pub fn app() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ras_window: Some(10),
            ras_temperature: 1.0,
            ras_top_p: 0.9,
            min_new_tokens: 2,
            cp_sample: false,
            cp_temperature: 1.0,
            cp_top_k: 0,
            cp_top_p: 1.0,
            seed: 0,
        }
    }

    /// No randomness anywhere: argmax for the talker (after the repetition
    /// penalty, as the official greedy path applies it) and for the codes.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            cp_sample: false,
            ..Self::default()
        }
    }
}

pub struct Sampler {
    cfg: SamplerCfg,
    rng: StdRng,
    // The acoustic codes draw from their own stream so switching them between
    // greedy and sampled leaves the talker's sequence for a seed unchanged.
    cp_rng: StdRng,
    seen: Vec<bool>,
    ras_recent: VecDeque<u32>,
    token_count: usize,
}

impl Sampler {
    pub fn new(cfg: SamplerCfg) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(cfg.seed),
            cp_rng: StdRng::seed_from_u64(cfg.seed ^ 0x9E37_79B9_7F4A_7C15),
            seen: vec![false; CODEC_VOCAB],
            ras_recent: VecDeque::new(),
            token_count: 0,
        }
    }

    pub fn cfg(&self) -> &SamplerCfg {
        &self.cfg
    }

    /// Suppression (only semantic vocab + EOS legal), min-new-tokens EOS gate,
    /// repetition penalty over seen tokens, temperature/top-k/top-p sampling,
    /// then the optional repetition-aware redraw.
    pub fn next_token(&mut self, logits: &[f32]) -> Result<u32, TtsError> {
        let mut base: Vec<f32> = logits[..CODEC_VOCAB.min(logits.len())].to_vec();
        for (i, v) in base.iter_mut().enumerate() {
            if i >= SEMANTIC_VOCAB && i as u32 != CODEC_EOS {
                *v = f32::NEG_INFINITY;
            }
        }
        if self.token_count < self.cfg.min_new_tokens {
            base[CODEC_EOS as usize] = f32::NEG_INFINITY;
        }

        let mut penalized = base.clone();
        if (self.cfg.repetition_penalty - 1.0).abs() > 1e-9 {
            let p = self.cfg.repetition_penalty;
            for (i, v) in penalized.iter_mut().enumerate() {
                if self.seen[i] && v.is_finite() {
                    *v = if *v > 0.0 { *v / p } else { *v * p };
                }
            }
        }

        let (t, k, p) = (self.cfg.temperature, self.cfg.top_k, self.cfg.top_p);
        let mut tok = draw(&mut self.rng, &penalized, t, k, p)?;
        if let Some(n) = self.cfg.ras_window {
            if self.ras_recent.contains(&tok) && tok != CODEC_EOS {
                let (rt, rp) = (self.cfg.ras_temperature, self.cfg.ras_top_p);
                tok = draw(&mut self.rng, &base, rt, 0, rp)?;
            }
            self.ras_recent.push_back(tok);
            while self.ras_recent.len() > n {
                self.ras_recent.pop_front();
            }
        }
        self.seen[tok as usize] = true;
        self.token_count += 1;
        Ok(tok)
    }

    /// One acoustic code from a code predictor group's logits.
    pub fn next_code(&mut self, logits: &[f32]) -> Result<u32, TtsError> {
        if !self.cfg.cp_sample {
            return argmax(logits);
        }
        let (t, k, p) = (
            self.cfg.cp_temperature,
            self.cfg.cp_top_k,
            self.cfg.cp_top_p,
        );
        draw(&mut self.cp_rng, logits, t, k, p)
    }
}

fn draw(
    rng: &mut StdRng,
    logits: &[f32],
    temperature: f64,
    top_k: usize,
    top_p: f64,
) -> Result<u32, TtsError> {
    let mut l: Vec<f32> = logits.to_vec();
    if temperature > 0.0 && (temperature - 1.0).abs() > 1e-12 {
        for v in l.iter_mut() {
            *v /= temperature as f32;
        }
    }
    if temperature < 0.01 {
        return argmax(&l);
    }
    if top_k > 0 && top_k < l.len() {
        let mut sorted: Vec<f32> = l.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted.len() > top_k {
            let thresh = sorted[top_k - 1];
            for v in l.iter_mut() {
                if *v < thresh {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..l.len()).filter(|&i| l[i].is_finite()).collect();
    if idx.is_empty() {
        return Err(TtsError::Numeric("no finite logits to sample from".into()));
    }
    if top_p > 0.0 && top_p < 1.0 {
        idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        let max = l[idx[0]];
        let exps: Vec<f32> = idx.iter().map(|&i| (l[i] - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut cum = 0f32;
        let mut cutoff = idx.len();
        for (rank, e) in exps.iter().enumerate() {
            cum += e / sum;
            if cum > top_p as f32 {
                cutoff = rank + 1;
                break;
            }
        }
        for &i in &idx[cutoff..] {
            l[i] = f32::NEG_INFINITY;
        }
    }
    let max = l
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = l
        .iter()
        .map(|v| if v.is_finite() { (v - max).exp() } else { 0.0 })
        .collect();
    let sum: f32 = exps.iter().sum();
    let r: f32 = rng.gen::<f32>() * sum;
    let mut cum = 0f32;
    for (i, e) in exps.iter().enumerate() {
        cum += e;
        if cum >= r {
            return Ok(i as u32);
        }
    }
    argmax(&l)
}

fn argmax(v: &[f32]) -> Result<u32, TtsError> {
    // First maximum on ties, the same choice the on-device argmax makes, so a
    // greedy code drawn here and one taken on the GPU agree.
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            continue;
        }
        match best {
            Some((_, b)) if x <= b => {}
            _ => best = Some((i, x)),
        }
    }
    best.map(|(i, _)| i as u32)
        .ok_or_else(|| TtsError::Numeric("no finite logits to take the argmax of".into()))
}
