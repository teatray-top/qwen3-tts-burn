use burn::prelude::*;

use crate::code_predictor::CodePredictor;
use crate::decoder::Decoder12Hz;
use crate::error::TtsError;
use crate::sampling::{Sampler, SamplerCfg, CODEC_EOS};
use crate::talker::{KvCache, Talker};

pub const IM_START: u32 = 151644;
pub const ASSISTANT: u32 = 77091;
pub const NEWLINE: u32 = 198;
pub const TTS_PAD: u32 = 151671;
pub const TTS_BOS: u32 = 151672;
pub const TTS_EOS: u32 = 151673;
pub const CODEC_PAD: u32 = 2148;
pub const CODEC_BOS: u32 = 2149;
pub const CODEC_THINK: u32 = 2154;
pub const CODEC_THINK_BOS: u32 = 2156;
pub const CODEC_THINK_EOS: u32 = 2157;

pub struct ClonePrompt {
    /// Codec language token for the prefill. The model picks its
    /// phonology from this, so it has to match the text being spoken.
    pub language: crate::lang::Language,
    pub speaker_embedding: Vec<f32>,
    pub ref_codes: Vec<Vec<u32>>,
    pub ref_text_ids: Vec<u32>,
}

pub struct Pipeline<B: Backend> {
    pub talker: Talker<B>,
    pub cp: CodePredictor<B>,
    pub decoder: Decoder12Hz<B>,
}

impl<B: Backend> Pipeline<B> {
    /// Voice-clone generation mirroring the candle port's
    /// `prefill_voice_clone` (ICL mode) + ICL prompt + frame loop.
    /// Returns frames of 16 codes each.
    pub fn generate(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        scfg: SamplerCfg,
        max_frames: usize,
        dev: &B::Device,
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        self.generate_cb(
            text_ids,
            prompt,
            scfg,
            max_frames,
            false,
            false,
            dev,
            &mut |_| true,
        )
    }

    /// Like `generate` but calls `on_frame` after each frame for streaming; with
    /// `speak_all`, EOS is held until all text is spoken plus a few release frames.
    /// `on_frame` returns false to stop generating — f16 makes EOS flaky, so the
    /// caller (which decodes the audio) can cut a run that is only emitting
    /// trailing silence instead of letting it run to `max_frames`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_cb(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        scfg: SamplerCfg,
        max_frames: usize,
        hold_eos: bool,
        attn_boost: bool,
        dev: &B::Device,
        on_frame: &mut dyn FnMut(&[Vec<u32>]) -> bool,
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        if text_ids.is_empty() {
            return Err(TtsError::EmptyText);
        }
        let Prefill {
            mut cache,
            mut offset,
            mut last_hidden,
            mut logits,
            trailing,
            trailing_len,
            boost_boundary,
            max_frames,
            hd,
            tts_pad_embed,
        } = self.prefill(text_ids, prompt, max_frames, attn_boost, dev)?;

        let mut cp_cache = self.cp.new_cache(dev);
        let mut sampler = Sampler::new(scfg);
        const RELEASE_FRAMES: usize = 3;
        // Keep the model speaking a few frames PAST the last text token: each
        // token's acoustic spans ~1-2 frames after it is fed, so allowing EOS the
        // instant the text ends clips the final phoneme mid-articulation (the
        // recurring "마지막 음소가 안 끝났는데 끝난다"). During the grace, pad text is
        // fed, so the model finishes/decays the last phoneme instead of starting a
        // new one; trailing_trim then removes the true silence afterward.
        const EOS_GRACE: usize = 3;
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut releasing: usize = 0;
        for frame_idx in 0..max_frames {
            // With hold_eos: mask EOS until all text is fed plus a grace tail, then
            // let a short release play out. Without it, stop at the first EOS as
            // the official implementation does.
            let block_eos = hold_eos && (frame_idx < trailing_len + EOS_GRACE || releasing > 0);
            let tok = if block_eos {
                let mut lg = logits.clone();
                if let Some(v) = lg.get_mut(CODEC_EOS as usize) {
                    *v = f32::NEG_INFINITY;
                }
                sampler.next_token(&lg)?
            } else {
                sampler.next_token(&logits)?
            };
            let tok = if tok == CODEC_EOS {
                if !hold_eos {
                    break;
                }
                releasing = RELEASE_FRAMES;
                let mut lg = logits.clone();
                if let Some(v) = lg.get_mut(CODEC_EOS as usize) {
                    *v = f32::NEG_INFINITY;
                }
                sampler.next_token(&lg)?
            } else {
                tok
            };
            let stop_after = releasing > 0 && {
                releasing -= 1;
                releasing == 0
            };
            let semantic_embed = self.talker.embed_codec_ids(&[tok], dev)?;
            let codes = self.cp.generate_codes(
                last_hidden.clone(),
                semantic_embed.clone(),
                &mut cp_cache,
                &mut sampler,
                dev,
            )?;

            let mut frame = vec![tok];
            frame.extend_from_slice(&codes);
            frames.push(frame);
            if !on_frame(&frames) {
                break;
            }

            let mut summed = semantic_embed;
            for (g, &c) in codes.iter().enumerate() {
                summed = summed + self.cp.embed_group(g, c, dev);
            }
            let text_add = if frame_idx < trailing_len {
                trailing
                    .clone()
                    .slice([0..1, frame_idx..frame_idx + 1, 0..hd])
            } else {
                tts_pad_embed.clone()
            };
            let step_in = summed + text_add;
            let hh =
                self.talker
                    .run_cached(step_in, &mut cache, offset, false, boost_boundary, dev);
            offset += 1;
            last_hidden = hh;
            logits = self.talker.head_logits(last_hidden.clone())?;
            if stop_after {
                break;
            }
        }
        Ok(frames)
    }

    pub fn synthesize(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        scfg: SamplerCfg,
        max_frames: usize,
        dev: &B::Device,
    ) -> Result<Vec<f32>, TtsError> {
        let frames = self.generate(text_ids, prompt, scfg, max_frames, dev)?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.decoder.decode(&frames, dev)
    }
}

/// Everything the frame loop starts from: the KV cache after the role header,
/// codec prefix, speaker vector and in-context reference; the logits the first
/// frame is drawn from; and the text that is fed one token per frame.
struct Prefill<B: Backend> {
    cache: KvCache<B>,
    offset: usize,
    last_hidden: Tensor<B, 3>,
    logits: Vec<f32>,
    trailing: Tensor<B, 3>,
    trailing_len: usize,
    boost_boundary: usize,
    max_frames: usize,
    hd: usize,
    tts_pad_embed: Tensor<B, 3>,
}

impl<B: Backend> Pipeline<B> {
    fn prefill(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        max_frames: usize,
        attn_boost: bool,
        dev: &B::Device,
    ) -> Result<Prefill<B>, TtsError> {
        // Size the KV cache to a bucket that fits this utterance's prefill +
        // generation. run_cached attends over the FULL cache max every step
        // (masked), so a fixed large cache would slow every short message; a fixed
        // small one (was 448 ≈ 30s) overflows and kills a long passage. Buckets
        // keep short messages fast and let long text run. The mask zeroes the
        // unused positions, so output (parity) is identical to any larger cache.
        // The prefill is only the ICL prefix (~ref frames + a small fixed header);
        // the input text is fed ONE token per generated frame, not prefilled — so
        // sizing by text_ids over-counted and forced a needlessly large (slow)
        // cache. Positions used ≈ prefill + generated frames.
        let prefill_est = prompt.ref_codes.len() + 24;
        let kv_max: usize = match prefill_est.saturating_add(max_frames) {
            n if n <= 448 => 448,
            n if n <= 1024 => 1024,
            _ => 2048,
        };
        let max_frames = max_frames.min(kv_max.saturating_sub(prefill_est + 8));
        let mut cache = KvCache::new(
            self.talker.cfg.layers,
            self.talker.cfg.kv_heads,
            kv_max,
            self.talker.cfg.head_dim,
            dev,
        );
        let is_icl = !prompt.ref_codes.is_empty();

        let role = self
            .talker
            .embed_text(&[IM_START, ASSISTANT, NEWLINE], dev)?;
        let prefix_ids = prompt.language.codec_prefix();
        let codec_prefix = self.talker.embed_codec_ids(&prefix_ids, dev)?;
        let spk: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(
                prompt.speaker_embedding.clone(),
                [1, 1, prompt.speaker_embedding.len()],
            ),
            dev,
        );
        let codec_suffix = self.talker.embed_codec_ids(&[CODEC_PAD, CODEC_BOS], dev)?;
        let codec_embed = Tensor::cat(vec![codec_prefix, spk, codec_suffix], 1);

        // Text side of the prefill: pads under every codec position but the
        // last, then TTS_BOS; the last codec position (BOS) pairs with the
        // first text token instead. Seven codec positions with a language
        // token, six without one.
        let [_, codec_len, hd] = codec_embed.dims();
        let mut pad_ids = vec![TTS_PAD; codec_len - 2];
        pad_ids.push(TTS_BOS);
        let tts_text = self.talker.embed_text(&pad_ids, dev)?;
        let codec_head = codec_embed.clone().slice([0..1, 0..codec_len - 1, 0..hd]);
        let overlay = tts_text + codec_head;
        let mut hidden_in = Tensor::cat(vec![role, overlay], 1);
        if !is_icl {
            let bos = codec_embed.slice([0..1, codec_len - 1..codec_len, 0..hd]);
            let first_text = self.talker.embed_text(&[text_ids[0]], dev)?;
            hidden_in = Tensor::cat(vec![hidden_in, first_text + bos], 1);
        }
        let [_, prefill_len, _] = hidden_in.dims();
        let h = self
            .talker
            .run_cached(hidden_in, &mut cache, 0, true, 0, dev);
        let mut offset = prefill_len;
        let mut last_hidden = h.clone().slice([0..1, prefill_len - 1..prefill_len, 0..hd]);
        let mut logits = self.talker.head_logits(last_hidden.clone())?;

        let tts_pad_embed = self.talker.embed_text(&[TTS_PAD], dev)?;
        // Attention-boost the ICL reference span [0..boost_boundary]; 0 = no boost.
        let mut boost_boundary = 0usize;
        let trailing = if is_icl {
            let t_ref = prompt.ref_codes.len();
            let sem: Vec<u32> = prompt.ref_codes.iter().map(|f| f[0]).collect();
            let mut ref_sum = self.talker.embed_codec_ids(&sem, dev)?;
            for g in 1..16 {
                let col: Vec<i32> = prompt.ref_codes.iter().map(|f| f[g] as i32).collect();
                let idx: Tensor<B, 1, burn::tensor::Int> =
                    Tensor::from_data(burn::tensor::TensorData::new(col, [t_ref]), dev);
                let e = self.cp.embed_group_batch(g - 1, idx);
                ref_sum = ref_sum + e;
            }

            let mut all_text: Vec<u32> = prompt.ref_text_ids.clone();
            all_text.extend_from_slice(text_ids);
            all_text.push(TTS_EOS);
            let text_embed = self.talker.embed_text(&all_text, dev)?;
            let n_text = all_text.len();

            let bos = self.talker.embed_codec_ids(&[CODEC_BOS], dev)?;
            let codec2 = Tensor::cat(vec![bos, ref_sum], 1);
            let n_codec = t_ref + 1;

            let (icl_embed, trailing) = if n_text > n_codec {
                let head = text_embed.clone().slice([0..1, 0..n_codec, 0..hd]);
                let tail = text_embed.slice([0..1, n_codec..n_text, 0..hd]);
                (head + codec2, tail)
            } else {
                let pad_count = n_codec - n_text;
                let padded = if pad_count > 0 {
                    let pads: Vec<Tensor<B, 3>> =
                        (0..pad_count).map(|_| tts_pad_embed.clone()).collect();
                    Tensor::cat(
                        std::iter::once(text_embed).chain(pads).collect::<Vec<_>>(),
                        1,
                    )
                } else {
                    text_embed
                };
                (padded + codec2, tts_pad_embed.clone())
            };

            let icl_len = icl_embed.dims()[1];
            let hh = self
                .talker
                .run_cached(icl_embed, &mut cache, offset, true, 0, dev);
            offset += icl_len;
            boost_boundary = offset;
            last_hidden = hh.slice([0..1, icl_len - 1..icl_len, 0..hd]);
            logits = self.talker.head_logits(last_hidden.clone())?;
            trailing
        } else {
            let mut rest: Vec<u32> = text_ids[1..].to_vec();
            rest.push(TTS_EOS);
            self.talker.embed_text(&rest, dev)?
        };
        let trailing_len = trailing.dims()[1];
        // Short text sits entirely in the prefill; boosting it makes the model
        // repeat the word, so only boost when there is real trailing text.
        if trailing_len <= 1 || !attn_boost {
            boost_boundary = 0;
        }
        Ok(Prefill {
            cache,
            offset,
            last_hidden,
            logits,
            trailing,
            trailing_len,
            boost_boundary,
            max_frames,
            hd,
            tts_pad_embed,
        })
    }

    /// The talker's logits at every step along a given frame sequence: after
    /// the prefill, then after each frame of `frames` is fed as if generated
    /// (its semantic token and fifteen acoustic codes, plus the next text
    /// token), so the model's opinion can be compared with another
    /// implementation's on the same history. Returns `frames.len() + 1`
    /// vectors; the code predictor is not run.
    pub fn forced_logits(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        frames: &[Vec<u32>],
        dev: &B::Device,
    ) -> Result<Vec<Vec<f32>>, TtsError> {
        if text_ids.is_empty() {
            return Err(TtsError::EmptyText);
        }
        let Prefill {
            mut cache,
            mut offset,
            logits,
            trailing,
            trailing_len,
            max_frames,
            hd,
            tts_pad_embed,
            ..
        } = self.prefill(text_ids, prompt, frames.len().max(1), false, dev)?;
        let mut out = vec![logits];
        for (frame_idx, frame) in frames.iter().enumerate().take(max_frames) {
            let (tok, codes) = frame
                .split_first()
                .ok_or_else(|| TtsError::InvalidFrames(format!("frame {frame_idx} is empty")))?;
            let mut summed = self.talker.embed_codec_ids(&[*tok], dev)?;
            for (g, &c) in codes.iter().enumerate() {
                summed = summed + self.cp.embed_group(g, c, dev);
            }
            let text_add = if frame_idx < trailing_len {
                trailing
                    .clone()
                    .slice([0..1, frame_idx..frame_idx + 1, 0..hd])
            } else {
                tts_pad_embed.clone()
            };
            let hh = self
                .talker
                .run_cached(summed + text_add, &mut cache, offset, false, 0, dev);
            offset += 1;
            out.push(self.talker.head_logits(hh)?);
        }
        Ok(out)
    }
}
