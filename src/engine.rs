use std::path::Path;

use burn::prelude::*;

use crate::code_predictor::{CodePredictor, CODE_PRED_1_7B};
use crate::decoder::Decoder12Hz;
use crate::encoder::Encoder12Hz;
use crate::error::TtsError;
use crate::lowpass::ButterworthLp;
use crate::pipeline::{ClonePrompt, Pipeline};
use crate::sampling::SamplerCfg;
use crate::speaker::SpeakerEncoder;
use crate::talker::{Talker, TALKER_1_7B};
use crate::tokenizer::TextTokenizer;

/// Codes per group in the 12 Hz codec; a frame holds one code per group.
const CODEC_GROUP_SIZE: u32 = 2048;

const CLONE_LPF_HZ: f64 = 10500.0;
const CLONE_LPF_ORDER: usize = 6;
/// Max sibilant-band reduction (dB) for the streaming de-esser.
const DEESS_MAX_DB: f32 = 12.0;

const MS_PER_FRAME: f64 = 80.0;
const DECODE_EVERY_FRAMES: usize = 24;
/// Frames held back for the final block so its end-fade owns the utterance end.
const TAIL_HOLD_FRAMES: usize = 8;
/// Trailing-silence cutoff for the flaky-EOS early stop: RMS below SILENCE_RMS
/// over SILENCE_WIN-sample windows, sustained for SILENCE_STOP_MS once the text
/// is spent, means the model is only padding and generation can end.
const SILENCE_RMS: f32 = 0.006;
const SILENCE_WIN: usize = 480; // 20 ms at 24 kHz
const SILENCE_STOP_MS: usize = 700;
const LEAD_FLOOR_MS: f64 = 300.0;
/// Ceiling on the pre-roll buffer (bounds first-audio wait on long passages).
const LEAD_CAP_MS: f64 = 6000.0;
const MS_AUDIO_PER_CHAR: f64 = 160.0;
const LEAD_SAFETY: f64 = 1.2;
/// Lead for continuation chunks, whose previous tail is still queued.
const CONTINUATION_LEAD_MS: f64 = 120.0;

/// Speech (ms) the onset must buffer so realtime playback can't overtake
/// sub-realtime generation: buffer `B = A·(r−1)/r` for total audio `A`, rate `r`.
fn required_lead_ms(char_count: usize, ms_per_frame: f64) -> f64 {
    let decode_amortized = 300.0 / DECODE_EVERY_FRAMES as f64;
    let r = (ms_per_frame + decode_amortized) / MS_PER_FRAME;
    if r <= 1.0 {
        return LEAD_FLOOR_MS;
    }
    let total_est = char_count as f64 * MS_AUDIO_PER_CHAR + 300.0;
    let buffer = total_est * (r - 1.0) / r * LEAD_SAFETY + TAIL_HOLD_FRAMES as f64 * MS_PER_FRAME;
    buffer.clamp(LEAD_FLOOR_MS, LEAD_CAP_MS)
}

/// A fully-loaded Vulkan Qwen3-TTS 1.7B voice-clone engine (all components resident).
/// Everything applied around the model: to the text before it, to the
/// generation loop, and to the decoder's output.
///
/// None of it is part of Qwen3-TTS. It exists because `f16` weights make the
/// end-of-speech token land early, and because the relay app this grew out of
/// wanted a particular sound. `PostProcess::none()` is the model as published;
/// `PostProcess::app_default()` is the relay app's set, each switch documented
/// with what it costs.
#[derive(Clone, Copy, Debug)]
pub struct PostProcess {
    /// Replace sentence-final punctuation with a comma before tokenising, see
    /// [`damp_ending`]. Keeps the last syllable from being cut, at the price of
    /// the question or exclamation intonation the punctuation would have asked
    /// for.
    pub damp_ending: bool,
    /// Mask end-of-speech until every text token has been fed plus a short
    /// grace, then let a few frames release. Guards against early stops; can
    /// add a beat of trailing sound that `trailing_trim` then removes.
    pub hold_eos: bool,
    /// Bias attention toward the in-context reference span while generating
    /// (the app's "expression" slider). Off reproduces the official attention.
    pub attn_boost: bool,
    /// Cut silence before the first speech. Only meaningful with an ICL prompt,
    /// where the model tends to open with a pause.
    pub leading_trim: bool,
    /// Cut trailing silence, which f16 EOS instability leaves behind.
    pub trailing_trim: bool,
    /// Low-pass in Hz. `None` leaves the full band.
    pub lowpass_hz: Option<f64>,
    /// De-esser ceiling in dB. `None` disables it.
    pub deess_max_db: Option<f32>,
}

impl PostProcess {
    /// The model as published: text, generation and audio untouched.
    pub fn none() -> Self {
        Self {
            damp_ending: false,
            hold_eos: false,
            attn_boost: false,
            leading_trim: false,
            trailing_trim: false,
            lowpass_hz: None,
            deess_max_db: None,
        }
    }

    /// What the relay app uses: comma ending, held EOS, attention boost, trims,
    /// a 10.5 kHz low-pass and a 12 dB de-esser.
    pub fn app_default() -> Self {
        Self {
            damp_ending: true,
            hold_eos: true,
            attn_boost: true,
            leading_trim: true,
            trailing_trim: true,
            lowpass_hz: Some(CLONE_LPF_HZ),
            deess_max_db: Some(DEESS_MAX_DB),
        }
    }
}

impl Default for PostProcess {
    fn default() -> Self {
        Self::none()
    }
}

/// Swap sentence-final punctuation for a comma.
///
/// A full stop or question mark is the model's cue to stop, and with `f16`
/// weights it tends to stop during the last phoneme rather than after it —
/// "들리나요?" came out as "들리나ㅇ". A comma reads as "more follows", so the
/// syllable is articulated in full and the model then goes quiet;
/// `PostProcess::trailing_trim` removes that silence. The trade is the
/// intonation the original punctuation asked for. Applied only when
/// `PostProcess::damp_ending` is set; `synthesize` never rewrites text.
fn validate_post(post: &PostProcess) -> Result<(), TtsError> {
    if let Some(hz) = post.lowpass_hz {
        if !(hz > 0.0 && hz < 12000.0) {
            return Err(TtsError::InvalidConfig(format!(
                "lowpass_hz {hz} must lie in (0, 12000) at 24 kHz"
            )));
        }
    }
    Ok(())
}

pub fn damp_ending(text: &str) -> String {
    let body = text
        .trim()
        .trim_end_matches(['.', '?', '!', '。', '？', '！', '…', ' ']);
    if body.is_empty() {
        return text.to_string();
    }
    format!("{body},")
}

pub struct Engine<B: Backend> {
    pipe: Pipeline<B>,
    // The speaker encoder and speech tokenizer encoder are only needed while a
    // prompt is being built. They are loaded from these files on demand and
    // dropped again, rather than held for the life of the engine.
    main_weights: crate::weights::WeightFile,
    codec_weights: crate::weights::WeightFile,
    tokenizer: TextTokenizer,
    dev: B::Device,
}

/// Frames a reference clip may contribute to the in-context prefill. The
/// talker's rotary table has 2048 positions and the prefill also carries the
/// role header, the text and the codec prefix.
pub const MAX_REFERENCE_FRAMES: usize = 2000;

impl<B: Backend> Engine<B> {
    /// The files `load` needs, checked before anything is read or uploaded.
    pub fn check_model_dir(model_dir: &str) -> Result<(), TtsError> {
        let dir = Path::new(model_dir);
        for rel in [
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "speech_tokenizer/model.safetensors",
        ] {
            let p = dir.join(rel);
            if !p.is_file() {
                return Err(TtsError::ModelFileMissing { path: p });
            }
        }
        Ok(())
    }

    pub fn load(model_dir: &str, dev: B::Device) -> Result<Self, TtsError> {
        Self::check_model_dir(model_dir)?;
        let dir = Path::new(model_dir);
        // Cheapest failure first: a broken tokenizer should not be reported
        // after the weights have been mapped and uploaded.
        let tokenizer = TextTokenizer::from_dir(dir)?;
        let wf = crate::weights::WeightFile::open(dir.join("model.safetensors"))?;
        let stf = crate::weights::WeightFile::open(
            dir.join("speech_tokenizer").join("model.safetensors"),
        )?;
        let engine = Self {
            pipe: Pipeline {
                talker: Talker::load(&wf, TALKER_1_7B, &dev)?,
                cp: CodePredictor::load(&wf, CODE_PRED_1_7B, &dev)?,
                decoder: Decoder12Hz::load(&stf, &dev)?,
            },
            main_weights: wf,
            codec_weights: stf,
            tokenizer,
            dev,
        };
        engine.trim_memory();
        Ok(engine)
    }

    /// Hand every free pool page back to the driver. Pages the allocator kept
    /// after a large transient (prompt encoders, the decoder's widest bucket)
    /// otherwise stay resident for the life of the process.
    pub fn trim_memory(&self) {
        let _ = B::sync(&self.dev);
        B::memory_cleanup(&self.dev);
    }

    /// Compiles and autotunes the kernels the model uses so the first real
    /// utterance is not the one that pays for it. With empty caches this takes
    /// about 2.7 minutes on an RTX 5070 Ti (the first prompt and first
    /// sentence add another 40 s of tuning); once the kernel and autotune
    /// caches exist it takes under a second.
    pub fn warmup(&self) -> Result<(), TtsError> {
        let prompt = ClonePrompt {
            language: crate::lang::Language::default(),
            speaker_embedding: vec![0.0; 2048],
            ref_codes: Vec::new(),
            ref_text_ids: Vec::new(),
        };
        let ids = self.tokenizer.encode("가")?;
        self.pipe
            .generate(&ids, &prompt, SamplerCfg::app(), 4, &self.dev)?;
        // Precompile the decode buckets a sentence can reach. The one-shot
        // decoder pads to the next power of two, and nothing in the app ever
        // one-shot-decodes more than ~90 frames (the streaming onset bound), so
        // the 512-frame bucket is left cold: it was the largest transient in
        // warmup and its pages stayed resident afterwards.
        let dummy: Vec<Vec<u32>> = vec![vec![0u32; 16]; 260];
        for n in [8usize, 40, 100] {
            self.pipe.decoder.decode(&dummy[..n], &self.dev)?;
        }
        // Warm the streaming decode_window upsampling buckets too.
        for (n, keep) in [(64usize, 32), (130, 96), (260, 216)] {
            self.pipe
                .decoder
                .decode_window(&dummy[..n], keep, &self.dev)?;
        }
        self.trim_memory();
        Ok(())
    }

    /// Build a hybrid ICL clone prompt: x-vector from `emb_wav`, ICL from `icl_wav`.
    ///
    /// `language` sets the codec prefix token and must match the language of
    /// the text that will be synthesized, not necessarily the reference clip.
    ///
    /// Both clips are trimmed to their speech before use — see
    /// [`crate::postproc::speech_bounds`] for why leading and trailing silence
    /// in a reference is destructive rather than merely wasteful.
    pub fn build_clone_prompt(
        &self,
        emb_wav: &str,
        icl_wav: &str,
        ref_text: &str,
        language: crate::lang::Language,
    ) -> Result<ClonePrompt, TtsError> {
        if ref_text.trim().is_empty() {
            return Err(TtsError::EmptyReferenceText);
        }
        let speaker_embedding = self.speaker_embedding(emb_wav)?;

        let (icl_raw, isr) = crate::audio::load_wav(icl_wav)?;
        let icl =
            crate::postproc::trim_silence(&crate::resample::resample(&icl_raw, isr, 24000)?, 24000);
        let ref_codes = {
            let enc = Encoder12Hz::<B>::load(&self.codec_weights, &self.dev)?;
            enc.encode(&icl, &self.dev)?
        };
        if ref_codes.len() > MAX_REFERENCE_FRAMES {
            return Err(TtsError::ReferenceTooLong {
                frames: ref_codes.len(),
                max: MAX_REFERENCE_FRAMES,
            });
        }
        let ref_text_ids = self.tokenizer.encode(ref_text)?;
        self.trim_memory();
        Ok(ClonePrompt {
            language,
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    fn speaker_embedding(&self, wav: &str) -> Result<Vec<f32>, TtsError> {
        let (raw, sr) = crate::audio::load_wav(wav)?;
        let emb =
            crate::postproc::trim_silence(&crate::resample::resample(&raw, sr, 24000)?, 24000);
        let mel = crate::mel::log_mel(&emb)?;
        let ecapa = SpeakerEncoder::<B>::load(&self.main_weights, &self.dev)?;
        ecapa.encode(&mel, &self.dev)
    }

    /// x-vector-only prompt (no ICL): timbre from `emb_wav`, model's own prosody.
    pub fn build_xvector_prompt(
        &self,
        emb_wav: &str,
        language: crate::lang::Language,
    ) -> Result<ClonePrompt, TtsError> {
        let speaker_embedding = self.speaker_embedding(emb_wav)?;
        self.trim_memory();
        Ok(ClonePrompt {
            language,
            speaker_embedding,
            ref_codes: Vec::new(),
            ref_text_ids: Vec::new(),
        })
    }

    /// Tokenize text to input ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        self.tokenizer.encode(text)
    }

    /// Frames the talker and code predictor can consume: sixteen codes each,
    /// every code inside the codec's 2048-entry group.
    fn validate_frames(&self, frames: &[Vec<u32>], what: &str) -> Result<(), TtsError> {
        let groups = self.pipe.decoder.groups();
        for (i, f) in frames.iter().enumerate() {
            if f.len() != groups {
                return Err(TtsError::InvalidFrames(format!(
                    "{what} {i} has {} codes, the codec takes {groups}",
                    f.len()
                )));
            }
            if let Some(c) = f.iter().find(|&&c| c >= CODEC_GROUP_SIZE) {
                return Err(TtsError::InvalidFrames(format!(
                    "{what} {i} holds code {c}, outside the codec's 0..{CODEC_GROUP_SIZE} range"
                )));
            }
        }
        Ok(())
    }

    fn validate_prompt(&self, prompt: &ClonePrompt) -> Result<(), TtsError> {
        if prompt.speaker_embedding.len() != 2048 {
            return Err(TtsError::InvalidPrompt(format!(
                "speaker embedding has {} values, the talker takes 2048",
                prompt.speaker_embedding.len()
            )));
        }
        self.validate_frames(&prompt.ref_codes, "reference frame")
            .map_err(|e| TtsError::InvalidPrompt(e.to_string()))?;
        if prompt.ref_codes.len() > MAX_REFERENCE_FRAMES {
            return Err(TtsError::ReferenceTooLong {
                frames: prompt.ref_codes.len(),
                max: MAX_REFERENCE_FRAMES,
            });
        }
        let rows = self.pipe.talker.text_vocab_rows() as u32;
        if prompt.ref_text_ids.iter().any(|&t| t >= rows) {
            return Err(TtsError::InvalidPrompt(
                "reference text id outside the vocabulary".into(),
            ));
        }
        Ok(())
    }

    fn validate_text_ids(&self, ids: &[u32]) -> Result<(), TtsError> {
        if ids.is_empty() {
            return Err(TtsError::EmptyText);
        }
        let rows = self.pipe.talker.text_vocab_rows() as u32;
        if ids.iter().any(|&t| t >= rows) {
            return Err(TtsError::InvalidPrompt(
                "text id outside the vocabulary".into(),
            ));
        }
        Ok(())
    }

    /// Generate frames (16 codes each) without decoding, the way the official
    /// implementation does: no text rewrite, no attention bias, stop at the
    /// first end-of-speech token.
    pub fn generate_frames(
        &self,
        text_ids: &[u32],
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        self.validate_text_ids(text_ids)?;
        self.validate_prompt(prompt)?;
        self.pipe
            .generate(text_ids, prompt, cfg, max_frames, &self.dev)
    }

    /// Decode frames to samples.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, TtsError> {
        self.pipe.decoder.decode(frames, &self.dev)
    }

    /// Stream-decode the tail [keep_from..] of the prefix.
    pub fn decode_window(
        &self,
        frames: &[Vec<u32>],
        keep_from: usize,
    ) -> Result<Vec<f32>, TtsError> {
        self.pipe
            .decoder
            .decode_window(frames, keep_from, &self.dev)
    }

    /// Synthesize `text` as the model was published: the text as given, the
    /// official stopping rule, the decoder's output untouched. 24 kHz mono.
    pub fn synthesize(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
    ) -> Result<Vec<f32>, TtsError> {
        self.synthesize_with(text, prompt, cfg, max_frames, PostProcess::none())
    }

    /// Synthesize `text` with the switches in `post` applied around the model.
    pub fn synthesize_with(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
        post: PostProcess,
    ) -> Result<Vec<f32>, TtsError> {
        let frames = self.synthesize_frames(text, prompt, cfg, max_frames, post)?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let raw = self.decode_after_reference(prompt, &frames)?;
        let wav = self.apply_post(raw, prompt, post)?;
        self.trim_memory();
        Ok(wav)
    }

    fn apply_post(
        &self,
        raw: Vec<f32>,
        prompt: &ClonePrompt,
        post: PostProcess,
    ) -> Result<Vec<f32>, TtsError> {
        let sr = 24000u32;
        validate_post(&post)?;
        // Leading silence is an ICL artefact; without a reference prompt there
        // is nothing to trim.
        let start = if post.leading_trim && !prompt.ref_codes.is_empty() {
            crate::postproc::leading_trim(&raw, sr).min(raw.len())
        } else {
            0
        };
        let end = if post.trailing_trim {
            crate::postproc::trailing_trim(&raw, sr).max(start)
        } else {
            raw.len()
        };
        let mut wav = raw[start..end].to_vec();
        if let Some(hz) = post.lowpass_hz {
            ButterworthLp::new(hz, sr as f64, CLONE_LPF_ORDER).process_buffer(&mut wav);
        }
        if let Some(db) = post.deess_max_db {
            crate::deesser::Deesser::new(sr as f64, db).process_buffer(&mut wav);
        }
        Ok(wav)
    }

    /// One-shot app synthesis: decode → leading_trim → release_trim → LPF → de-ess.
    pub fn synthesize_speak(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
        post: PostProcess,
    ) -> Result<Vec<f32>, TtsError> {
        let text = if post.damp_ending {
            damp_ending(text)
        } else {
            text.to_string()
        };
        let ids = self.tokenizer.encode(&text)?;
        self.validate_text_ids(&ids)?;
        self.validate_prompt(prompt)?;
        validate_post(&post)?;
        let frames = self.pipe.generate_cb(
            &ids,
            prompt,
            cfg,
            max_frames,
            post.hold_eos,
            post.attn_boost,
            &self.dev,
            &mut |_| true,
        )?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let sr = 24000u32;
        let raw = self.decode_after_reference(prompt, &frames)?;
        let start = if post.leading_trim && !prompt.ref_codes.is_empty() {
            crate::postproc::leading_trim(&raw, sr)
        } else {
            0
        }
        .min(raw.len());
        let body = &raw[start..];
        // Drop the trailing silence a held EOS or a comma ending leaves behind.
        let end = if post.trailing_trim {
            crate::postproc::release_trim(body, sr)
        } else {
            body.len()
        };
        let mut wav = body[..end].to_vec();
        if let Some(hz) = post.lowpass_hz {
            ButterworthLp::new(hz, sr as f64, CLONE_LPF_ORDER).process_buffer(&mut wav);
        }
        if let Some(db) = post.deess_max_db {
            crate::deesser::Deesser::new(sr as f64, db).process_buffer(&mut wav);
        }
        // De-click: the trim above cuts mid-waveform, so ramp the last 10 ms to
        // zero. Untrimmed audio ends where the model ended and needs no ramp.
        if post.trailing_trim && !wav.is_empty() {
            let fade = (sr as usize * 10 / 1000).min(wav.len());
            let n = wav.len();
            for (i, s) in wav[n - fade..].iter_mut().enumerate() {
                *s *= 1.0 - (i as f32 + 1.0) / fade as f32;
            }
        }
        Ok(wav)
    }

    /// Streaming synthesis: emits audio chunks via `on_chunk` after a lead builds.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_streaming(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
        full_lead: bool,
        post: PostProcess,
        mut on_chunk: impl FnMut(&[f32]),
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        let text = if post.damp_ending {
            damp_ending(text)
        } else {
            text.to_string()
        };
        let ids = self.tokenizer.encode(&text)?;
        self.validate_text_ids(&ids)?;
        self.validate_prompt(prompt)?;
        validate_post(&post)?;
        let trim_onset = post.leading_trim && !prompt.ref_codes.is_empty();
        // The frame callback returns a bool, so a decode failure inside it is
        // parked here and raised once generation has stopped.
        let mut cb_err: Option<TtsError> = None;
        let sr = 24000u32;
        let chars = text.chars().count();

        let mut lpf = post
            .lowpass_hz
            .map(|hz| ButterworthLp::new(hz, sr as f64, CLONE_LPF_ORDER));
        let mut deesser = post
            .deess_max_db
            .map(|db| crate::deesser::Deesser::new(sr as f64, db));
        let mut filter = move |chunk: &mut Vec<f32>| {
            if let Some(f) = lpf.as_mut() {
                f.process_buffer(chunk);
            }
            if let Some(d) = deesser.as_mut() {
                d.process_buffer(chunk);
            }
        };
        let mut emitted_frame = 0usize;
        let mut onset_done = false;

        let dev = &self.dev;
        let decoder = &self.pipe.decoder;
        // The decoder is causal and needs real speech before the first generated
        // frame; the reference supplies it, exactly as the official implementation
        // does (qwen_tts/inference/qwen3_tts_model.py: cat(ref_code, codes), then
        // cut ref_len). Everything below decodes ref ++ frames and keeps only the
        // generated span.
        let ref_len = prompt.ref_codes.len();
        let with_ref = |gen: &[Vec<u32>]| -> Vec<Vec<u32>> {
            let mut all = Vec::with_capacity(ref_len + gen.len());
            all.extend_from_slice(&prompt.ref_codes);
            all.extend_from_slice(gen);
            all
        };

        // f16 makes EOS flaky: the model often fails to stop and keeps emitting
        // near-silence to max_frames (observed: a 1-char message producing 12.9 s
        // and holding the play queue that long). Once the text is spent, any
        // sustained silence is trailing silence — not an inter-clause pause — so
        // it is safe to cut there. `text_frames` is the point past which no input
        // token remains to be spoken.
        let text_frames = ids.len();
        let mut silent_run: usize = 0;
        // Generation rate is measured from the first frame onward. Counting
        // from t0 folds the prefill into the per-frame figure, which at the
        // onset made the rate look several times slower than realtime and
        // the lead buffer grow to the whole utterance (an English line of
        // 3.4 s waited 6.5 s for its first chunk).
        let mut t_first: Option<std::time::Instant> = None;
        let frames = {
            let mut cb = |frames: &[Vec<u32>]| -> bool {
                let n = frames.len();
                let t_first = *t_first.get_or_insert_with(std::time::Instant::now);
                // Hold the last TAIL_HOLD_FRAMES for the final block's end-fade.
                let emit_n = n.saturating_sub(TAIL_HOLD_FRAMES);
                if emit_n <= emitted_frame {
                    return true;
                }
                let ms_per_frame = if n > 1 {
                    (t_first.elapsed().as_secs_f64() * 1000.0 / (n - 1) as f64).max(1.0)
                } else {
                    MS_PER_FRAME
                };
                if !onset_done {
                    // Onset: full-decode the prefix for the lead check, emit up to emit_n.
                    let full = match decoder.decode_window(&with_ref(&frames[0..n]), ref_len, dev) {
                        Ok(v) => v,
                        Err(e) => {
                            cb_err = Some(e);
                            return false;
                        }
                    };
                    let onset = if trim_onset {
                        crate::postproc::leading_trim(&full, sr)
                    } else {
                        0
                    };
                    let speech_ms = full.len().saturating_sub(onset) as f64 / sr as f64 * 1000.0;
                    let target = if full_lead {
                        required_lead_ms(chars, ms_per_frame)
                    } else {
                        CONTINUATION_LEAD_MS
                    };
                    if speech_ms < target {
                        return true;
                    }
                    onset_done = true;
                    let spf = (full.len() / n.max(1)).max(1);
                    let emit_to = (emit_n * spf).min(full.len());
                    if emit_to > onset {
                        let mut chunk = full[onset..emit_to].to_vec();
                        filter(&mut chunk);
                        on_chunk(&chunk);
                    }
                    emitted_frame = emit_n;
                } else {
                    if emit_n - emitted_frame < DECODE_EVERY_FRAMES {
                        return true;
                    }
                    // Windowed decode of the new tail (bit-exact to one-shot).
                    let mut chunk = match decoder.decode_window(
                        &with_ref(&frames[0..emit_n]),
                        ref_len + emitted_frame,
                        dev,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            cb_err = Some(e);
                            return false;
                        }
                    };
                    if !chunk.is_empty() {
                        filter(&mut chunk);
                        // Measure the chunk's TRAILING silence, not whether the
                        // whole chunk is silent: a chunk holds ~2 s, so a chunk of
                        // "speech then silence" would otherwise reset the run and
                        // force a whole extra chunk of padding before stopping.
                        let win = SILENCE_WIN.min(chunk.len().max(1));
                        let mut tail = 0usize;
                        for w in chunk.chunks(win).rev() {
                            let rms =
                                (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt();
                            if rms < SILENCE_RMS {
                                tail += w.len();
                            } else {
                                break;
                            }
                        }
                        silent_run = if tail == chunk.len() {
                            silent_run + tail
                        } else {
                            tail
                        };
                        on_chunk(&chunk);
                    }
                    emitted_frame = emit_n;
                }
                // Past the text with a long silent tail: stop instead of running
                // to max_frames and occupying the play queue with nothing. Only
                // with `hold_eos`, which is what makes a run overshoot; without
                // it the model's own end-of-speech token stops the loop.
                !(post.hold_eos
                    && n > text_frames + TAIL_HOLD_FRAMES
                    && silent_run > sr as usize * SILENCE_STOP_MS / 1000)
            };
            self.pipe.generate_cb(
                &ids,
                prompt,
                cfg,
                max_frames,
                post.hold_eos,
                post.attn_boost,
                dev,
                &mut cb,
            )?
        };
        if let Some(e) = cb_err {
            return Err(e);
        }

        if frames.is_empty() {
            return Ok(frames);
        }
        // Final tail: windowed decode of [emitted_frame..N], or full decode if onset never fired.
        let mut chunk = if onset_done {
            self.pipe
                .decoder
                .decode_window(&with_ref(&frames), ref_len + emitted_frame, dev)?
        } else {
            let full = self
                .pipe
                .decoder
                .decode_window(&with_ref(&frames), ref_len, dev)?;
            let s = if trim_onset {
                crate::postproc::leading_trim(&full, sr).min(full.len())
            } else {
                0
            };
            full[s..].to_vec()
        };
        if post.trailing_trim {
            let end = crate::postproc::trailing_trim(&chunk, sr).min(chunk.len());
            chunk.truncate(end);
        }
        if !chunk.is_empty() {
            filter(&mut chunk);
            // Fade the last 40 ms of actual speech so the trim's hard stop reads
            // as a release. Untrimmed audio is left as the model ended it.
            if post.trailing_trim {
                let win = (sr as usize * 20 / 1000).max(1);
                let mut speech_end = chunk.len();
                for w in (0..chunk.len() / win).rev() {
                    let s = &chunk[w * win..(w + 1) * win];
                    let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
                    if rms > 0.005 {
                        speech_end = ((w + 1) * win).min(chunk.len());
                        break;
                    }
                }
                let fade = (sr as usize * 40 / 1000).min(speech_end);
                for (i, s) in chunk[speech_end - fade..speech_end].iter_mut().enumerate() {
                    *s *= 1.0 - (i as f32 + 1.0) / fade as f32;
                }
            }
            on_chunk(&chunk);
        }
        self.trim_memory();
        Ok(frames)
    }

    /// The frames `synthesize_with` would decode, without decoding them. Only
    /// the generation-time switches of `post` apply (`damp_ending`, `hold_eos`,
    /// `attn_boost`).
    pub fn synthesize_frames(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        cfg: SamplerCfg,
        max_frames: usize,
        post: PostProcess,
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        if text.trim().is_empty() {
            return Err(TtsError::EmptyText);
        }
        let text = if post.damp_ending {
            damp_ending(text)
        } else {
            text.to_string()
        };
        let ids = self.tokenizer.encode(&text)?;
        self.validate_text_ids(&ids)?;
        self.validate_prompt(prompt)?;
        self.pipe.generate_cb(
            &ids,
            prompt,
            cfg,
            max_frames,
            post.hold_eos,
            post.attn_boost,
            &self.dev,
            &mut |_| true,
        )
    }

    /// The talker's logits along a given frame sequence, see
    /// [`crate::pipeline::Pipeline::forced_logits`]. The text is fed as given.
    pub fn forced_logits(
        &self,
        text: &str,
        prompt: &ClonePrompt,
        frames: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, TtsError> {
        if text.trim().is_empty() {
            return Err(TtsError::EmptyText);
        }
        let ids = self.tokenizer.encode(text)?;
        self.validate_text_ids(&ids)?;
        self.validate_prompt(prompt)?;
        self.validate_frames(frames, "frame")?;
        self.pipe.forced_logits(&ids, prompt, frames, &self.dev)
    }

    /// Decode generated frames with the reference clip's frames in front, so
    /// the causal decoder has real speech as context for the first frames
    /// instead of nothing; the reference span is not returned. This is what the
    /// official implementation does.
    pub fn decode_after_reference(
        &self,
        prompt: &ClonePrompt,
        frames: &[Vec<u32>],
    ) -> Result<Vec<f32>, TtsError> {
        if prompt.ref_codes.is_empty() || frames.is_empty() {
            return self.decode(frames);
        }
        let mut all = prompt.ref_codes.clone();
        all.extend_from_slice(frames);
        self.decode_window(&all, prompt.ref_codes.len())
    }
}
