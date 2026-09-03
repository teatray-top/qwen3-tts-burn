use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::Int;

use crate::error::TtsError;
use crate::talker::{expect_layer_count, rmsnorm, rope, KvCache, Layer};
use crate::weights::WeightFile;

#[derive(Clone, Copy)]
pub struct CodePredictorConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub vocab: usize,
    pub groups: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
}

pub const CODE_PRED_1_7B: CodePredictorConfig = CodePredictorConfig {
    layers: 5,
    hidden: 1024,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    embed_dim: 2048,
    vocab: 2048,
    groups: 16,
    rope_theta: 1_000_000.0,
    rms_eps: 1e-6,
};

pub struct CodePredictor<B: Backend> {
    pub cfg: CodePredictorConfig,
    embeds: Vec<Tensor<B, 2>>,
    proj_w: Tensor<B, 2>,
    proj_b: Tensor<B, 1>,
    layers: Vec<Layer<B>>,
    norm: Tensor<B, 1>,
    heads: Vec<Tensor<B, 2>>,
    cos_table: Tensor<B, 2>,
    sin_table: Tensor<B, 2>,
}

impl<B: Backend> CodePredictor<B> {
    pub fn load(
        wf: &WeightFile,
        cfg: CodePredictorConfig,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        if cfg.groups < 2
            || cfg.kv_heads == 0
            || !cfg.heads.is_multiple_of(cfg.kv_heads)
            || !cfg.head_dim.is_multiple_of(2)
        {
            return Err(TtsError::InvalidConfig(format!(
                "code predictor groups {} / heads {} / kv_heads {} / head_dim {} are inconsistent",
                cfg.groups, cfg.heads, cfg.kv_heads, cfg.head_dim
            )));
        }
        expect_layer_count(wf, "talker.code_predictor.model.layers", cfg.layers)?;
        let n_ac = cfg.groups - 1;
        let shape = |name: &str, expected: String, got: &[usize]| TtsError::TensorShape {
            name: name.into(),
            expected,
            got: got.to_vec(),
        };
        let inv_freq: Vec<f64> = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64))
            .collect();
        let max_pos = 32usize;
        let half = cfg.head_dim / 2;
        let mut cos_v = Vec::with_capacity(max_pos * half);
        let mut sin_v = Vec::with_capacity(max_pos * half);
        for ppos in 0..max_pos {
            for f in &inv_freq {
                let a = ppos as f64 * f;
                cos_v.push(a.cos() as f32);
                sin_v.push(a.sin() as f32);
            }
        }
        let cos_table =
            Tensor::from_data(burn::tensor::TensorData::new(cos_v, [max_pos, half]), dev);
        let sin_table =
            Tensor::from_data(burn::tensor::TensorData::new(sin_v, [max_pos, half]), dev);
        let mut embeds = Vec::with_capacity(n_ac);
        for i in 0..n_ac {
            let name = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
            let e = wf.tensor2(&name, dev)?;
            if e.dims() != [cfg.vocab, cfg.embed_dim] {
                return Err(shape(
                    &name,
                    format!("[{}, {}]", cfg.vocab, cfg.embed_dim),
                    &e.dims(),
                ));
            }
            embeds.push(e);
        }
        let proj_w = wf.linear_t("talker.code_predictor.small_to_mtp_projection.weight", dev)?;
        if proj_w.dims() != [cfg.embed_dim, cfg.hidden] {
            return Err(shape(
                "talker.code_predictor.small_to_mtp_projection.weight",
                format!("[{}, {}]", cfg.hidden, cfg.embed_dim),
                &[proj_w.dims()[1], proj_w.dims()[0]],
            ));
        }
        let proj_b = wf.tensor1("talker.code_predictor.small_to_mtp_projection.bias", dev)?;
        if proj_b.dims()[0] != cfg.hidden {
            return Err(shape(
                "talker.code_predictor.small_to_mtp_projection.bias",
                format!("[{}]", cfg.hidden),
                &proj_b.dims(),
            ));
        }
        let layers = (0..cfg.layers)
            .map(|i| {
                Layer::load(
                    wf,
                    &format!("talker.code_predictor.model.layers.{i}"),
                    cfg.hidden,
                    cfg.heads,
                    cfg.kv_heads,
                    cfg.head_dim,
                    dev,
                )
            })
            .collect::<Result<_, _>>()?;
        let norm = wf.tensor1("talker.code_predictor.model.norm.weight", dev)?;
        if norm.dims()[0] != cfg.hidden {
            return Err(shape(
                "talker.code_predictor.model.norm.weight",
                format!("[{}]", cfg.hidden),
                &norm.dims(),
            ));
        }
        let mut heads = Vec::with_capacity(n_ac);
        for i in 0..n_ac {
            let name = format!("talker.code_predictor.lm_head.{i}.weight");
            let h = wf.linear_t(&name, dev)?;
            if h.dims() != [cfg.hidden, cfg.vocab] {
                return Err(shape(
                    &name,
                    format!("[{}, {}]", cfg.vocab, cfg.hidden),
                    &[h.dims()[1], h.dims()[0]],
                ));
            }
            heads.push(h);
        }
        Ok(Self {
            cfg,
            embeds,
            proj_w,
            proj_b,
            layers,
            norm,
            heads,
            cos_table,
            sin_table,
        })
    }

    fn run_layers(
        &self,
        x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        offset: usize,
        causal: bool,
        dev: &B::Device,
    ) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let [_, s, _] = x.dims();
        let group = cfg.heads / cfg.kv_heads;
        let scale = (cfg.head_dim as f64).powf(-0.5);
        let _ = (causal, dev);
        let half = cfg.head_dim / 2;
        let cos = self.cos_table.clone().slice([offset..offset + s, 0..half]);
        let sin = self.sin_table.clone().slice([offset..offset + s, 0..half]);

        let max = cache.max;
        let mask: Tensor<B, 4> = cache
            .mask_full
            .clone()
            .slice([offset..offset + s, 0..max])
            .reshape([1, 1, s, max]);

        let mut h = x;
        for (i, l) in self.layers.iter().enumerate() {
            let normed = rmsnorm(h.clone(), &l.ln_in, cfg.rms_eps);
            let nh = cfg.heads + 2 * cfg.kv_heads;
            let nqk = cfg.heads + cfg.kv_heads;
            let qkv = normed
                .matmul(l.wqkv.clone().unsqueeze())
                .reshape([1, s, nh, cfg.head_dim])
                .swap_dims(1, 2);
            let qk = qkv.clone().slice([0..1, 0..nqk, 0..s, 0..cfg.head_dim]);
            let v = qkv.slice([0..1, nqk..nh, 0..s, 0..cfg.head_dim]);

            // Single fused qk-norm: rms-normalize q and k together, then scale
            // by a per-head weight table (q_norm x heads rows, k_norm x kv rows).
            let sq = qk.clone().powf_scalar(2.0).mean_dim(3);
            let qk = qk / (sq + cfg.rms_eps).sqrt();
            let qk = qk * l.qk_table.clone().reshape([1, nqk, 1, cfg.head_dim]);
            let qk = rope(qk, &cos, &sin);
            let q = qk
                .clone()
                .slice([0..1, 0..cfg.heads, 0..s, 0..cfg.head_dim]);
            let k = qk.slice([0..1, cfg.heads..nqk, 0..s, 0..cfg.head_dim]);

            cache.push(i, k, v, offset);
            let ck = cache.slots[i].k.clone();
            let cv = cache.slots[i].v.clone();

            let kx = ck
                .unsqueeze_dim::<5>(2)
                .expand([1, cfg.kv_heads, group, max, cfg.head_dim])
                .reshape([1, cfg.heads, max, cfg.head_dim]);
            let vx = cv
                .unsqueeze_dim::<5>(2)
                .expand([1, cfg.kv_heads, group, max, cfg.head_dim])
                .reshape([1, cfg.heads, max, cfg.head_dim]);

            let att = q.matmul(kx.swap_dims(2, 3)) * scale + mask.clone();
            let att = softmax(att, 3);
            let out = att
                .matmul(vx)
                .swap_dims(1, 2)
                .reshape([1, s, cfg.heads * cfg.head_dim]);
            h = h + out.matmul(l.wo.clone().unsqueeze());

            let normed2 = rmsnorm(h.clone(), &l.ln_post, cfg.rms_eps);
            let gu = normed2.matmul(l.w_gate_up.clone().unsqueeze());
            let [_, _, gud] = gu.dims();
            let gate = silu(gu.clone().slice([0..1, 0..s, 0..gud / 2]));
            let up = gu.slice([0..1, 0..s, gud / 2..gud]);
            h = h + (gate * up).matmul(l.w_down.clone().unsqueeze());
        }
        rmsnorm(h, &self.norm, cfg.rms_eps)
    }

    fn project(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.matmul(self.proj_w.clone().unsqueeze()) + self.proj_b.clone().unsqueeze::<3>()
    }

    pub fn embed_group_batch(&self, group: usize, idx: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        let e = self.embeds[group].clone().select(0, idx);
        let [s, d] = e.dims();
        e.reshape([1, s, d])
    }

    pub fn embed_group(&self, group: usize, code: u32, dev: &B::Device) -> Tensor<B, 3> {
        let idx: Tensor<B, 1, Int> =
            Tensor::from_data(burn::tensor::TensorData::new(vec![code as i32], [1]), dev);
        let e = self.embeds[group].clone().select(0, idx);
        let [s, d] = e.dims();
        e.reshape([1, s, d])
    }

    /// Greedy 15-code generation, mirroring candle's `generate_acoustic_codes`:
    /// prefill [talker_hidden, semantic_embed] (projected 2048→1024), then 14
    /// cached single-token steps, each embedding the previous group's code.
    pub fn new_cache(&self, dev: &B::Device) -> KvCache<B> {
        KvCache::new(
            self.cfg.layers,
            self.cfg.kv_heads,
            17,
            self.cfg.head_dim,
            dev,
        )
    }

    /// `cache` is reused across frames: every position is rewritten before it
    /// becomes visible (the mask only exposes j <= offset+i), so no reset is
    /// needed and per-frame allocations are avoided. With `cp_sample` off the
    /// codes stay on the device as argmax tensors and are read back once per
    /// frame; sampling needs each group's logits on the host, one readback per
    /// group, which the official implementation also pays.
    pub fn generate_codes(
        &self,
        talker_hidden: Tensor<B, 3>,
        semantic_embed: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        sampler: &mut crate::sampling::Sampler,
        dev: &B::Device,
    ) -> Result<Vec<u32>, TtsError> {
        let n_ac = self.cfg.groups - 1;

        let input = Tensor::cat(vec![talker_hidden, semantic_embed], 1);
        let input = self.project(input);
        let hidden = self.run_layers(input, cache, 0, true, dev);
        let [_, s, hd] = hidden.dims();
        let last = hidden.slice([0..1, s - 1..s, 0..hd]);
        let first_logits = last.matmul(self.heads[0].clone().unsqueeze());

        if !sampler.cfg().cp_sample {
            let mut code_t: Tensor<B, 1, Int> = first_logits.argmax(2).reshape([1]);
            let mut codes_t: Vec<Tensor<B, 1, Int>> = vec![code_t.clone()];
            for (g, offset) in (1..n_ac).zip(s..) {
                let emb = self.embeds[g - 1].clone().select(0, code_t.clone());
                let [_, ed] = emb.dims();
                let emb = self.project(emb.reshape([1, 1, ed]));
                let h = self.run_layers(emb, cache, offset, false, dev);
                let logits = h.matmul(self.heads[g].clone().unsqueeze());
                code_t = logits.argmax(2).reshape([1]);
                codes_t.push(code_t.clone());
            }
            let all = Tensor::cat(codes_t, 0);
            let v: Vec<i32> = all
                .into_data()
                .to_vec()
                .map_err(|e| TtsError::Numeric(format!("code predictor readback: {e:?}")))?;
            return Ok(v.into_iter().map(|c| c as u32).collect());
        }

        let read = |logits: Tensor<B, 3>| -> Result<Vec<f32>, TtsError> {
            logits
                .into_data()
                .convert::<f32>()
                .to_vec()
                .map_err(|e| TtsError::Numeric(format!("code predictor logits: {e:?}")))
        };
        let mut code = sampler.next_code(&read(first_logits)?)?;
        let mut codes: Vec<u32> = Vec::with_capacity(n_ac);
        codes.push(code);
        for (g, offset) in (1..n_ac).zip(s..) {
            let idx: Tensor<B, 1, Int> =
                Tensor::from_data(burn::tensor::TensorData::new(vec![code as i32], [1]), dev);
            let emb = self.embeds[g - 1].clone().select(0, idx);
            let [_, ed] = emb.dims();
            let emb = self.project(emb.reshape([1, 1, ed]));
            let h = self.run_layers(emb, cache, offset, false, dev);
            code = sampler.next_code(&read(h.matmul(self.heads[g].clone().unsqueeze()))?)?;
            codes.push(code);
        }
        Ok(codes)
    }
}
