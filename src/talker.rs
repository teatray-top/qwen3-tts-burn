use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::Int;

use crate::error::TtsError;
use crate::weights::WeightFile;

#[derive(Clone, Copy)]
pub struct TalkerConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
}

pub const TALKER_1_7B: TalkerConfig = TalkerConfig {
    layers: 28,
    hidden: 2048,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    rope_theta: 1_000_000.0,
    rms_eps: 1e-6,
};

/// Attention logit bias on the ICL reference prefix (0 disables). Live-tunable.
static ATTN_BOOST: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0x3f00_0000); // 0.5f32 bits

/// Set the ICL-reference attention boost (the GUI expression slider).
pub fn set_attn_boost(v: f32) {
    ATTN_BOOST.store(v.to_bits(), core::sync::atomic::Ordering::Relaxed);
}

/// Current attention boost value.
fn attn_boost_bias() -> f32 {
    f32::from_bits(ATTN_BOOST.load(core::sync::atomic::Ordering::Relaxed))
}

pub struct Layer<B: Backend> {
    pub(crate) wqkv: Tensor<B, 2>,
    pub(crate) wo: Tensor<B, 2>,
    pub(crate) w_gate_up: Tensor<B, 2>,
    pub(crate) w_down: Tensor<B, 2>,
    pub(crate) ln_in: Tensor<B, 1>,
    pub(crate) ln_post: Tensor<B, 1>,
    pub(crate) q_norm: Tensor<B, 1>,
    pub(crate) k_norm: Tensor<B, 1>,
    pub(crate) qk_table: Tensor<B, 2>,
}

fn shape_err(name: String, expected: String, got: &[usize]) -> TtsError {
    TtsError::TensorShape {
        name,
        expected,
        got: got.to_vec(),
    }
}

/// `[in, out]` tensor from `linear_t`, checked against the `[out, in]` the config implies.
fn expect_linear<B: Backend>(
    name: &str,
    w: &Tensor<B, 2>,
    out: usize,
    inp: usize,
) -> Result<(), TtsError> {
    let [i, o] = w.dims();
    if (i, o) != (inp, out) {
        return Err(shape_err(name.into(), format!("[{out}, {inp}]"), &[o, i]));
    }
    Ok(())
}

fn expect_vec<B: Backend>(name: &str, w: &Tensor<B, 1>, len: usize) -> Result<(), TtsError> {
    if w.dims()[0] != len {
        return Err(shape_err(name.into(), format!("[{len}]"), &w.dims()));
    }
    Ok(())
}

/// Layer count check: the config must claim exactly the layers the file holds.
pub(crate) fn expect_layer_count(
    wf: &WeightFile,
    prefix: &str,
    layers: usize,
) -> Result<(), TtsError> {
    let probe = |i: usize| wf.has(&format!("{prefix}.{i}.input_layernorm.weight"));
    if layers == 0 || !probe(layers - 1) {
        return Err(TtsError::InvalidConfig(format!(
            "config says {layers} layers under {prefix} but the weights hold fewer"
        )));
    }
    if probe(layers) {
        return Err(TtsError::InvalidConfig(format!(
            "config says {layers} layers under {prefix} but the weights hold more"
        )));
    }
    Ok(())
}

impl<B: Backend> Layer<B> {
    pub(crate) fn load(
        wf: &WeightFile,
        p: &str,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        let qd = heads * head_dim;
        let kvd = kv_heads * head_dim;
        let wq = wf.linear_t(&format!("{p}.self_attn.q_proj.weight"), dev)?;
        expect_linear(&format!("{p}.self_attn.q_proj.weight"), &wq, qd, hidden)?;
        let wk = wf.linear_t(&format!("{p}.self_attn.k_proj.weight"), dev)?;
        expect_linear(&format!("{p}.self_attn.k_proj.weight"), &wk, kvd, hidden)?;
        let wv = wf.linear_t(&format!("{p}.self_attn.v_proj.weight"), dev)?;
        expect_linear(&format!("{p}.self_attn.v_proj.weight"), &wv, kvd, hidden)?;
        let wg = wf.linear_t(&format!("{p}.mlp.gate_proj.weight"), dev)?;
        let wu = wf.linear_t(&format!("{p}.mlp.up_proj.weight"), dev)?;
        let inter = wg.dims()[1];
        expect_linear(&format!("{p}.mlp.gate_proj.weight"), &wg, inter, hidden)?;
        expect_linear(&format!("{p}.mlp.up_proj.weight"), &wu, inter, hidden)?;
        let qn = wf.tensor1(&format!("{p}.self_attn.q_norm.weight"), dev)?;
        expect_vec(&format!("{p}.self_attn.q_norm.weight"), &qn, head_dim)?;
        let kn = wf.tensor1(&format!("{p}.self_attn.k_norm.weight"), dev)?;
        expect_vec(&format!("{p}.self_attn.k_norm.weight"), &kn, head_dim)?;
        let wo = wf.linear_t(&format!("{p}.self_attn.o_proj.weight"), dev)?;
        expect_linear(&format!("{p}.self_attn.o_proj.weight"), &wo, hidden, qd)?;
        let w_down = wf.linear_t(&format!("{p}.mlp.down_proj.weight"), dev)?;
        expect_linear(&format!("{p}.mlp.down_proj.weight"), &w_down, hidden, inter)?;
        let ln_in = wf.tensor1(&format!("{p}.input_layernorm.weight"), dev)?;
        expect_vec(&format!("{p}.input_layernorm.weight"), &ln_in, hidden)?;
        let ln_post = wf.tensor1(&format!("{p}.post_attention_layernorm.weight"), dev)?;
        expect_vec(
            &format!("{p}.post_attention_layernorm.weight"),
            &ln_post,
            hidden,
        )?;
        let hd = head_dim;
        let mut rows: Vec<Tensor<B, 2>> = Vec::new();
        for _ in 0..heads {
            rows.push(qn.clone().reshape([1, hd]));
        }
        for _ in 0..kv_heads {
            rows.push(kn.clone().reshape([1, hd]));
        }
        let qk_table = Tensor::cat(rows, 0);
        Ok(Layer {
            wqkv: Tensor::cat(vec![wq, wk, wv], 1),
            wo,
            w_gate_up: Tensor::cat(vec![wg, wu], 1),
            w_down,
            ln_in,
            ln_post,
            q_norm: qn,
            k_norm: kn,
            qk_table,
        })
    }
}

/// Preallocated fixed-shape KV cache: every decode step sees identical tensor
/// shapes (k/v always [1, kv_heads, MAX, head_dim]), so burn's fusion/kernel
/// caches hit instead of re-JITting per step — cat-based caches grow the shape
/// every step and defeat all kernel caching. Positions >= len are masked.
pub struct KvSlot<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
}

pub struct KvCache<B: Backend> {
    pub slots: Vec<KvSlot<B>>,
    pub len: usize,
    pub max: usize,
    pub mask_full: Tensor<B, 2>,
}

impl<B: Backend> KvCache<B> {
    pub fn new(
        layers: usize,
        kv_heads: usize,
        max: usize,
        head_dim: usize,
        dev: &B::Device,
    ) -> Self {
        let vals: Vec<f32> = (0..max)
            .flat_map(|i| (0..max).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Self {
            slots: (0..layers)
                .map(|_| KvSlot {
                    k: Tensor::zeros([1, kv_heads, max, head_dim], dev),
                    v: Tensor::zeros([1, kv_heads, max, head_dim], dev),
                })
                .collect(),
            len: 0,
            max,
            mask_full: Tensor::from_data(burn::tensor::TensorData::new(vals, [max, max]), dev),
        }
    }

    pub(crate) fn push(&mut self, layer: usize, k: Tensor<B, 4>, v: Tensor<B, 4>, offset: usize) {
        let [_, kvh, s, hd] = k.dims();
        let slot = &mut self.slots[layer];
        slot.k = slot
            .k
            .clone()
            .slice_assign([0..1, 0..kvh, offset..offset + s, 0..hd], k);
        slot.v = slot
            .v
            .clone()
            .slice_assign([0..1, 0..kvh, offset..offset + s, 0..hd], v);
    }
}

pub struct Talker<B: Backend> {
    pub cfg: TalkerConfig,
    // 151936 x 2048 f16 = 594 MiB that is only ever gathered a few rows at a
    // time; it stays in host memory and the rows travel to the device.
    text_embedding: crate::weights::HostTable,
    text_fc1_w: Tensor<B, 2>,
    text_fc1_b: Tensor<B, 1>,
    text_fc2_w: Tensor<B, 2>,
    text_fc2_b: Tensor<B, 1>,
    pub codec_embedding: Tensor<B, 2>,
    layers: Vec<Layer<B>>,
    norm: Tensor<B, 1>,
    codec_head: Tensor<B, 2>,
    inv_freq: Vec<f64>,
    cos_table: Tensor<B, 2>,
    sin_table: Tensor<B, 2>,
}

pub(crate) fn rmsnorm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    w: &Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, D> {
    // Compute the norm in f32: the sum of squares over the hidden dim overflows
    // f16's 65504 range once the residual stream grows (deep layers), producing
    // NaNs that wreck generation. No-op on an f32 backend (parity preserved).
    let dt = x.dtype();
    let xf = x.cast(burn::tensor::FloatDType::F32);
    let sq = xf.clone().powf_scalar(2.0).mean_dim(D - 1);
    let normed = (xf / (sq + eps).sqrt()).cast(dt);
    normed * w.clone().unsqueeze()
}

/// Standard half-split RoPE: [x1*cos - x2*sin, x2*cos + x1*sin].
/// (The model's MRoPE reduces to this when all three position streams carry
/// the same value, which is always true for TTS.)
pub(crate) fn rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..d / 2]);
    let x2 = x.slice([0..b, 0..h, 0..s, d / 2..d]);
    let cos = cos.clone().reshape([1, 1, s, d / 2]);
    let sin = sin.clone().reshape([1, 1, s, d / 2]);
    let r1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let r2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![r1, r2], 3)
}

impl<B: Backend> Talker<B> {
    pub fn load(wf: &WeightFile, cfg: TalkerConfig, dev: &B::Device) -> Result<Self, TtsError> {
        if cfg.kv_heads == 0
            || !cfg.heads.is_multiple_of(cfg.kv_heads)
            || !cfg.head_dim.is_multiple_of(2)
        {
            return Err(TtsError::InvalidConfig(format!(
                "talker heads {} / kv_heads {} / head_dim {} are inconsistent",
                cfg.heads, cfg.kv_heads, cfg.head_dim
            )));
        }
        expect_layer_count(wf, "talker.model.layers", cfg.layers)?;
        let layers = (0..cfg.layers)
            .map(|i| {
                Layer::load(
                    wf,
                    &format!("talker.model.layers.{i}"),
                    cfg.hidden,
                    cfg.heads,
                    cfg.kv_heads,
                    cfg.head_dim,
                    dev,
                )
            })
            .collect::<Result<_, _>>()?;
        let inv_freq: Vec<f64> = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64))
            .collect();
        let max_pos = 2048usize;
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
        let text_embedding = wf.rows_f16("talker.model.text_embedding.weight")?;
        let text_fc1_w = wf.linear_t("talker.text_projection.linear_fc1.weight", dev)?;
        let text_fc1_b = wf.tensor1("talker.text_projection.linear_fc1.bias", dev)?;
        let text_fc2_w = wf.linear_t("talker.text_projection.linear_fc2.weight", dev)?;
        let text_fc2_b = wf.tensor1("talker.text_projection.linear_fc2.bias", dev)?;
        let fc1_out = text_fc1_w.dims()[1];
        let tp = "talker.text_projection";
        let in_dim = text_embedding.cols;
        expect_linear(
            &format!("{tp}.linear_fc1.weight"),
            &text_fc1_w,
            fc1_out,
            in_dim,
        )?;
        expect_vec(&format!("{tp}.linear_fc1.bias"), &text_fc1_b, fc1_out)?;
        expect_linear(
            &format!("{tp}.linear_fc2.weight"),
            &text_fc2_w,
            cfg.hidden,
            fc1_out,
        )?;
        expect_vec(&format!("{tp}.linear_fc2.bias"), &text_fc2_b, cfg.hidden)?;
        let codec_embedding = wf.tensor2("talker.model.codec_embedding.weight", dev)?;
        let vocab = codec_embedding.dims()[0];
        if codec_embedding.dims()[1] != cfg.hidden {
            return Err(shape_err(
                "talker.model.codec_embedding.weight".into(),
                format!("[vocab, {}]", cfg.hidden),
                &codec_embedding.dims(),
            ));
        }
        let norm = wf.tensor1("talker.model.norm.weight", dev)?;
        expect_vec("talker.model.norm.weight", &norm, cfg.hidden)?;
        let codec_head = wf.linear_t("talker.codec_head.weight", dev)?;
        expect_linear("talker.codec_head.weight", &codec_head, vocab, cfg.hidden)?;
        Ok(Self {
            cfg,
            text_embedding,
            text_fc1_w,
            text_fc1_b,
            text_fc2_w,
            text_fc2_b,
            codec_embedding,
            layers,
            norm,
            codec_head,
            inv_freq,
            cos_table,
            sin_table,
        })
    }

    /// Rows of the text embedding table: every text id must be below this.
    pub fn text_vocab_rows(&self) -> usize {
        self.text_embedding.rows
    }

    /// Rows of the codec embedding table: every codec id must be below this.
    pub fn codec_vocab_rows(&self) -> usize {
        self.codec_embedding.dims()[0]
    }

    fn cos_sin(&self, offset: usize, seq: usize, dev: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let half = self.cfg.head_dim / 2;
        let mut cos = Vec::with_capacity(seq * half);
        let mut sin = Vec::with_capacity(seq * half);
        for p in offset..offset + seq {
            for f in &self.inv_freq {
                let a = p as f64 * f;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        (
            Tensor::from_data(burn::tensor::TensorData::new(cos, [seq, half]), dev),
            Tensor::from_data(burn::tensor::TensorData::new(sin, [seq, half]), dev),
        )
    }

    pub fn embed_text(&self, ids: &[u32], dev: &B::Device) -> Result<Tensor<B, 3>, TtsError> {
        let emb = self.text_embedding.gather::<B>(ids, dev)?;
        let h = silu(emb.matmul(self.text_fc1_w.clone()) + self.text_fc1_b.clone().unsqueeze());
        let h = h.matmul(self.text_fc2_w.clone()) + self.text_fc2_b.clone().unsqueeze();
        let [s, d] = h.dims();
        Ok(h.reshape([1, s, d]))
    }

    /// Full-sequence causal forward over hidden states -> logits [1, S, vocab].
    pub fn forward_hidden(&self, hidden: Tensor<B, 3>, dev: &B::Device) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let [_, s, _] = hidden.dims();
        let group = cfg.heads / cfg.kv_heads;
        let scale = (cfg.head_dim as f64).powf(-0.5);
        let (cos, sin) = self.cos_sin(0, s, dev);

        let mask_vals: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let mask: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(mask_vals, [1, 1, s, s]), dev);

        let mut h = hidden;
        for l in &self.layers {
            let normed = rmsnorm(h.clone(), &l.ln_in, cfg.rms_eps);
            let qd = cfg.heads * cfg.head_dim;
            let kvd = cfg.kv_heads * cfg.head_dim;
            let qkv = normed.matmul(l.wqkv.clone().unsqueeze());
            let q = qkv
                .clone()
                .slice([0..1, 0..s, 0..qd])
                .reshape([1, s, cfg.heads, cfg.head_dim]);
            let k = qkv.clone().slice([0..1, 0..s, qd..qd + kvd]).reshape([
                1,
                s,
                cfg.kv_heads,
                cfg.head_dim,
            ]);
            let v = qkv.slice([0..1, 0..s, qd + kvd..qd + 2 * kvd]).reshape([
                1,
                s,
                cfg.kv_heads,
                cfg.head_dim,
            ]);
            let q = rmsnorm(q, &l.q_norm, cfg.rms_eps).swap_dims(1, 2);
            let k = rmsnorm(k, &l.k_norm, cfg.rms_eps).swap_dims(1, 2);
            let v = v.swap_dims(1, 2);

            let q = rope(q, &cos, &sin);
            let k = rope(k, &cos, &sin);

            let kx = k
                .unsqueeze_dim::<5>(2)
                .expand([1, cfg.kv_heads, group, s, cfg.head_dim])
                .reshape([1, cfg.heads, s, cfg.head_dim]);
            let vx = v
                .unsqueeze_dim::<5>(2)
                .expand([1, cfg.kv_heads, group, s, cfg.head_dim])
                .reshape([1, cfg.heads, s, cfg.head_dim]);

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
        rmsnorm(h, &self.norm, cfg.rms_eps).matmul(self.codec_head.clone().unsqueeze())
    }

    pub fn embed_codec_ids(&self, ids: &[u32], dev: &B::Device) -> Result<Tensor<B, 3>, TtsError> {
        let rows = self.codec_embedding.dims()[0];
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= rows) {
            return Err(TtsError::InvalidPrompt(format!(
                "codec token id {bad} is out of range ({rows} rows)"
            )));
        }
        let idx: Tensor<B, 1, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(
                ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                [ids.len()],
            ),
            dev,
        );
        let e = self.codec_embedding.clone().select(0, idx);
        let [s, d] = e.dims();
        Ok(e.reshape([1, s, d]))
    }

    pub fn head_logits(&self, normed_last: Tensor<B, 3>) -> Result<Vec<f32>, TtsError> {
        let h = normed_last.cast(burn::tensor::FloatDType::F32);
        let w = self
            .codec_head
            .clone()
            .cast(burn::tensor::FloatDType::F32)
            .unsqueeze();
        h.matmul(w)
            .into_data()
            .to_vec()
            .map_err(|e| TtsError::Gpu(format!("logits readback: {e:?}")))
    }

    /// KV-cached pass over a chunk (or single step). Returns NORMED hidden for
    /// the chunk; caller slices the last position for logits / code predictor.
    pub fn run_cached(
        &self,
        x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        offset: usize,
        causal: bool,
        boost_prefix: usize,
        dev: &B::Device,
    ) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let [_, s, _] = x.dims();
        let group = cfg.heads / cfg.kv_heads;
        let scale = (cfg.head_dim as f64).powf(-0.5);
        let _ = causal;
        let half = cfg.head_dim / 2;
        let cos = self.cos_table.clone().slice([offset..offset + s, 0..half]);
        let sin = self.sin_table.clone().slice([offset..offset + s, 0..half]);

        let max = cache.max;
        let mut mask: Tensor<B, 4> = cache
            .mask_full
            .clone()
            .slice([offset..offset + s, 0..max])
            .reshape([1, 1, s, max]);

        // Bias the reference prefix's attention up so it isn't diluted as the cache grows.
        let bias = attn_boost_bias();
        if boost_prefix > 0 && bias != 0.0 {
            let bp = boost_prefix.min(max);
            let mut bvals = vec![0.0f32; max];
            for v in bvals[0..bp].iter_mut() {
                *v = bias;
            }
            let boost: Tensor<B, 4> =
                Tensor::from_data(burn::tensor::TensorData::new(bvals, [1, 1, 1, max]), dev);
            mask = mask + boost;
        }

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

    /// Mirrors the candle port's `TalkerModel::forward`: text ids -> logits.
    pub fn forward_text(&self, ids: &[u32], dev: &B::Device) -> Result<Tensor<B, 3>, TtsError> {
        let hidden = self.embed_text(ids, dev)?;
        Ok(self.forward_hidden(hidden, dev))
    }
}
