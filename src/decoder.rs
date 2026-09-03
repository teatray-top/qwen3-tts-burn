use burn::prelude::*;
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::Int;

use crate::error::TtsError;
use crate::weights::WeightFile;

const RMS_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-6;
const SNAKE_EPS: f64 = 1e-9;
/// Left-context frames the streaming upsampling window keeps.
const UPSAMPLE_CTX_FRAMES: usize = 40;
const PT_HEADS: usize = 16;
const PT_HEAD_DIM: usize = 64;

type ResUnit<B> = (Snake<B>, Conv<B>, Snake<B>, Conv<B>);
type UpBlock<B> = (Snake<B>, TConv<B>, Vec<ResUnit<B>>);

struct Conv<B: Backend> {
    w: Tensor<B, 3>,
    b: Option<Tensor<B, 1>>,
    dilation: usize,
    groups: usize,
    kernel: usize,
}

impl<B: Backend> Conv<B> {
    fn load(
        wf: &WeightFile,
        prefix: &str,
        dilation: usize,
        groups: usize,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        let w = wf.tensor3(&format!("{prefix}.weight"), dev)?;
        let kernel = w.dims()[2];
        let b = wf.try_tensor1(&format!("{prefix}.bias"), dev)?;
        Ok(Self {
            w,
            b,
            dilation,
            groups,
            kernel,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bt, c, t] = x.dims();
        let pad = self.dilation * (self.kernel - 1);
        let x = if pad > 0 {
            let zeros = Tensor::zeros([bt, c, pad], &x.device());
            Tensor::cat(vec![zeros, x], 2)
        } else {
            x
        };
        let _ = t;
        conv1d(
            x,
            self.w.clone(),
            self.b.clone(),
            ConvOptions::new([1], [0], [self.dilation], self.groups),
        )
    }
}

struct TConv<B: Backend> {
    w: Tensor<B, 3>,
    b: Tensor<B, 1>,
    stride: usize,
    kernel: usize,
}

impl<B: Backend> TConv<B> {
    fn load(
        wf: &WeightFile,
        prefix: &str,
        stride: usize,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        let w = wf.tensor3(&format!("{prefix}.weight"), dev)?;
        let kernel = w.dims()[2];
        if kernel < stride {
            return Err(TtsError::TensorShape {
                name: format!("{prefix}.weight"),
                expected: format!("kernel >= stride {stride}"),
                got: w.dims().to_vec(),
            });
        }
        let b = wf.tensor1(&format!("{prefix}.bias"), dev)?;
        Ok(Self {
            w,
            b,
            stride,
            kernel,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = conv_transpose1d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvTransposeOptions::new([self.stride], [0], [0], [1], 1),
        );
        let [bt, c, t] = y.dims();
        let trim = self.kernel - self.stride;
        if trim > 0 {
            y.slice([0..bt, 0..c, 0..t - trim])
        } else {
            y
        }
    }
}

struct Snake<B: Backend> {
    alpha: Tensor<B, 3>,
    beta: Tensor<B, 3>,
}

impl<B: Backend> Snake<B> {
    fn load(wf: &WeightFile, prefix: &str, dev: &B::Device) -> Result<Self, TtsError> {
        let a = wf.tensor1(&format!("{prefix}.alpha"), dev)?;
        let b = wf.tensor1(&format!("{prefix}.beta"), dev)?;
        let c = a.dims()[0];
        if b.dims()[0] != c {
            return Err(TtsError::TensorShape {
                name: format!("{prefix}.beta"),
                expected: format!("[{c}]"),
                got: b.dims().to_vec(),
            });
        }
        Ok(Self {
            alpha: a.exp().reshape([1, c, 1]),
            beta: (b.exp() + SNAKE_EPS).reshape([1, c, 1]),
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let s = (x.clone() * self.alpha.clone()).sin();
        x + s.clone() * s / self.beta.clone()
    }
}

struct ConvNext<B: Backend> {
    dw: Conv<B>,
    ln_w: Tensor<B, 1>,
    ln_b: Tensor<B, 1>,
    pw1_w: Tensor<B, 2>,
    pw1_b: Tensor<B, 1>,
    pw2_w: Tensor<B, 2>,
    pw2_b: Tensor<B, 1>,
    gamma: Tensor<B, 1>,
}

impl<B: Backend> ConvNext<B> {
    fn load(wf: &WeightFile, prefix: &str, dim: usize, dev: &B::Device) -> Result<Self, TtsError> {
        Ok(Self {
            dw: Conv::load(wf, &format!("{prefix}.dwconv.conv"), 1, dim, dev)?,
            ln_w: wf.tensor1(&format!("{prefix}.norm.weight"), dev)?,
            ln_b: wf.tensor1(&format!("{prefix}.norm.bias"), dev)?,
            pw1_w: wf.linear_t(&format!("{prefix}.pwconv1.weight"), dev)?,
            pw1_b: wf.tensor1(&format!("{prefix}.pwconv1.bias"), dev)?,
            pw2_w: wf.linear_t(&format!("{prefix}.pwconv2.weight"), dev)?,
            pw2_b: wf.tensor1(&format!("{prefix}.pwconv2.bias"), dev)?,
            gamma: wf.tensor1(&format!("{prefix}.gamma"), dev)?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let res = x.clone();
        let h = self.dw.forward(x).swap_dims(1, 2);
        let mean = h.clone().mean_dim(2);
        let centered = h - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let h = centered / (var + LN_EPS).sqrt();
        let h = h * self.ln_w.clone().unsqueeze() + self.ln_b.clone().unsqueeze();
        let h = gelu(h.matmul(self.pw1_w.clone().unsqueeze()) + self.pw1_b.clone().unsqueeze());
        let h = h.matmul(self.pw2_w.clone().unsqueeze()) + self.pw2_b.clone().unsqueeze();
        let h = h * self.gamma.clone().unsqueeze();
        res + h.swap_dims(1, 2)
    }
}

struct PreLayer<B: Backend> {
    ln1: Tensor<B, 1>,
    wq: Tensor<B, 2>,
    wk: Tensor<B, 2>,
    wv: Tensor<B, 2>,
    wo: Tensor<B, 2>,
    attn_scale_vec: Tensor<B, 1>,
    ln2: Tensor<B, 1>,
    w_gate: Tensor<B, 2>,
    w_up: Tensor<B, 2>,
    w_down: Tensor<B, 2>,
    mlp_scale_vec: Tensor<B, 1>,
}

pub struct Decoder12Hz<B: Backend> {
    cb_first: Tensor<B, 2>,
    cb_rest: Vec<Tensor<B, 2>>,
    first_out_proj: Tensor<B, 2>,
    rest_out_proj: Tensor<B, 2>,
    pre_conv: Conv<B>,
    pt_in_w: Tensor<B, 2>,
    pt_in_b: Tensor<B, 1>,
    pt_layers: Vec<PreLayer<B>>,
    pt_norm: Tensor<B, 1>,
    pt_out_w: Tensor<B, 2>,
    pt_out_b: Tensor<B, 1>,
    up_tconv: Vec<TConv<B>>,
    up_next: Vec<ConvNext<B>>,
    init_conv: Conv<B>,
    blocks: Vec<UpBlock<B>>,
    final_snake: Snake<B>,
    final_conv: Conv<B>,
    cb_sizes: Vec<usize>,
}

fn rmsn<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 1>) -> Tensor<B, 3> {
    let sq = x.clone().powf_scalar(2.0).mean_dim(2);
    (x / (sq + RMS_EPS).sqrt()) * w.clone().unsqueeze()
}

fn codebook<B: Backend>(
    wf: &WeightFile,
    prefix: &str,
    dev: &B::Device,
) -> Result<Tensor<B, 2>, TtsError> {
    let sum = wf.tensor2(&format!("{prefix}.embedding_sum"), dev)?;
    let usage = wf.tensor1(&format!("{prefix}.cluster_usage"), dev)?;
    if usage.dims()[0] != sum.dims()[0] {
        return Err(TtsError::TensorShape {
            name: format!("{prefix}.cluster_usage"),
            expected: format!("[{}]", sum.dims()[0]),
            got: usage.dims().to_vec(),
        });
    }
    Ok(sum / usage.clamp_min(1e-7).unsqueeze_dim::<2>(1))
}

impl<B: Backend> Decoder12Hz<B> {
    pub fn load(wf: &WeightFile, dev: &B::Device) -> Result<Self, TtsError> {
        let squeeze_proj = |name: &str| -> Result<Tensor<B, 2>, TtsError> {
            let w = wf.tensor3(name, dev)?;
            let [o, i, _] = w.dims();
            Ok(w.reshape([o, i]).swap_dims(0, 1))
        };
        let pt_width = PT_HEADS * PT_HEAD_DIM;
        let pt_layers = (0..8)
            .map(|i| {
                let p = format!("decoder.pre_transformer.layers.{i}");
                let wq = wf.linear_t(&format!("{p}.self_attn.q_proj.weight"), dev)?;
                let wk = wf.linear_t(&format!("{p}.self_attn.k_proj.weight"), dev)?;
                let wv = wf.linear_t(&format!("{p}.self_attn.v_proj.weight"), dev)?;
                for (name, w) in [("q_proj", &wq), ("k_proj", &wk), ("v_proj", &wv)] {
                    if w.dims()[1] != pt_width {
                        return Err(TtsError::TensorShape {
                            name: format!("{p}.self_attn.{name}.weight"),
                            expected: format!("[{pt_width}, hidden]"),
                            got: vec![w.dims()[1], w.dims()[0]],
                        });
                    }
                }
                Ok(PreLayer {
                    ln1: wf.tensor1(&format!("{p}.input_layernorm.weight"), dev)?,
                    wq,
                    wk,
                    wv,
                    wo: wf.linear_t(&format!("{p}.self_attn.o_proj.weight"), dev)?,
                    attn_scale_vec: wf.tensor1(&format!("{p}.self_attn_layer_scale.scale"), dev)?,
                    ln2: wf.tensor1(&format!("{p}.post_attention_layernorm.weight"), dev)?,
                    w_gate: wf.linear_t(&format!("{p}.mlp.gate_proj.weight"), dev)?,
                    w_up: wf.linear_t(&format!("{p}.mlp.up_proj.weight"), dev)?,
                    w_down: wf.linear_t(&format!("{p}.mlp.down_proj.weight"), dev)?,
                    mlp_scale_vec: wf.tensor1(&format!("{p}.mlp_layer_scale.scale"), dev)?,
                })
            })
            .collect::<Result<_, TtsError>>()?;
        let block_specs = [
            (1536usize, 768usize, 8usize),
            (768, 384, 5),
            (384, 192, 4),
            (192, 96, 3),
        ];
        let blocks = block_specs
            .iter()
            .enumerate()
            .map(|(bi, &(_ci, _co, rate))| {
                let p = format!("decoder.decoder.{}", bi + 1);
                let snake = Snake::load(wf, &format!("{p}.block.0"), dev)?;
                let tconv = TConv::load(wf, &format!("{p}.block.1.conv"), rate, dev)?;
                let residuals = (2..5)
                    .zip([1usize, 3, 9])
                    .map(|(slot, dil)| {
                        let rp = format!("{p}.block.{slot}");
                        Ok((
                            Snake::load(wf, &format!("{rp}.act1"), dev)?,
                            Conv::load(wf, &format!("{rp}.conv1.conv"), dil, 1, dev)?,
                            Snake::load(wf, &format!("{rp}.act2"), dev)?,
                            Conv::load(wf, &format!("{rp}.conv2.conv"), 1, 1, dev)?,
                        ))
                    })
                    .collect::<Result<_, TtsError>>()?;
                Ok((snake, tconv, residuals))
            })
            .collect::<Result<_, TtsError>>()?;
        let cb_first: Tensor<B, 2> =
            codebook(wf, "decoder.quantizer.rvq_first.vq.layers.0._codebook", dev)?;
        let cb_rest: Vec<Tensor<B, 2>> = (0..15)
            .map(|i| {
                codebook(
                    wf,
                    &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook"),
                    dev,
                )
            })
            .collect::<Result<_, _>>()?;
        let cb_sizes = std::iter::once(&cb_first)
            .chain(&cb_rest)
            .map(|cb| cb.dims()[0])
            .collect();
        Ok(Self {
            cb_first,
            cb_rest,
            first_out_proj: squeeze_proj("decoder.quantizer.rvq_first.output_proj.weight")?,
            rest_out_proj: squeeze_proj("decoder.quantizer.rvq_rest.output_proj.weight")?,
            pre_conv: Conv::load(wf, "decoder.pre_conv.conv", 1, 1, dev)?,
            pt_in_w: wf.linear_t("decoder.pre_transformer.input_proj.weight", dev)?,
            pt_in_b: wf.tensor1("decoder.pre_transformer.input_proj.bias", dev)?,
            pt_layers,
            pt_norm: wf.tensor1("decoder.pre_transformer.norm.weight", dev)?,
            pt_out_w: wf.linear_t("decoder.pre_transformer.output_proj.weight", dev)?,
            pt_out_b: wf.tensor1("decoder.pre_transformer.output_proj.bias", dev)?,
            up_tconv: (0..2)
                .map(|i| TConv::load(wf, &format!("decoder.upsample.{i}.0.conv"), 2, dev))
                .collect::<Result<_, _>>()?,
            up_next: (0..2)
                .map(|i| ConvNext::load(wf, &format!("decoder.upsample.{i}.1"), 1024, dev))
                .collect::<Result<_, _>>()?,
            init_conv: Conv::load(wf, "decoder.decoder.0.conv", 1, 1, dev)?,
            blocks,
            final_snake: Snake::load(wf, "decoder.decoder.5", dev)?,
            final_conv: Conv::load(wf, "decoder.decoder.6.conv", 1, 1, dev)?,
            cb_sizes,
        })
    }

    /// Codes per frame (semantic first).
    pub fn groups(&self) -> usize {
        self.cb_sizes.len()
    }

    fn validate_frames(&self, frames: &[Vec<u32>]) -> Result<(), TtsError> {
        let groups = self.cb_sizes.len();
        for (i, f) in frames.iter().enumerate() {
            if f.len() != groups {
                return Err(TtsError::InvalidFrames(format!(
                    "frame {i} has {} codes, expected {groups}",
                    f.len()
                )));
            }
            for (g, (&c, &size)) in f.iter().zip(&self.cb_sizes).enumerate() {
                if c as usize >= size {
                    return Err(TtsError::InvalidFrames(format!(
                        "frame {i} code {g} is {c}, codebook has {size} entries"
                    )));
                }
            }
        }
        Ok(())
    }

    fn pre_transformer(&self, x: Tensor<B, 3>, dev: &B::Device) -> Tensor<B, 3> {
        let heads = PT_HEADS;
        let hd = PT_HEAD_DIM;
        let [_, t, _] = x.dims();
        let mut h = x.matmul(self.pt_in_w.clone().unsqueeze()) + self.pt_in_b.clone().unsqueeze();

        let half = hd / 2;
        let mut cos_v = Vec::with_capacity(t * hd);
        let mut sin_v = Vec::with_capacity(t * hd);
        for p in 0..t {
            let row: Vec<f64> = (0..half)
                .map(|k| p as f64 / 10000f64.powf(2.0 * k as f64 / hd as f64))
                .collect();
            for pass in 0..2 {
                let _ = pass;
                for a in &row {
                    cos_v.push(a.cos() as f32);
                    sin_v.push(a.sin() as f32);
                }
            }
        }
        let cos: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(cos_v, [1, 1, t, hd]), dev);
        let sin: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(sin_v, [1, 1, t, hd]), dev);

        let mask_vals: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let mask: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(mask_vals, [1, 1, t, t]), dev);

        let rot = |x: Tensor<B, 4>| -> Tensor<B, 4> {
            let [b, hh, s, d] = x.dims();
            let x1 = x.clone().slice([0..b, 0..hh, 0..s, 0..d / 2]);
            let x2 = x.slice([0..b, 0..hh, 0..s, d / 2..d]);
            Tensor::cat(vec![x2.neg(), x1], 3)
        };

        for l in &self.pt_layers {
            let n = rmsn(h.clone(), &l.ln1);
            let q = n
                .clone()
                .matmul(l.wq.clone().unsqueeze())
                .reshape([1, t, heads, hd])
                .swap_dims(1, 2);
            let k = n
                .clone()
                .matmul(l.wk.clone().unsqueeze())
                .reshape([1, t, heads, hd])
                .swap_dims(1, 2);
            let v = n
                .matmul(l.wv.clone().unsqueeze())
                .reshape([1, t, heads, hd])
                .swap_dims(1, 2);
            let q = q.clone() * cos.clone() + rot(q) * sin.clone();
            let k = k.clone() * cos.clone() + rot(k) * sin.clone();
            let att = softmax(q.matmul(k.swap_dims(2, 3)) * 0.125 + mask.clone(), 3);
            let out = att.matmul(v).swap_dims(1, 2).reshape([1, t, heads * hd]);
            let out = out.matmul(l.wo.clone().unsqueeze());
            h = h + out * l.attn_scale_vec.clone().unsqueeze();

            let n2 = rmsn(h.clone(), &l.ln2);
            let mlp = (silu(n2.clone().matmul(l.w_gate.clone().unsqueeze()))
                * n2.matmul(l.w_up.clone().unsqueeze()))
            .matmul(l.w_down.clone().unsqueeze());
            h = h + mlp * l.mlp_scale_vec.clone().unsqueeze();
        }
        let h = rmsn(h, &self.pt_norm);
        h.matmul(self.pt_out_w.clone().unsqueeze()) + self.pt_out_b.clone().unsqueeze()
    }

    /// Embed a `t`-bucketed frame block to the pre-conv input latent.
    fn embed_frames(&self, frames: &[Vec<u32>], t: usize, dev: &B::Device) -> Tensor<B, 3> {
        let real_t = frames.len();
        let padded: Vec<&Vec<u32>> = frames
            .iter()
            .chain(std::iter::repeat(&frames[real_t - 1]))
            .take(t)
            .collect();
        let idx = |vals: Vec<i32>| -> Tensor<B, 1, Int> {
            Tensor::from_data(burn::tensor::TensorData::new(vals, [t]), dev)
        };
        let first_ids = idx(padded.iter().map(|f| f[0] as i32).collect());
        let first = self.cb_first.clone().select(0, first_ids);
        let mut rest = self.cb_rest[0]
            .clone()
            .select(0, idx(padded.iter().map(|f| f[1] as i32).collect()));
        for g in 1..15 {
            rest = rest
                + self.cb_rest[g]
                    .clone()
                    .select(0, idx(padded.iter().map(|f| f[g + 1] as i32).collect()));
        }
        let [_, e] = first.dims();
        let first = first
            .reshape([1, t, e])
            .matmul(self.first_out_proj.clone().unsqueeze());
        let rest = rest
            .reshape([1, t, e])
            .matmul(self.rest_out_proj.clone().unsqueeze());
        (first + rest).swap_dims(1, 2)
    }

    /// Upsample the post-transformer latent to a 24 kHz waveform.
    fn upsample_latent(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for i in 0..2 {
            x = self.up_tconv[i].forward(x);
            x = self.up_next[i].forward(x);
        }
        x = self.init_conv.forward(x);
        for (snake, tconv, residuals) in &self.blocks {
            x = snake.forward(x);
            x = tconv.forward(x);
            for (a1, c1, a2, c2) in residuals {
                let y = c2.forward(a2.forward(c1.forward(a1.forward(x.clone()))));
                x = x + y;
            }
        }
        x = self.final_snake.forward(x);
        self.final_conv.forward(x).clamp(-1.0, 1.0)
    }

    /// Decode frames (× 16 codes, semantic first) to mono f32 samples at 24 kHz.
    pub fn decode(&self, frames: &[Vec<u32>], dev: &B::Device) -> Result<Vec<f32>, TtsError> {
        self.validate_frames(frames)?;
        let real_t = frames.len();
        if real_t == 0 {
            return Ok(Vec::new());
        }
        // Bucket the frame count to a power of two (min 32) so the decoder sees few
        // shapes; trailing padding is dropped since the decoder is causal.
        let t = real_t.next_power_of_two().max(32);
        let quantized = self.embed_frames(frames, t, dev);
        let x = self.pre_conv.forward(quantized);
        let x = self.pre_transformer(x.swap_dims(1, 2), dev).swap_dims(1, 2);
        let x = self.upsample_latent(x);
        let mut out: Vec<f32> = readback(x)?;
        let spf = out.len() / t;
        out.truncate(real_t * spf);
        Ok(out)
    }

    /// Stream-decode the tail [keep_from..N]: full-prefix attention (global, so
    /// the voice can't drift between chunks) but windowed upsampling. Bit-exact to
    /// `decode` over the returned region at O(new) upsampling cost.
    pub fn decode_window(
        &self,
        frames: &[Vec<u32>],
        keep_from: usize,
        dev: &B::Device,
    ) -> Result<Vec<f32>, TtsError> {
        self.validate_frames(frames)?;
        let real_t = frames.len();
        if real_t == 0 || keep_from >= real_t {
            return Ok(Vec::new());
        }
        let t = real_t.next_power_of_two().max(32);
        let quantized = self.embed_frames(frames, t, dev);
        let x = self.pre_conv.forward(quantized);
        let x = self.pre_transformer(x.swap_dims(1, 2), dev).swap_dims(1, 2);

        let w0 = keep_from.saturating_sub(UPSAMPLE_CTX_FRAMES);
        let wlen = real_t - w0;
        let tw = wlen.next_power_of_two().max(32);
        let [_, c, _] = x.dims();
        let mut xw = x.slice([0..1, 0..c, w0..w0 + wlen]);
        if tw > wlen {
            let last = xw.clone().slice([0..1, 0..c, wlen - 1..wlen]);
            let pads: Vec<Tensor<B, 3>> = (0..tw - wlen).map(|_| last.clone()).collect();
            xw = Tensor::cat(std::iter::once(xw).chain(pads).collect::<Vec<_>>(), 2);
        }
        let up = self.upsample_latent(xw);
        let mut out: Vec<f32> = readback(up)?;
        let spf = out.len() / tw;
        out.truncate(wlen * spf);
        Ok(out.split_off((keep_from - w0) * spf))
    }
}

fn readback<B: Backend>(x: Tensor<B, 3>) -> Result<Vec<f32>, TtsError> {
    x.cast(burn::tensor::FloatDType::F32)
        .into_data()
        .to_vec()
        .map_err(|e| TtsError::Gpu(format!("waveform readback: {e:?}")))
}
