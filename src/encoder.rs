use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use crate::error::TtsError;
use crate::weights::WeightFile;

const LN_EPS: f64 = 1e-5;
const SR: f64 = 24000.0;
const STRIDES: [usize; 4] = [4, 5, 6, 8];
const DOWNSAMPLE_STRIDE: usize = 2;
/// 24 kHz samples per 12.5 Hz frame: the product of every stride in the encoder.
pub const SAMPLES_PER_FRAME: usize =
    STRIDES[0] * STRIDES[1] * STRIDES[2] * STRIDES[3] * DOWNSAMPLE_STRIDE;

fn elu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.clone().clamp_min(0.0) + (x.clamp_max(0.0).exp() - 1.0)
}

struct EConv<B: Backend> {
    w: Tensor<B, 3>,
    b: Option<Tensor<B, 1>>,
    stride: usize,
    kernel: usize,
    replicate: bool,
}

impl<B: Backend> EConv<B> {
    fn load(
        wf: &WeightFile,
        prefix: &str,
        stride: usize,
        replicate: bool,
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
        Ok(Self {
            w,
            b: wf.try_tensor1(&format!("{prefix}.bias"), dev)?,
            stride,
            kernel,
            replicate,
        })
    }

    /// Causal pad (mimi StreamableConv1d): left = k_eff - s, right pads the
    /// signal up to a multiple of the stride. Zeros for SEANet, replicate for
    /// the 25→12.5Hz downsample conv.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, t] = x.dims();
        let pad_left = self.kernel - self.stride;
        let pad_right = t.div_ceil(self.stride) * self.stride - t;
        let mut parts = Vec::new();
        if pad_left > 0 {
            if self.replicate {
                parts.push(x.clone().slice([0..b, 0..c, 0..1]).expand([b, c, pad_left]));
            } else {
                parts.push(Tensor::zeros([b, c, pad_left], &x.device()));
            }
        }
        parts.push(x.clone());
        if pad_right > 0 {
            if self.replicate {
                parts.push(
                    x.clone()
                        .slice([0..b, 0..c, t - 1..t])
                        .expand([b, c, pad_right]),
                );
            } else {
                parts.push(Tensor::zeros([b, c, pad_right], &x.device()));
            }
        }
        let x = Tensor::cat(parts, 2);
        conv1d(
            x,
            self.w.clone(),
            self.b.clone(),
            ConvOptions::new([self.stride], [0], [1], 1),
        )
    }
}

struct ResBlock<B: Backend> {
    c1: EConv<B>,
    c3: EConv<B>,
}

struct TLayer<B: Backend> {
    ln1_w: Tensor<B, 1>,
    ln1_b: Tensor<B, 1>,
    wq: Tensor<B, 2>,
    wk: Tensor<B, 2>,
    wv: Tensor<B, 2>,
    wo: Tensor<B, 2>,
    ls1: Tensor<B, 1>,
    ln2_w: Tensor<B, 1>,
    ln2_b: Tensor<B, 1>,
    fc1: Tensor<B, 2>,
    fc2: Tensor<B, 2>,
    ls2: Tensor<B, 1>,
}

pub struct Encoder12Hz<B: Backend> {
    conv0: EConv<B>,
    res: Vec<ResBlock<B>>,
    down: Vec<EConv<B>>,
    conv14: EConv<B>,
    tlayers: Vec<TLayer<B>>,
    downsample: EConv<B>,
    sem_in_proj: Tensor<B, 2>,
    sem_cb: Tensor<B, 2>,
    ac_in_proj: Tensor<B, 2>,
    ac_cbs: Vec<Tensor<B, 2>>,
}

fn layernorm<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    centered / (var + LN_EPS).sqrt() * w.clone().unsqueeze() + b.clone().unsqueeze()
}

fn codebook<B: Backend>(
    wf: &WeightFile,
    prefix: &str,
    dev: &B::Device,
) -> Result<Tensor<B, 2>, TtsError> {
    let sum = wf.tensor2(&format!("{prefix}.embed_sum"), dev)?;
    let usage = wf
        .tensor1(&format!("{prefix}.cluster_usage"), dev)?
        .clamp_min(1e-5);
    Ok(sum / usage.unsqueeze_dim::<2>(1))
}

fn proj1x1<B: Backend>(
    wf: &WeightFile,
    name: &str,
    dev: &B::Device,
) -> Result<Tensor<B, 2>, TtsError> {
    let w = wf.tensor3(name, dev)?;
    let [o, i, _] = w.dims();
    Ok(w.reshape([o, i]).swap_dims(0, 1))
}

impl<B: Backend> Encoder12Hz<B> {
    pub fn load(wf: &WeightFile, dev: &B::Device) -> Result<Self, TtsError> {
        let res_idx = [1usize, 4, 7, 10];
        let down_idx = [3usize, 6, 9, 12];
        Ok(Self {
            conv0: EConv::load(wf, "encoder.encoder.layers.0.conv", 1, false, dev)?,
            res: res_idx
                .iter()
                .map(|&i| {
                    let p = format!("encoder.encoder.layers.{i}.block");
                    Ok(ResBlock {
                        c1: EConv::load(wf, &format!("{p}.1.conv"), 1, false, dev)?,
                        c3: EConv::load(wf, &format!("{p}.3.conv"), 1, false, dev)?,
                    })
                })
                .collect::<Result<_, TtsError>>()?,
            down: down_idx
                .iter()
                .zip(STRIDES)
                .map(|(&i, s)| EConv::load(wf, &format!("encoder.encoder.layers.{i}.conv"), s, false, dev))
                .collect::<Result<_, _>>()?,
            conv14: EConv::load(wf, "encoder.encoder.layers.14.conv", 1, false, dev)?,
            tlayers: (0..8)
                .map(|i| {
                    let p = format!("encoder.encoder_transformer.layers.{i}");
                    Ok(TLayer {
                        ln1_w: wf.tensor1(&format!("{p}.input_layernorm.weight"), dev)?,
                        ln1_b: wf.tensor1(&format!("{p}.input_layernorm.bias"), dev)?,
                        wq: wf.linear_t(&format!("{p}.self_attn.q_proj.weight"), dev)?,
                        wk: wf.linear_t(&format!("{p}.self_attn.k_proj.weight"), dev)?,
                        wv: wf.linear_t(&format!("{p}.self_attn.v_proj.weight"), dev)?,
                        wo: wf.linear_t(&format!("{p}.self_attn.o_proj.weight"), dev)?,
                        ls1: wf.tensor1(&format!("{p}.self_attn_layer_scale.scale"), dev)?,
                        ln2_w: wf.tensor1(&format!("{p}.post_attention_layernorm.weight"), dev)?,
                        ln2_b: wf.tensor1(&format!("{p}.post_attention_layernorm.bias"), dev)?,
                        fc1: wf.linear_t(&format!("{p}.mlp.fc1.weight"), dev)?,
                        fc2: wf.linear_t(&format!("{p}.mlp.fc2.weight"), dev)?,
                        ls2: wf.tensor1(&format!("{p}.mlp_layer_scale.scale"), dev)?,
                    })
                })
                .collect::<Result<_, TtsError>>()?,
            downsample: EConv::load(wf, "encoder.downsample.conv", DOWNSAMPLE_STRIDE, true, dev)?,
            sem_in_proj: proj1x1(
                wf,
                "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight",
                dev,
            )?,
            sem_cb: codebook(
                wf,
                "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook",
                dev,
            )?,
            ac_in_proj: proj1x1(
                wf,
                "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight",
                dev,
            )?,
            ac_cbs: (0..15)
                .map(|i| {
                    codebook(
                        wf,
                        &format!("encoder.quantizer.acoustic_residual_vector_quantizer.layers.{i}.codebook"),
                        dev,
                    )
                })
                .collect::<Result<_, _>>()?,
        })
    }

    fn transformer(&self, x: Tensor<B, 3>, dev: &B::Device) -> Tensor<B, 3> {
        let heads = 8usize;
        let hd = 64usize;
        let [_, t, _] = x.dims();

        let half = hd / 2;
        let mut cos_v = Vec::with_capacity(t * half);
        let mut sin_v = Vec::with_capacity(t * half);
        for p in 0..t {
            for i in 0..half {
                let f = 1.0 / 10000f64.powf(2.0 * i as f64 / hd as f64);
                let a = p as f64 * f;
                cos_v.push(a.cos() as f32);
                sin_v.push(a.sin() as f32);
            }
        }
        let cos: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(cos_v, [1, 1, t, half]), dev);
        let sin: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(sin_v, [1, 1, t, half]), dev);

        let mask_vals: Vec<f32> = (0..t)
            .flat_map(|i| {
                (0..t).map(move |j| {
                    if j > i || j + 250 < i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask: Tensor<B, 4> =
            Tensor::from_data(burn::tensor::TensorData::new(mask_vals, [1, 1, t, t]), dev);

        // Interleaved RoPE: pairs (x[2i], x[2i+1]) rotated by angle p·f_i.
        let rope_i = |x: Tensor<B, 4>| -> Tensor<B, 4> {
            let [b, hh, s, d] = x.dims();
            let xr = x.reshape([b, hh, s, d / 2, 2]);
            let a = xr
                .clone()
                .slice([0..b, 0..hh, 0..s, 0..d / 2, 0..1])
                .reshape([b, hh, s, d / 2]);
            let bb = xr
                .slice([0..b, 0..hh, 0..s, 0..d / 2, 1..2])
                .reshape([b, hh, s, d / 2]);
            let ra = a.clone() * cos.clone() - bb.clone() * sin.clone();
            let rb = a * sin.clone() + bb * cos.clone();
            Tensor::stack::<5>(vec![ra, rb], 4).reshape([b, hh, s, d])
        };

        let mut h = x;
        for l in &self.tlayers {
            let n = layernorm(h.clone(), &l.ln1_w, &l.ln1_b);
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
            let q = rope_i(q);
            let k = rope_i(k);
            let att = softmax(q.matmul(k.swap_dims(2, 3)) * 0.125 + mask.clone(), 3);
            let out = att.matmul(v).swap_dims(1, 2).reshape([1, t, heads * hd]);
            let out = out.matmul(l.wo.clone().unsqueeze());
            h = h + out * l.ls1.clone().unsqueeze();

            let n2 = layernorm(h.clone(), &l.ln2_w, &l.ln2_b);
            let mlp = gelu(n2.matmul(l.fc1.clone().unsqueeze())).matmul(l.fc2.clone().unsqueeze());
            h = h + mlp * l.ls2.clone().unsqueeze();
        }
        h
    }

    fn rvq_encode(
        &self,
        y: Tensor<B, 3>,
        codebooks: &[&Tensor<B, 2>],
    ) -> Result<Vec<Vec<u32>>, TtsError> {
        let [_, t, d] = y.dims();
        let mut residual = y;
        let mut out: Vec<Vec<u32>> = Vec::new();
        for cb in codebooks {
            let e = (*cb).clone();
            let [nc, _] = e.dims();
            let c2 = e.clone().powf_scalar(2.0).sum_dim(1).reshape([nc]) / 2.0;
            let dot = residual
                .clone()
                .reshape([t, d])
                .matmul(e.clone().swap_dims(0, 1));
            let scores = dot - c2.unsqueeze::<2>();
            let idx = scores.argmax(1);
            let codes: Vec<i32> = idx
                .clone()
                .into_data()
                .to_vec()
                .map_err(|e| TtsError::Gpu(format!("rvq code readback: {e:?}")))?;
            let sel: Tensor<B, 2> = e.select(0, idx.reshape([t]));
            residual = residual - sel.reshape([1, t, d]);
            out.push(codes.iter().map(|&c| c as u32).collect());
        }
        Ok(out)
    }

    /// 24 kHz mono [-1,1] → [T,16] codes (semantic first).
    pub fn encode(&self, samples: &[f32], dev: &B::Device) -> Result<Vec<Vec<u32>>, TtsError> {
        let n = samples.len();
        if n < SAMPLES_PER_FRAME {
            return Err(TtsError::ReferenceTooShort {
                samples_ms: n as f64 * 1000.0 / SR,
                min_ms: SAMPLES_PER_FRAME as f64 * 1000.0 / SR,
            });
        }
        let x: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(samples.to_vec(), [1, 1, n]),
            dev,
        );
        let mut h = self.conv0.forward(x);
        for (r, down) in self.res.iter().zip(&self.down) {
            let y = r.c3.forward(elu(r.c1.forward(elu(h.clone()))));
            h = h + y;
            h = elu(h);
            h = down.forward(h);
        }
        h = elu(h);
        h = self.conv14.forward(h);

        let h = self.transformer(h.swap_dims(1, 2), dev).swap_dims(1, 2);
        let h = self.downsample.forward(h);
        let h = h.swap_dims(1, 2);

        let sem_y = h.clone().matmul(self.sem_in_proj.clone().unsqueeze());
        let sem = self.rvq_encode(sem_y, &[&self.sem_cb])?;
        let ac_y = h.matmul(self.ac_in_proj.clone().unsqueeze());
        let ac_refs: Vec<&Tensor<B, 2>> = self.ac_cbs.iter().collect();
        let ac = self.rvq_encode(ac_y, &ac_refs)?;

        let t = sem[0].len();
        Ok((0..t)
            .map(|i| {
                let mut f = vec![sem[0][i]];
                for a in &ac {
                    f.push(a[i]);
                }
                f
            })
            .collect())
    }
}
