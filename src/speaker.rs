use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use crate::error::TtsError;
use crate::weights::WeightFile;

struct Tdnn<B: Backend> {
    w: Tensor<B, 3>,
    b: Tensor<B, 1>,
    dilation: usize,
    kernel: usize,
}

impl<B: Backend> Tdnn<B> {
    fn load(
        wf: &WeightFile,
        prefix: &str,
        dilation: usize,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        let w = wf.tensor3(&format!("{prefix}.weight"), dev)?;
        let kernel = w.dims()[2];
        Ok(Self {
            w,
            b: wf.tensor1(&format!("{prefix}.bias"), dev)?,
            dilation,
            kernel,
        })
    }

    /// Frames the reflect padding needs on its right side, plus the edge sample it mirrors around.
    fn min_frames(&self) -> usize {
        let total = self.dilation * (self.kernel - 1);
        total - total / 2 + 1
    }

    fn conv(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = reflect_pad(x, self.dilation * (self.kernel - 1));
        conv1d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvOptions::new([1], [0], [self.dilation], 1),
        )
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(self.conv(x))
    }
}

fn reflect_pad<B: Backend>(x: Tensor<B, 3>, total: usize) -> Tensor<B, 3> {
    if total == 0 {
        return x;
    }
    let left = total / 2;
    let right = total - left;
    let [b, c, t] = x.dims();
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(x.clone().slice([0..b, 0..c, 1..left + 1]).flip([2]));
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.slice([0..b, 0..c, t - 1 - right..t - 1]).flip([2]));
    }
    Tensor::cat(parts, 2)
}

struct SeRes2Net<B: Backend> {
    tdnn1: Tdnn<B>,
    res2: Vec<Tdnn<B>>,
    tdnn2: Tdnn<B>,
    se1_w: Tensor<B, 3>,
    se1_b: Tensor<B, 1>,
    se2_w: Tensor<B, 3>,
    se2_b: Tensor<B, 1>,
}

impl<B: Backend> SeRes2Net<B> {
    fn load(
        wf: &WeightFile,
        prefix: &str,
        dilation: usize,
        dev: &B::Device,
    ) -> Result<Self, TtsError> {
        Ok(Self {
            tdnn1: Tdnn::load(wf, &format!("{prefix}.tdnn1.conv"), 1, dev)?,
            res2: (0..7)
                .map(|i| {
                    Tdnn::load(
                        wf,
                        &format!("{prefix}.res2net_block.blocks.{i}.conv"),
                        dilation,
                        dev,
                    )
                })
                .collect::<Result<_, _>>()?,
            tdnn2: Tdnn::load(wf, &format!("{prefix}.tdnn2.conv"), 1, dev)?,
            se1_w: wf.tensor3(&format!("{prefix}.se_block.conv1.weight"), dev)?,
            se1_b: wf.tensor1(&format!("{prefix}.se_block.conv1.bias"), dev)?,
            se2_w: wf.tensor3(&format!("{prefix}.se_block.conv2.weight"), dev)?,
            se2_b: wf.tensor1(&format!("{prefix}.se_block.conv2.bias"), dev)?,
        })
    }

    fn min_frames(&self) -> usize {
        [&self.tdnn1, &self.tdnn2]
            .into_iter()
            .chain(&self.res2)
            .map(Tdnn::min_frames)
            .max()
            .unwrap_or(1)
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.tdnn1.forward(x);
        let [b, c, t] = h.dims();
        let chunk = c / 8;
        let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(8);
        let mut prev: Option<Tensor<B, 3>> = None;
        for i in 0..8 {
            let xi = h.clone().slice([0..b, i * chunk..(i + 1) * chunk, 0..t]);
            let yi = match (i, &prev) {
                (0, _) => xi,
                (1, _) => self.res2[0].forward(xi),
                (_, Some(p)) => self.res2[i - 1].forward(xi + p.clone()),
                _ => unreachable!(),
            };
            if i > 0 {
                prev = Some(yi.clone());
            }
            outs.push(yi);
        }
        let h = Tensor::cat(outs, 1);
        let h = self.tdnn2.forward(h);

        let s = h.clone().mean_dim(2);
        let s = conv1d(
            s,
            self.se1_w.clone(),
            Some(self.se1_b.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        let s = relu(s);
        let s = conv1d(
            s,
            self.se2_w.clone(),
            Some(self.se2_b.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        let s = sigmoid(s);
        h * s + residual
    }
}

pub struct SpeakerEncoder<B: Backend> {
    block0: Tdnn<B>,
    blocks: Vec<SeRes2Net<B>>,
    mfa: Tdnn<B>,
    asp_tdnn: Tdnn<B>,
    asp_w: Tensor<B, 3>,
    asp_b: Tensor<B, 1>,
    fc_w: Tensor<B, 3>,
    fc_b: Tensor<B, 1>,
    n_mels: usize,
    min_frames: usize,
}

impl<B: Backend> SpeakerEncoder<B> {
    pub fn load(wf: &WeightFile, dev: &B::Device) -> Result<Self, TtsError> {
        let block0 = Tdnn::load(wf, "speaker_encoder.blocks.0.conv", 1, dev)?;
        let blocks: Vec<SeRes2Net<B>> = (1..4)
            .map(|i| SeRes2Net::load(wf, &format!("speaker_encoder.blocks.{i}"), i + 1, dev))
            .collect::<Result<_, _>>()?;
        let mfa = Tdnn::load(wf, "speaker_encoder.mfa.conv", 1, dev)?;
        let asp_tdnn = Tdnn::load(wf, "speaker_encoder.asp.tdnn.conv", 1, dev)?;
        let n_mels = block0.w.dims()[1];
        let min_frames = [&block0, &mfa, &asp_tdnn]
            .into_iter()
            .map(Tdnn::min_frames)
            .chain(blocks.iter().map(SeRes2Net::min_frames))
            .max()
            .unwrap_or(1);
        Ok(Self {
            block0,
            blocks,
            mfa,
            asp_tdnn,
            asp_w: wf.tensor3("speaker_encoder.asp.conv.weight", dev)?,
            asp_b: wf.tensor1("speaker_encoder.asp.conv.bias", dev)?,
            fc_w: wf.tensor3("speaker_encoder.fc.weight", dev)?,
            fc_b: wf.tensor1("speaker_encoder.fc.bias", dev)?,
            n_mels,
            min_frames,
        })
    }

    /// Smallest mel frame count `encode` accepts.
    pub fn min_frames(&self) -> usize {
        self.min_frames
    }

    /// mel: [n_mels][frames] natural-log mel. Returns unnormalized [enc_dim].
    pub fn encode(&self, mel: &[Vec<f32>], dev: &B::Device) -> Result<Vec<f32>, TtsError> {
        let n_mels = mel.len();
        if n_mels != self.n_mels {
            return Err(TtsError::InvalidConfig(format!(
                "speaker encoder expects {} mel bands, got {n_mels}",
                self.n_mels
            )));
        }
        let t = mel[0].len();
        if mel.iter().any(|r| r.len() != t) {
            return Err(TtsError::InvalidPrompt(
                "mel rows have unequal lengths".into(),
            ));
        }
        if t < self.min_frames {
            return Err(TtsError::ReferenceTooShort {
                samples_ms: t as f64 * crate::mel::FRAME_MS,
                min_ms: self.min_frames as f64 * crate::mel::FRAME_MS,
            });
        }
        let flat: Vec<f32> = mel.iter().flat_map(|r| r.iter().copied()).collect();
        let x: Tensor<B, 3> =
            Tensor::from_data(burn::tensor::TensorData::new(flat, [1, n_mels, t]), dev);

        let h = self.block0.forward(x);
        let mut feats = Vec::new();
        let mut cur = h;
        for blk in &self.blocks {
            cur = blk.forward(cur);
            feats.push(cur.clone());
        }
        let h = Tensor::cat(feats, 1);
        let h = self.mfa.forward(h);

        let [b, c, tt] = h.dims();
        let mean = h.clone().mean_dim(2);
        let var = (h.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2);
        let std = (var + 1e-5).sqrt();
        let ctx = Tensor::cat(
            vec![h.clone(), mean.expand([b, c, tt]), std.expand([b, c, tt])],
            1,
        );
        let a = self.asp_tdnn.forward(ctx);
        let a = tanh(a);
        let a = conv1d(
            a,
            self.asp_w.clone(),
            Some(self.asp_b.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        let a = softmax(a, 2);
        let w_mean = (h.clone() * a.clone()).sum_dim(2);
        let w_std = ((h - w_mean.clone().expand([b, c, tt])).powf_scalar(2.0) * a)
            .sum_dim(2)
            .add_scalar(1e-5)
            .sqrt();
        let pooled = Tensor::cat(vec![w_mean, w_std], 1);
        let out = conv1d(
            pooled,
            self.fc_w.clone(),
            Some(self.fc_b.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        out.cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec()
            .map_err(|e| TtsError::Gpu(format!("speaker embedding readback: {e:?}")))
    }
}
