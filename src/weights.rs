use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::tensor::TensorData;
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

use crate::error::TtsError;

pub struct WeightFile {
    path: PathBuf,
    mmap: memmap2::Mmap,
    // The header is parsed once; the previous per-tensor deserialize walked a
    // ~60 KB JSON header for every one of the ~1000 tensors loaded.
    names: Vec<String>,
}

fn check_rank(name: &str, shape: &[usize], rank: usize) -> Result<(), TtsError> {
    if shape.len() == rank {
        Ok(())
    } else {
        Err(TtsError::TensorShape {
            name: name.to_string(),
            expected: format!("{rank}D"),
            got: shape.to_vec(),
        })
    }
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let path = path.as_ref().to_path_buf();
        let f = std::fs::File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                TtsError::ModelFileMissing { path: path.clone() }
            } else {
                TtsError::Io {
                    path: path.clone(),
                    source: e,
                }
            }
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| TtsError::Io {
            path: path.clone(),
            source: e,
        })?;
        let names = SafeTensors::deserialize(&mmap)
            .map_err(|e| TtsError::BadSafetensors {
                path: path.clone(),
                detail: e.to_string(),
            })?
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        Ok(Self { path, mmap, names })
    }

    fn view(&self) -> Result<SafeTensors<'_>, TtsError> {
        SafeTensors::deserialize(&self.mmap).map_err(|e| TtsError::BadSafetensors {
            path: self.path.clone(),
            detail: e.to_string(),
        })
    }

    fn to_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), TtsError> {
        let st = self.view()?;
        let t = st.tensor(name).map_err(|_| TtsError::MissingTensor {
            name: name.to_string(),
        })?;
        let shape = t.shape().to_vec();
        let raw = t.data();
        let v: Vec<f32> = match t.dtype() {
            Dtype::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            other => {
                return Err(TtsError::TensorDtype {
                    name: name.to_string(),
                    dtype: format!("{other:?}"),
                })
            }
        };
        Ok((v, shape))
    }

    /// A 2-D tensor as f16 rows on the host. Used for tables that are gathered
    /// row by row, where uploading the whole table to the device buys nothing.
    pub fn rows_f16(&self, name: &str) -> Result<HostTable, TtsError> {
        let st = self.view()?;
        let t = st.tensor(name).map_err(|_| TtsError::MissingTensor {
            name: name.to_string(),
        })?;
        let shape = t.shape().to_vec();
        check_rank(name, &shape, 2)?;
        let raw = t.data();
        let data: Vec<f16> = match t.dtype() {
            Dtype::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_f32(bf16::from_le_bytes(*c).to_f32()))
                .collect(),
            Dtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c))
                .collect(),
            Dtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f16::from_f32(f32::from_le_bytes(*c)))
                .collect(),
            other => {
                return Err(TtsError::TensorDtype {
                    name: name.to_string(),
                    dtype: format!("{other:?}"),
                })
            }
        };
        Ok(HostTable {
            data,
            rows: shape[0],
            cols: shape[1],
        })
    }

    pub fn tensor2<B: Backend>(
        &self,
        name: &str,
        dev: &B::Device,
    ) -> Result<Tensor<B, 2>, TtsError> {
        let (v, shape) = self.to_f32(name)?;
        check_rank(name, &shape, 2)?;
        Ok(Tensor::from_data(
            TensorData::new(v, [shape[0], shape[1]]),
            dev,
        ))
    }

    /// Loads a `[out, in]` linear weight and returns it transposed to `[in, out]`
    /// so forward is a plain `x.matmul(w)`.
    pub fn linear_t<B: Backend>(
        &self,
        name: &str,
        dev: &B::Device,
    ) -> Result<Tensor<B, 2>, TtsError> {
        Ok(self.tensor2::<B>(name, dev)?.swap_dims(0, 1))
    }

    pub fn tensor3<B: Backend>(
        &self,
        name: &str,
        dev: &B::Device,
    ) -> Result<Tensor<B, 3>, TtsError> {
        let (v, shape) = self.to_f32(name)?;
        check_rank(name, &shape, 3)?;
        Ok(Tensor::from_data(
            TensorData::new(v, [shape[0], shape[1], shape[2]]),
            dev,
        ))
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    pub fn try_tensor1<B: Backend>(
        &self,
        name: &str,
        dev: &B::Device,
    ) -> Result<Option<Tensor<B, 1>>, TtsError> {
        if self.has(name) {
            Ok(Some(self.tensor1(name, dev)?))
        } else {
            Ok(None)
        }
    }

    pub fn tensor1<B: Backend>(
        &self,
        name: &str,
        dev: &B::Device,
    ) -> Result<Tensor<B, 1>, TtsError> {
        let (v, shape) = self.to_f32(name)?;
        check_rank(name, &shape, 1)?;
        Ok(Tensor::from_data(TensorData::new(v, [shape[0]]), dev))
    }
}

/// A row-major f16 matrix kept in host memory.
pub struct HostTable {
    data: Vec<f16>,
    pub rows: usize,
    pub cols: usize,
}

impl HostTable {
    /// Gather `ids` as a `[len, cols]` device tensor.
    pub fn gather<B: Backend>(
        &self,
        ids: &[u32],
        dev: &B::Device,
    ) -> Result<Tensor<B, 2>, TtsError> {
        let mut out: Vec<f16> = Vec::with_capacity(ids.len() * self.cols);
        for &i in ids {
            let i = i as usize;
            if i >= self.rows {
                return Err(TtsError::InvalidPrompt(format!(
                    "text token id {i} is out of range ({} rows)",
                    self.rows
                )));
            }
            out.extend_from_slice(&self.data[i * self.cols..(i + 1) * self.cols]);
        }
        Ok(Tensor::from_data(
            TensorData::new(out, [ids.len(), self.cols]),
            dev,
        ))
    }
}
