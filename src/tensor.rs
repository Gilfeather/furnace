//! Tensors as the server sees them: named inputs and outputs of a model, with a datatype and a
//! shape, held in host memory between the wire formats and the engine that runs the model.
//!
//! Every request reaches a model as [`HostTensor`]s, one per input, batch first. The flat
//! `Vec<f32>` entry points of `Model` (`predict*`) wrap their buffer as one such tensor without
//! copying it.

use serde::{Deserialize, Serialize};

use crate::error::{ModelError, Result};

/// Element types a model's inputs and outputs can have
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Fp32,
    Int64,
    Int32,
}

impl DataType {
    /// The Open Inference Protocol name: `FP32`, `INT64`, `INT32`
    pub fn v2_name(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Int64 => "INT64",
            Self::Int32 => "INT32",
        }
    }

    /// The type an Open Inference Protocol name stands for; `None` for one the server does not
    /// take
    pub fn from_v2_name(name: &str) -> Option<Self> {
        match name {
            "FP32" => Some(Self::Fp32),
            "INT64" => Some(Self::Int64),
            "INT32" => Some(Self::Int32),
            _ => None,
        }
    }

    /// Bytes per element in the raw (binary) format
    pub fn size(self) -> usize {
        match self {
            Self::Fp32 | Self::Int32 => 4,
            Self::Int64 => 8,
        }
    }

    /// Whether elements are integers (held as `i64` in [`Values::I64`])
    pub fn is_int(self) -> bool {
        matches!(self, Self::Int64 | Self::Int32)
    }
}

/// A model input or output: its name, element type and per-item shape (without the batch
/// dimension; `None` where the graph leaves a dimension open)
#[derive(Debug, Clone, PartialEq)]
pub struct IoSpec {
    pub name: String,
    pub dtype: DataType,
    pub shape: Vec<Option<usize>>,
}

impl IoSpec {
    /// A tensor of `dtype` with per-item `shape` (`None` for a dimension each request fixes)
    pub fn new(name: &str, dtype: DataType, shape: &[Option<usize>]) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// An `f32` tensor with a fixed per-item shape
    pub fn fp32(name: &str, shape: &[usize]) -> Self {
        Self {
            name: name.to_string(),
            dtype: DataType::Fp32,
            shape: shape.iter().map(|&d| Some(d)).collect(),
        }
    }

    /// The per-item shape if every dimension is fixed
    pub fn static_shape(&self) -> Option<Vec<usize>> {
        self.shape.iter().copied().collect()
    }

    /// For metadata: the full shape with `-1` for the batch and for open dimensions
    pub fn v2_shape(&self) -> Vec<i64> {
        std::iter::once(-1)
            .chain(self.shape.iter().map(|d| d.map_or(-1, |d| d as i64)))
            .collect()
    }
}

/// The legacy single-tensor description in `GET /model/info` (`input_spec`, `output_spec`):
/// per-item shape, open dimensions as 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    /// `float32`, `int64` or `int32`
    pub dtype: String,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}

/// Description of a model input or output in `GET /model/info`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    /// `FP32`, `INT64` or `INT32`
    pub datatype: String,
    /// Full shape; `-1` for the batch dimension and for dimensions fixed per request
    pub shape: Vec<i64>,
}

impl From<&IoSpec> for TensorInfo {
    fn from(spec: &IoSpec) -> Self {
        Self {
            name: spec.name.clone(),
            datatype: spec.dtype.v2_name().to_string(),
            shape: spec.v2_shape(),
        }
    }
}

/// The number of elements of a tensor of `shape`; `None` when it does not fit in a `usize`,
/// as a client's shape may not
pub(crate) fn element_count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
}

/// Tensor elements in host memory
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl Values {
    /// Number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::I64(v) => v.len(),
        }
    }

    /// Whether there are no elements
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A tensor in host memory: full shape (batch first) and row-major elements.
///
/// What [`TensorModel::run`](crate::TensorModel::run) takes and returns. furnace-burn turns one
/// into a Burn tensor and back (its `HostTensorExt`).
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub values: Values,
}

impl HostTensor {
    /// Items along the batch dimension; 0 for a tensor without one (an empty shape), which no
    /// model accepts
    pub fn batch(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Items `start..end` along the batch dimension
    pub fn slice_batch(&self, start: usize, end: usize) -> Self {
        let per_item: usize = self.shape[1..].iter().product();
        let range = start * per_item..end * per_item;
        let mut shape = self.shape.clone();
        shape[0] = end - start;
        Self {
            shape,
            values: match &self.values {
                Values::F32(v) => Values::F32(v[range].to_vec()),
                Values::I64(v) => Values::I64(v[range].to_vec()),
            },
        }
    }

    /// Join tensors of the same type and per-item shape along the batch dimension
    pub fn concat_batch(parts: Vec<Self>) -> Result<Self> {
        let total: usize = parts.iter().map(|part| part.values.len()).sum();
        let mut parts = parts.into_iter();
        let mut joined = parts
            .next()
            .ok_or_else(|| ModelError::InferenceFailed("nothing to join".to_string()))?;
        let rest = total - joined.values.len();
        match &mut joined.values {
            Values::F32(values) => values.reserve(rest),
            Values::I64(values) => values.reserve(rest),
        }
        for part in parts {
            if part.shape[1..] != joined.shape[1..] {
                return Err(ModelError::InferenceFailed(format!(
                    "chunks of one batch returned shapes {:?} and {:?}",
                    joined.shape, part.shape
                ))
                .into());
            }
            joined.shape[0] += part.shape[0];
            match (&mut joined.values, part.values) {
                (Values::F32(a), Values::F32(b)) => a.extend(b),
                (Values::I64(a), Values::I64(b)) => a.extend(b),
                _ => {
                    return Err(ModelError::InferenceFailed(
                        "chunks of one batch returned different types".to_string(),
                    )
                    .into());
                }
            }
        }
        Ok(joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_split_and_join() {
        let tensor = HostTensor {
            shape: vec![3, 2],
            values: Values::I64(vec![1, 2, 3, 4, 5, 6]),
        };
        let parts = vec![tensor.slice_batch(0, 1), tensor.slice_batch(1, 3)];
        assert_eq!(parts[1].shape, [2, 2]);
        assert_eq!(HostTensor::concat_batch(parts).unwrap(), tensor);
    }

    #[test]
    fn metadata_shapes() {
        let spec = IoSpec {
            name: "ids".to_string(),
            dtype: DataType::Int64,
            shape: vec![Some(256), None],
        };
        assert_eq!(spec.v2_shape(), [-1, 256, -1]);
        assert_eq!(spec.static_shape(), None);
        assert_eq!(TensorInfo::from(&spec).datatype, "INT64");
        assert_eq!(DataType::from_v2_name("INT32"), Some(DataType::Int32));
        assert_eq!(DataType::from_v2_name("FP16"), None);
    }
}
