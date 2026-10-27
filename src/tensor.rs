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

/// Why two tensors cannot be joined along the batch dimension
#[derive(Debug, PartialEq)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
enum Unjoinable {
    /// Either has no batch dimension, or their items' shapes differ
    Shapes,
    /// Their batches together do not fit in a `usize`
    TooMany,
}

/// The batch of tensors of shapes `a` and `b` joined: each needs a batch dimension and the
/// same per-item shape. Its contract is what the proof of [`HostTensor::concat_batch`] builds
/// on.
#[cfg_attr(kani, kani::ensures(|joined: &std::result::Result<usize, Unjoinable>| match joined {
    Ok(batch) => {
        same_items(a, b) && *batch as u128 == a[0] as u128 + b[0] as u128
    }
    Err(Unjoinable::Shapes) => !same_items(a, b),
    Err(Unjoinable::TooMany) => {
        same_items(a, b) && a[0] as u128 + b[0] as u128 > usize::MAX as u128
    }
}))]
fn joined_batch(a: &[usize], b: &[usize]) -> std::result::Result<usize, Unjoinable> {
    match (a.split_first(), b.split_first()) {
        (Some((&n, item)), Some((&m, other))) if item == other => {
            n.checked_add(m).ok_or(Unjoinable::TooMany)
        }
        _ => Err(Unjoinable::Shapes),
    }
}

/// Whether shapes `a` and `b` both have a batch dimension and the same dimensions after it,
/// compared one by one (for the contracts: a slice comparison is a `memcmp` Kani unwinds byte
/// by byte)
#[cfg(kani)]
fn same_items(a: &[usize], b: &[usize]) -> bool {
    !a.is_empty() && a.len() == b.len() && (1..a.len()).all(|i| a[i] == b[i])
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

    /// Items `start..end` along the batch dimension.
    ///
    /// Panics unless the shape has a batch dimension, `start..end` lies within it and the
    /// values are as many as the shape holds (as [`Model`](crate::Model) checks of every
    /// tensor it slices).
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
        let mut total = 0;
        for part in &parts {
            total += part.values.len();
        }
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
            // The parts are an engine's outputs, not checked yet: without a batch dimension
            // there is nothing to join along
            joined.shape[0] = match joined_batch(&joined.shape, &part.shape) {
                Ok(batch) => batch,
                Err(Unjoinable::Shapes) => {
                    return Err(ModelError::InferenceFailed(format!(
                        "chunks of one batch returned shapes {:?} and {:?}",
                        joined.shape, part.shape
                    ))
                    .into());
                }
                Err(Unjoinable::TooMany) => {
                    return Err(ModelError::InferenceFailed(
                        "chunks of one batch returned more items than fit in a usize".to_string(),
                    )
                    .into());
                }
            };
            match (&mut joined.values, part.values) {
                (Values::F32(a), Values::F32(b)) => a.extend_from_slice(&b),
                (Values::I64(a), Values::I64(b)) => a.extend_from_slice(&b),
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

/// Kani proofs (`cargo kani`): the element counts of any shape, and splitting and joining
/// tensors along the batch dimension
#[cfg(kani)]
mod verification {
    use super::*;

    /// `element_count` of any shape of up to four dimensions, each of any size, never
    /// overflows: it is the exact product when that fits in a `usize`, and `None` only when the
    /// product of the dimensions up to one of them does not
    #[kani::proof]
    #[kani::unwind(6)]
    fn element_count_is_the_exact_product() {
        let dims: [usize; 4] = kani::any();
        let shape = &dims[..kani::any_where(|&len: &usize| len <= 4)];
        // The product in 128 bits: no overflow while every partial product fits in 64
        let mut exact: u128 = 1;
        let mut overflows = false;
        for &d in shape {
            exact *= d as u128;
            if exact > usize::MAX as u128 {
                overflows = true;
                break;
            }
        }
        match element_count(shape) {
            Some(n) => {
                assert!(!overflows);
                assert_eq!(n as u128, exact);
            }
            None => {
                assert!(overflows);
            }
        }
    }

    /// A tensor whose values are as many as its shape holds, with a batch dimension
    fn any_tensor(max_batch: usize) -> HostTensor {
        let batch = kani::any_where(|&b: &usize| b <= max_batch);
        let item = kani::any_where(|&d: &usize| d <= 2);
        let rank2: bool = kani::any();
        let shape = if rank2 {
            vec![batch, item]
        } else {
            vec![batch]
        };
        let len = element_count(&shape).unwrap();
        // Any values, copied from a slice: `vec![x; len]` fills in a loop Kani unwinds slowly
        let values = if kani::any() {
            let values: [f32; 4] = kani::any();
            Values::F32(values[..len].to_vec())
        } else {
            let values: [i64; 4] = kani::any();
            Values::I64(values[..len].to_vec())
        };
        HostTensor { shape, values }
    }

    /// `slice_batch` of any range within the batch takes exactly its items' values
    #[kani::proof]
    #[kani::unwind(8)]
    fn slice_batch_stays_in_bounds() {
        let tensor = any_tensor(3);
        let end = kani::any_where(|&e: &usize| e <= tensor.batch());
        let start = kani::any_where(|&s: &usize| s <= end);
        let part = tensor.slice_batch(start, end);
        assert_eq!(part.batch(), end - start);
        // Element by element: a slice comparison is a `memcmp` Kani unwinds byte by byte
        assert_eq!(part.shape.len(), tensor.shape.len());
        assert_eq!(part.shape.get(1), tensor.shape.get(1));
        assert_eq!(element_count(&part.shape), Some(part.values.len()));
    }

    /// `joined_batch` of any two shapes of up to three dimensions is its contract
    #[kani::proof_for_contract(joined_batch)]
    #[kani::unwind(26)]
    fn joined_batch_keeps_its_contract() {
        let a: [usize; 3] = kani::any();
        let b: [usize; 3] = kani::any();
        let a = &a[..kani::any_where(|&len: &usize| len <= 3)];
        let b = &b[..kani::any_where(|&len: &usize| len <= 3)];
        let _ = joined_batch(a, b);
    }

    /// `concat_batch` of up to three parts, even of other ranks, types or shapes or none at
    /// all, never panics; what it joins has every part's items and values
    #[kani::proof]
    #[kani::stub_verified(joined_batch)]
    #[kani::unwind(8)]
    fn concat_batch_never_panics() {
        let count = kani::any_where(|&n: &usize| n <= 3);
        // Built and counted in plain loops: iterator adapters unwind into far more paths for
        // Kani
        let mut parts = Vec::with_capacity(3);
        let mut items = Some(0);
        let mut values = 0;
        for _ in 0..count {
            let mut part = any_tensor(2);
            // A part as an engine may return it: without a batch dimension
            if kani::any() {
                part.shape.clear();
            }
            items = match (items, part.shape.first()) {
                (Some(sum), Some(&batch)) => Some(sum + batch),
                _ => None,
            };
            values += part.values.len();
            parts.push(part);
        }
        if let Ok(joined) = HostTensor::concat_batch(parts) {
            assert_eq!(joined.values.len(), values);
            if count > 1 {
                assert_eq!(joined.shape.first().copied(), items);
            }
        }
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
    fn parts_without_a_batch_dimension_are_not_joined() {
        // An engine's outputs, joined before they are checked: indexing their shapes panicked
        let scalar = HostTensor {
            shape: vec![],
            values: Values::F32(vec![1.0]),
        };
        let row = HostTensor {
            shape: vec![1],
            values: Values::F32(vec![2.0]),
        };
        assert!(HostTensor::concat_batch(vec![row.clone(), scalar.clone()]).is_err());
        assert!(HostTensor::concat_batch(vec![scalar, row]).is_err());
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
