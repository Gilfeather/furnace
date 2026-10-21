//! Tensors as the server sees them: named inputs and outputs of a model, with a datatype and a
//! shape, held in host memory between the wire formats and Burn.
//!
//! Every request reaches a model as [`HostTensor`]s, one per input, batch first. The flat
//! `Vec<f32>` entry points of `Model` (`predict*`) wrap their buffer as one such tensor without
//! copying it.

use burn::tensor::{Device, Int, IntDType, Tensor, TensorData};
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
/// What [`BurnModel::run`](crate::BurnModel::run) takes and returns. Turn one into a Burn tensor
/// with [`HostTensor::into_float`] or [`HostTensor::into_int`], and back with
/// [`HostTensor::from_float`] or [`HostTensor::from_int`].
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
        let mut parts = parts.into_iter();
        let mut joined = parts
            .next()
            .ok_or_else(|| ModelError::InferenceFailed("nothing to join".to_string()))?;
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

    /// A Burn float tensor of rank `D`
    pub fn into_float<const D: usize>(self, device: &Device) -> Result<Tensor<D>> {
        let shape = self.checked_shape::<D>()?;
        match self.values {
            Values::F32(values) => Ok(Tensor::from_data(TensorData::new(values, shape), device)),
            Values::I64(_) => Err(type_mismatch("a float", "integers")),
        }
    }

    /// A Burn float tensor of shape `[batch, elements per item]`, whatever the per-item rank
    pub(crate) fn into_float_flat(self, device: &Device) -> Result<Tensor<2>> {
        let per_item = self.shape.get(1..).map_or(0, |item| item.iter().product());
        HostTensor {
            shape: vec![self.batch(), per_item],
            values: self.values,
        }
        .into_float(device)
    }

    /// A Burn integer tensor of rank `D`. The backend may store integers in fewer than 64 bits
    /// (the CPU backend uses `i32`); values that do not fit are refused rather than truncated.
    pub fn into_int<const D: usize>(self, device: &Device) -> Result<Tensor<D, Int>> {
        let shape = self.checked_shape::<D>()?;
        let Values::I64(values) = self.values else {
            return Err(type_mismatch("an integer", "floats"));
        };
        let dtype = int_dtype(device);
        let (min, max) = range_of(dtype);
        let outside = |pos: usize, value: i64| -> crate::error::FurnaceError {
            ModelError::InvalidInput(format!(
                "integer {value} at index {pos} is outside {min}..={max}, the range of this \
                 backend's integers"
            ))
            .into()
        };
        let data = if dtype == IntDType::I32 {
            // Narrow here, checking each value on the way, rather than scan first and leave the
            // same conversion to Burn
            let mut narrowed = Vec::with_capacity(values.len());
            for (pos, &value) in values.iter().enumerate() {
                narrowed.push(i32::try_from(value).map_err(|_| outside(pos, value))?);
            }
            TensorData::new(narrowed, shape)
        } else {
            if let Some(pos) = values.iter().position(|v| !(min..=max).contains(v)) {
                return Err(outside(pos, values[pos]));
            }
            TensorData::new(values, shape)
        };
        Ok(Tensor::from_data(data, device))
    }

    fn checked_shape<const D: usize>(&self) -> Result<[usize; D]> {
        self.shape.as_slice().try_into().map_err(|_| {
            ModelError::InvalidInput(format!(
                "expected a tensor of rank {D}, got shape {:?}",
                self.shape
            ))
            .into()
        })
    }

    /// A Burn float tensor's shape and elements, read back to the host
    pub fn from_float<const D: usize>(tensor: Tensor<D>) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let data = tensor.into_data();
        let values = match data.as_slice::<f32>() {
            Ok(_) => data.try_into_vec::<f32>().map_err(read_error)?,
            Err(_) => data.iter::<f32>().collect(),
        };
        Ok(Self {
            shape,
            values: Values::F32(values),
        })
    }

    /// A Burn integer tensor's shape and elements, read back to the host as `i64`
    pub fn from_int<const D: usize>(tensor: Tensor<D, Int>) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let data = tensor.into_data();
        let values = match data.as_slice::<i64>() {
            Ok(_) => data.try_into_vec::<i64>().map_err(read_error)?,
            Err(_) => data.iter::<i64>().collect(),
        };
        Ok(Self {
            shape,
            values: Values::I64(values),
        })
    }
}

/// The range of the integers `device` stores by default
pub fn int_range(device: &Device) -> (i64, i64) {
    range_of(int_dtype(device))
}

/// The integer type `device` gives a new tensor. Read from Burn's per-device settings, a lookup
/// rather than a probe tensor, so it is cheap on every request
fn int_dtype(device: &Device) -> IntDType {
    device.settings().int_dtype
}

fn range_of(dtype: IntDType) -> (i64, i64) {
    match dtype {
        IntDType::I32 => (i32::MIN.into(), i32::MAX.into()),
        IntDType::I16 => (i16::MIN.into(), i16::MAX.into()),
        IntDType::I8 => (i8::MIN.into(), i8::MAX.into()),
        IntDType::U32 => (0, u32::MAX.into()),
        IntDType::U16 => (0, u16::MAX.into()),
        IntDType::U8 => (0, u8::MAX.into()),
        IntDType::I64 | IntDType::U64 => (i64::MIN, i64::MAX),
    }
}

/// A model's inputs as an array, for `run` to destructure: `let [ids, mask] = take_inputs(inputs)?;`
pub(crate) fn take_inputs<const N: usize>(inputs: Vec<HostTensor>) -> Result<[HostTensor; N]> {
    let got = inputs.len();
    inputs
        .try_into()
        .map_err(|_| ModelError::InvalidInput(format!("expected {N} input(s), got {got}")).into())
}

fn type_mismatch(expected: &str, got: &str) -> crate::error::FurnaceError {
    ModelError::InvalidDataType(format!("expected {expected} tensor, got {got}")).into()
}

fn read_error(e: impl std::fmt::Debug) -> crate::error::FurnaceError {
    ModelError::InferenceFailed(format!("could not read the output: {e:?}")).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;

    #[test]
    fn round_trips_through_burn() {
        let device = BackendKind::Cpu.device();
        let floats = HostTensor {
            shape: vec![2, 3],
            values: Values::F32(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        };
        let back = HostTensor::from_float(floats.clone().into_float::<2>(&device).unwrap());
        assert_eq!(back.unwrap(), floats);

        let ints = HostTensor {
            shape: vec![2, 2],
            values: Values::I64(vec![101, 2023, -7, 30_522]),
        };
        let tensor = ints.clone().into_int::<2>(&device).unwrap();
        assert_eq!(HostTensor::from_int(tensor).unwrap(), ints);

        // A value the backend's integers cannot hold is refused, not truncated (or a panic)
        let (_, max) = int_range(&device);
        if max < i64::MAX {
            let huge = HostTensor {
                shape: vec![1],
                values: Values::I64(vec![max + 1]),
            };
            let err = huge.into_int::<1>(&device).unwrap_err().to_string();
            assert!(err.contains("outside"), "{err}");
        }

        assert!(floats.clone().into_int::<2>(&device).is_err());
        assert!(floats.into_float::<3>(&device).is_err());
    }

    #[test]
    fn inputs_by_position_and_flattened() {
        let device = BackendKind::Cpu.device();
        let tensor = HostTensor {
            shape: vec![2, 1, 3],
            values: Values::F32(vec![0.0; 6]),
        };
        let [only] = take_inputs::<1>(vec![tensor.clone()]).unwrap();
        assert_eq!(only.into_float_flat(&device).unwrap().dims(), [2, 3]);
        let err = take_inputs::<2>(vec![tensor]).unwrap_err().to_string();
        assert!(err.contains("expected 2 input(s), got 1"), "{err}");
    }

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
