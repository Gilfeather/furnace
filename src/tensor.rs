//! Tensors as the server sees them: named inputs and outputs of a model, with a datatype and a
//! shape, held in host memory between the wire formats and Burn.
//!
//! Most models take one `f32` tensor and return one (`Model::is_simple`), and those keep the
//! flat `Vec<f32>` path with dynamic batching. Everything else, e.g. BERT (several `int64`
//! inputs) or YOLO (several outputs), goes through [`HostTensor`]s.

use burn::tensor::{Device, Int, Tensor, TensorData};
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
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A tensor in host memory: full shape (batch first) and row-major elements
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

    /// A Burn integer tensor of rank `D`. The backend may store integers in fewer than 64 bits
    /// (the CPU backend uses `i32`); values that do not fit are refused rather than truncated.
    pub fn into_int<const D: usize>(self, device: &Device) -> Result<Tensor<D, Int>> {
        let shape = self.checked_shape::<D>()?;
        let Values::I64(values) = self.values else {
            return Err(type_mismatch("an integer", "floats"));
        };
        let (min, max) = int_range(device);
        if let Some(pos) = values.iter().position(|v| !(min..=max).contains(v)) {
            return Err(ModelError::InvalidInput(format!(
                "integer {} at index {pos} is outside {min}..={max}, the range of this backend's \
                 integers",
                values[pos]
            ))
            .into());
        }
        Ok(Tensor::from_data(TensorData::new(values, shape), device))
    }

    fn checked_shape<const D: usize>(&self) -> Result<[usize; D]> {
        self.shape.as_slice().try_into().map_err(|_| {
            ModelError::InputValidation {
                expected: vec![D],
                actual: self.shape.clone(),
            }
            .into()
        })
    }

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

/// The range of the integers `device` stores, found from the dtype it gives a new tensor
pub fn int_range(device: &Device) -> (i64, i64) {
    use burn::tensor::DType;
    let probe = Tensor::<1, Int>::from_data(TensorData::new(vec![0i64], [1]), device);
    match probe.dtype() {
        DType::I32 => (i32::MIN.into(), i32::MAX.into()),
        DType::I16 => (i16::MIN.into(), i16::MAX.into()),
        DType::I8 => (i8::MIN.into(), i8::MAX.into()),
        DType::U32 => (0, u32::MAX.into()),
        DType::U16 => (0, u16::MAX.into()),
        DType::U8 => (0, u8::MAX.into()),
        _ => (i64::MIN, i64::MAX),
    }
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
