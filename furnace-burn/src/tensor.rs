//! [`HostTensor`]s to Burn tensors and back.
//!
//! The server hands a model its inputs as `HostTensor`s and takes its outputs back as such;
//! [`HostTensorExt`] converts them at the model's edge:
//!
//! ```ignore
//! use furnace_burn::HostTensorExt;
//!
//! let ids = input.into_int::<2>(&device)?;
//! Ok(vec![HostTensor::from_float(logits)?])
//! ```

use burn::tensor::{Device, Int, IntDType, Tensor, TensorData};
use furnace::error::{FurnaceError, ModelError, Result};
use furnace::{HostTensor, Values};

/// Converting [`HostTensor`]s to Burn tensors and back. Import it to call these as
/// `HostTensor`'s own methods.
pub trait HostTensorExt: Sized {
    /// A Burn float tensor of rank `D`
    fn into_float<const D: usize>(self, device: &Device) -> Result<Tensor<D>>;

    /// A Burn integer tensor of rank `D`. The backend may store integers in fewer than 64 bits
    /// (the CPU backend uses `i32`); values that do not fit are refused rather than truncated.
    fn into_int<const D: usize>(self, device: &Device) -> Result<Tensor<D, Int>>;

    /// A Burn float tensor's shape and elements, read back to the host
    fn from_float<const D: usize>(tensor: Tensor<D>) -> Result<Self>;

    /// A Burn integer tensor's shape and elements, read back to the host as `i64`
    fn from_int<const D: usize>(tensor: Tensor<D, Int>) -> Result<Self>;
}

impl HostTensorExt for HostTensor {
    fn into_float<const D: usize>(self, device: &Device) -> Result<Tensor<D>> {
        let shape = checked_shape::<D>(&self)?;
        match self.values {
            Values::F32(values) => Ok(Tensor::from_data(TensorData::new(values, shape), device)),
            Values::I64(_) => Err(type_mismatch("a float", "integers")),
        }
    }

    fn into_int<const D: usize>(self, device: &Device) -> Result<Tensor<D, Int>> {
        let shape = checked_shape::<D>(&self)?;
        let Values::I64(values) = self.values else {
            return Err(type_mismatch("an integer", "floats"));
        };
        let dtype = int_dtype(device);
        let (min, max) = range_of(dtype);
        let outside = |pos: usize, value: i64| -> FurnaceError {
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

    fn from_float<const D: usize>(tensor: Tensor<D>) -> Result<Self> {
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

    fn from_int<const D: usize>(tensor: Tensor<D, Int>) -> Result<Self> {
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

/// A Burn float tensor of shape `[batch, elements per item]`, whatever the per-item rank
pub(crate) fn into_float_flat(tensor: HostTensor, device: &Device) -> Result<Tensor<2>> {
    let per_item = tensor
        .shape
        .get(1..)
        .map_or(0, |item| item.iter().product());
    HostTensor {
        shape: vec![tensor.batch(), per_item],
        values: tensor.values,
    }
    .into_float(device)
}

fn checked_shape<const D: usize>(tensor: &HostTensor) -> Result<[usize; D]> {
    tensor.shape.as_slice().try_into().map_err(|_| {
        ModelError::InvalidInput(format!(
            "expected a tensor of rank {D}, got shape {:?}",
            tensor.shape
        ))
        .into()
    })
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

fn type_mismatch(expected: &str, got: &str) -> FurnaceError {
    ModelError::InvalidDataType(format!("expected {expected} tensor, got {got}")).into()
}

fn read_error(e: impl std::fmt::Debug) -> FurnaceError {
    ModelError::InferenceFailed(format!("could not read the output: {e:?}")).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendKind;

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
        assert_eq!(into_float_flat(only, &device).unwrap().dims(), [2, 3]);
        let err = take_inputs::<2>(vec![tensor]).unwrap_err().to_string();
        assert!(err.contains("expected 2 input(s), got 1"), "{err}");
    }
}
