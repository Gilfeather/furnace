//! Models for tests and benchmarks of the server itself, behind the `test-models` feature:
//! they answer without computing anything worth serving.

use super::TensorModel;
use crate::error::{ModelError, Result};
use crate::tensor::{DataType, HostTensor, IoSpec, Values};

/// The integers of Burn's CPU backend (`i32`), which these models report as theirs, so the
/// server's range check runs as it does for a real CPU model
const CPU_INT_RANGE: (i64, i64) = (i32::MIN as i64, i32::MAX as i64);

/// Constant-output model for tests and benchmarking server overhead.
///
/// Never used as a silent fallback: if a real model fails to load, loading fails.
#[derive(Debug)]
pub struct DummyModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl DummyModel {
    /// A model taking `input_shape` per item and returning `output_shape` per item, all 0.5
    pub fn new(name: String, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            output_shape,
        }
    }
}

impl TensorModel for DummyModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> &str {
        "cpu"
    }

    fn int_range(&self) -> (i64, i64) {
        CPU_INT_RANGE
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &self.input_shape)]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &self.output_shape)]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let batch = match inputs.as_slice() {
            [input] => input.batch(),
            _ => {
                return Err(ModelError::InvalidInput(format!(
                    "expected 1 input(s), got {}",
                    inputs.len()
                ))
                .into());
            }
        };
        let mut shape = vec![batch];
        shape.extend_from_slice(&self.output_shape);
        let len = shape.iter().product();
        Ok(vec![HostTensor {
            shape,
            values: Values::F32(vec![0.5; len]),
        }])
    }
}

/// A model of named tensors for tests: `ids` (INT64, 4 per item) and `scale` (FP32, one per
/// item) in; `scores` (FP32, `ids * scale`) and `count` (INT64, the sum of `ids`) out. With a
/// fixed batch it refuses any other batch size, like a graph exported that way.
#[derive(Debug)]
pub struct DummyTensorsModel {
    fixed_batch: Option<usize>,
    runs: std::sync::atomic::AtomicUsize,
}

impl DummyTensorsModel {
    /// With `fixed_batch`, the model runs batches of exactly that size
    pub fn new(fixed_batch: Option<usize>) -> Self {
        Self {
            fixed_batch,
            runs: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Forward passes so far
    pub fn runs(&self) -> usize {
        self.runs.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl TensorModel for DummyTensorsModel {
    fn name(&self) -> &str {
        "tensors"
    }

    fn backend(&self) -> &str {
        "cpu"
    }

    fn int_range(&self) -> (i64, i64) {
        CPU_INT_RANGE
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![
            IoSpec {
                name: "ids".to_string(),
                dtype: DataType::Int64,
                shape: vec![Some(4)],
            },
            IoSpec {
                name: "scale".to_string(),
                dtype: DataType::Fp32,
                shape: vec![],
            },
        ]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![
            IoSpec::fp32("scores", &[4]),
            IoSpec {
                name: "count".to_string(),
                dtype: DataType::Int64,
                shape: vec![],
            },
        ]
    }

    fn fixed_batch(&self) -> Option<usize> {
        self.fixed_batch
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (Values::I64(ids), Values::F32(scale)) = (&inputs[0].values, &inputs[1].values) else {
            return Err(ModelError::InferenceFailed("unexpected types".to_string()).into());
        };
        let batch = inputs[0].batch();
        if self.fixed_batch.is_some_and(|fixed| fixed != batch) {
            return Err(ModelError::InferenceFailed(format!("got a batch of {batch}")).into());
        }
        let scores = ids
            .chunks(4)
            .zip(scale)
            .flat_map(|(ids, s)| ids.iter().map(move |&id| id as f32 * s))
            .collect();
        let counts = ids.chunks(4).map(|ids| ids.iter().sum()).collect();
        Ok(vec![
            HostTensor {
                shape: vec![batch, 4],
                values: Values::F32(scores),
            },
            HostTensor {
                shape: vec![batch],
                values: Values::I64(counts),
            },
        ])
    }
}
