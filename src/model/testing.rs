//! Models for tests and benchmarks of the server itself, behind the `test-models` feature:
//! they answer without computing anything worth serving.

use burn::tensor::{Device, Tensor, TensorData};

use super::BurnModel;
use super::kinds::single_f32_run;
use crate::backend::BackendKind;
use crate::error::{ModelError, Result};
use crate::tensor::{DataType, HostTensor, IoSpec, Values};

/// Constant-output model for tests and benchmarking server overhead.
///
/// Never used as a silent fallback: if a real model fails to load, loading fails.
#[derive(Debug)]
pub struct DummyModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    device: Device,
}

impl DummyModel {
    /// A model taking `input_shape` per item and returning `output_shape` per item, all 0.5
    pub fn new(name: String, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            output_shape,
            device: BackendKind::Cpu.device(),
        }
    }
}

impl BurnModel for DummyModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &self.input_shape)]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &self.output_shape)]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let output_size = self.output_shape.iter().product::<usize>();
        single_f32_run(
            inputs,
            &self.device,
            |input| {
                let [batch_size, _] = input.dims();
                Ok(Tensor::from_data(
                    TensorData::new(
                        vec![0.5f32; batch_size * output_size],
                        [batch_size, output_size],
                    ),
                    &self.device,
                ))
            },
            &self.output_shape,
        )
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

impl BurnModel for DummyTensorsModel {
    fn name(&self) -> &str {
        "tensors"
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
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
