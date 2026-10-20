//! The [`BurnModel`] implementations the crate ships: [`Model::from_fn`]'s wrapper and the
//! file-based MLP. The dummies for tests and benchmarks are in `testing`.
//!
//! [`Model::from_fn`]: super::Model::from_fn

use burn::tensor::{Device, Tensor};

use super::BurnModel;
use crate::backend::BackendKind;
use crate::burn_model::BurnModelContainer;
use crate::error::Result;
use crate::tensor::{HostTensor, IoSpec, take_inputs};

/// [`Model::from_fn`]: a forward function from one float tensor to one
pub(super) struct FnModel<F, const DI: usize, const DO: usize> {
    pub(super) name: String,
    pub(super) backend: BackendKind,
    pub(super) device: Device,
    pub(super) input: IoSpec,
    pub(super) output: IoSpec,
    pub(super) forward: F,
}

impl<F, const DI: usize, const DO: usize> std::fmt::Debug for FnModel<F, DI, DO> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnModel")
            .field("name", &self.name)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl<F, const DI: usize, const DO: usize> BurnModel for FnModel<F, DI, DO>
where
    F: Fn(Tensor<DI>) -> Tensor<DO> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![self.input.clone()]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![self.output.clone()]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let [input] = take_inputs(inputs)?;
        let output = (self.forward)(input.into_float::<DI>(&self.device)?);
        Ok(vec![HostTensor::from_float(output)?])
    }
}

/// One `f32` tensor in, one out, for the models below
pub(super) fn single_f32_run(
    inputs: Vec<HostTensor>,
    device: &Device,
    forward: impl FnOnce(Tensor<2>) -> Result<Tensor<2>>,
    output_shape: &[usize],
) -> Result<Vec<HostTensor>> {
    let [input] = take_inputs(inputs)?;
    let mut shape = vec![input.batch()];
    shape.extend_from_slice(output_shape);
    let output = forward(input.into_float_flat(device)?)?;
    Ok(vec![HostTensor {
        shape,
        values: HostTensor::from_float(output)?.values,
    }])
}

/// File-based MLP model (`.bpk` weights + `.json` metadata)
#[derive(Debug)]
pub struct MlpModel {
    container: BurnModelContainer,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl MlpModel {
    /// Serve a loaded MLP container, with the shapes its metadata declares
    pub fn new(container: BurnModelContainer) -> Self {
        let input_shape = container.input_shape();
        let output_shape = container.output_shape();
        Self {
            container,
            input_shape,
            output_shape,
        }
    }
}

impl BurnModel for MlpModel {
    fn name(&self) -> &str {
        &self.container.metadata.name
    }

    fn backend(&self) -> BackendKind {
        self.container.backend
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &self.input_shape)]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &self.output_shape)]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        single_f32_run(
            inputs,
            self.container.device(),
            |x| Ok(self.container.predict(x)),
            &self.output_shape,
        )
    }
}
