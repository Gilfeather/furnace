// Built-in models generated from ONNX files in models/ by build.rs (burn-onnx).
// Weights are embedded in the binary, so built-in models need no files at runtime.

#[cfg(model_resnet18)]
#[allow(clippy::all, unused)]
pub mod resnet18 {
    include!(concat!(env!("OUT_DIR"), "/models/resnet18.rs"));
}

#[cfg(model_resnet18)]
use burn::tensor::{Device, Tensor};

use crate::backend::BackendKind;
use crate::error::{ModelError, Result};
use crate::model::BurnModel;

/// ResNet-18 image classifier (ImageNet, 1000 classes)
#[cfg(model_resnet18)]
#[derive(Debug)]
pub struct ResNet18 {
    model: resnet18::Model,
    device: Device,
    backend: BackendKind,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[cfg(model_resnet18)]
impl ResNet18 {
    pub fn new(backend: BackendKind) -> Self {
        let device = backend.device();
        Self {
            model: resnet18::Model::from_embedded(&device),
            device,
            backend,
            input_shape: vec![3, 224, 224],
            output_shape: vec![1000],
        }
    }
}

#[cfg(model_resnet18)]
impl BurnModel for ResNet18 {
    fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        let [batch_size, input_size] = input.dims();
        let expected: usize = self.input_shape.iter().product();
        if input_size != expected {
            return Err(ModelError::InputValidation {
                expected: self.input_shape.clone(),
                actual: vec![input_size],
            }
            .into());
        }

        // Flattened [batch, C*H*W] -> NCHW image tensor
        let images = input.reshape([
            batch_size,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]);
        Ok(self.model.forward(images))
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn get_name(&self) -> &str {
        "resnet18"
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

/// Models compiled into the binary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInModel {
    ResNet18,
}

impl BuiltInModel {
    /// Look up a built-in model by name
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "resnet18" | "resnet-18" => Ok(Self::ResNet18),
            _ => Err(ModelError::InvalidArgument {
                arg: "model-name".to_string(),
                value: name.to_string(),
                reason: format!(
                    "unknown built-in model. Available models: {}",
                    Self::available_models().join(", ")
                ),
            }
            .into()),
        }
    }

    /// Built-in models that were successfully generated at build time
    pub fn available_models() -> Vec<&'static str> {
        vec![
            #[cfg(model_resnet18)]
            "resnet18",
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ResNet18 => "resnet18",
        }
    }

    /// Instantiate the model on the given backend
    pub fn create_model(&self, backend: BackendKind) -> Result<Box<dyn BurnModel>> {
        match self {
            Self::ResNet18 => {
                #[cfg(model_resnet18)]
                {
                    Ok(Box::new(ResNet18::new(backend)))
                }
                #[cfg(not(model_resnet18))]
                {
                    let _ = backend;
                    Err(ModelError::InvalidFormat(
                        "resnet18 was not compiled into this binary. \
                         Place models/resnet18.onnx and rebuild with the `onnx` feature"
                            .to_string(),
                    )
                    .into())
                }
            }
        }
    }
}
