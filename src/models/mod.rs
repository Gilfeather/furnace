use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use tracing::info;

use crate::error::{ModelError, Result};
use crate::model::{BurnModel, OptimizationInfo};

type B = NdArray<f32>;

// Include generated ONNX models directly
#[cfg(feature = "burn-import")]
pub mod resnet18 {
    // Include the generated ResNet18 model code
    include!(concat!(env!("OUT_DIR"), "/models/resnet18.rs"));
}

/// Generated ONNX model wrapper
#[derive(Debug)]
pub struct GeneratedModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl GeneratedModel {
    pub fn new(name: String, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            output_shape,
        }
    }

    /// Load a model by name
    pub fn load_by_name(model_name: &str) -> Result<Self> {
        match model_name {
            "resnet18" => {
                info!("Loading ResNet18 model");
                Ok(Self::new(
                    "resnet18".to_string(),
                    vec![1, 3, 224, 224], // Standard ResNet input: [batch, channels, height, width]
                    vec![1000],           // ImageNet classes
                ))
            }
            _ => Err(ModelError::InvalidFormat(format!("Unknown model: {}", model_name)).into()),
        }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Perform inference using the generated model
    pub fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let [batch_size, input_size] = input.dims();

        // Validate input shape
        let expected_input_size: usize = self.input_shape.iter().product();
        if input_size != expected_input_size {
            return Err(ModelError::InputValidation {
                expected: self.input_shape.clone(),
                actual: vec![input_size],
            }
            .into());
        }

        match self.name.as_str() {
            "resnet18" => {
                #[cfg(feature = "burn-import")]
                {
                    // Use placeholder implementation for now
                    // TODO: Integrate actual generated model when Module trait is properly implemented
                    info!("Running ResNet18 inference (placeholder implementation)");

                    let output_size = self.output_shape.iter().product::<usize>();
                    let mut data = vec![0.001; batch_size * output_size];
                    // Simulate ImageNet classification output
                    for i in 0..batch_size {
                        data[i * output_size + 285] = 0.1; // Random class
                    }
                    let output_tensor = Tensor::from_data(
                        burn::tensor::TensorData::new(data, [batch_size, output_size]),
                        &Default::default(),
                    );

                    info!(
                        "ResNet18 inference completed for batch size: {}",
                        batch_size
                    );
                    Ok(output_tensor)
                }
                #[cfg(not(feature = "burn-import"))]
                {
                    Err(
                        ModelError::InvalidFormat("burn-import feature not enabled".to_string())
                            .into(),
                    )
                }
            }
            _ => Err(ModelError::InvalidFormat(format!("Unsupported model: {}", self.name)).into()),
        }
    }
}

impl BurnModel for GeneratedModel {
    fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        self.predict(input)
    }

    fn get_input_shape(&self) -> &[usize] {
        self.get_input_shape()
    }

    fn get_output_shape(&self) -> &[usize] {
        self.get_output_shape()
    }

    fn get_name(&self) -> &str {
        self.get_name()
    }

    fn get_backend_info(&self) -> String {
        "onnx-generated".to_string()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        OptimizationInfo {
            kernel_fusion: false,
            autotuning_cache: false,
            backend_type: self.get_backend_info(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "burn-import")]
    fn test_resnet18_model_creation() {
        let result = GeneratedModel::load_by_name("resnet18");
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.get_name(), "resnet18");
        assert_eq!(model.get_input_shape(), &[1, 3, 224, 224]);
        assert_eq!(model.get_output_shape(), &[1000]);
    }

    #[test]
    fn test_unknown_model() {
        let result = GeneratedModel::load_by_name("unknown");
        assert!(result.is_err());
    }
}
