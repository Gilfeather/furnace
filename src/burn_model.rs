use burn::backend::ndarray::NdArray;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{ModelError, Result as FurnaceResult};

type B = NdArray<f32>;

/// Simple MLP model configuration
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

/// Simple MLP model implementation
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    /// Create a new MLP model
    pub fn new(config: &MlpConfig, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(config.input_size, config.hidden_size).init(device);
        let linear2 = LinearConfig::new(config.hidden_size, config.output_size).init(device);
        let activation = Relu::new();

        Self {
            linear1,
            linear2,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

/// Model metadata stored alongside the model weights
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BurnModelMetadata {
    pub name: String,
    pub version: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub model_type: String,
    pub created_at: String,
}

/// Container for the complete model with metadata
#[derive(Debug)]
pub struct BurnModelContainer {
    pub model: std::sync::Arc<std::sync::Mutex<Mlp<B>>>,
    pub metadata: BurnModelMetadata,
    #[allow(dead_code)]
    pub config: MlpConfig,
}

impl BurnModelContainer {
    /// Create a new model container
    pub fn new(config: MlpConfig, name: String) -> Self {
        let device = <B as Backend>::Device::default();
        let model = Mlp::new(&config, &device);

        let metadata = BurnModelMetadata {
            name,
            version: "1.0.0".to_string(),
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            model_type: "mlp".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            metadata,
            config,
        }
    }

    /// Save model to file
    #[allow(dead_code)]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> FurnaceResult<()> {
        let recorder = CompactRecorder::new();

        // Save model weights
        let model_path = path.as_ref().with_extension("burn");
        let model = self.model.lock().unwrap();
        model
            .clone()
            .save_file(model_path.clone(), &recorder)
            .map_err(|e| ModelError::LoadFailed {
                path: model_path.clone(),
                source: Box::new(e),
            })?;

        // Save metadata
        let metadata_path = path.as_ref().with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to serialize metadata: {e}")))?;

        std::fs::write(&metadata_path, metadata_json).map_err(|e| ModelError::LoadFailed {
            path: metadata_path,
            source: Box::new(e),
        })?;

        Ok(())
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> FurnaceResult<Self> {
        let model_path = path.as_ref();
        let metadata_path = model_path.with_extension("json");

        // Load metadata first
        let metadata_content =
            std::fs::read_to_string(&metadata_path).map_err(|e| ModelError::LoadFailed {
                path: metadata_path.clone(),
                source: Box::new(e),
            })?;

        let metadata: BurnModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to parse metadata: {e}")))?;

        // Create config from metadata
        let config = MlpConfig::new(
            metadata.input_size,
            metadata.hidden_size,
            metadata.output_size,
        );

        // Load model weights
        let device = <B as Backend>::Device::default();
        let model = Mlp::new(&config, &device);
        let recorder = CompactRecorder::new();

        // Try to load from .mpk file
        let mpk_path = path.as_ref().with_extension("mpk");
        let model = model
            .load_file(mpk_path.clone(), &recorder, &device)
            .map_err(|e| ModelError::LoadFailed {
                path: mpk_path,
                source: Box::new(e),
            })?;

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            metadata,
            config,
        })
    }

    /// Run inference
    pub fn predict(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let model = self.model.lock().unwrap();
        model.forward(input)
    }

    /// Get input shape
    pub fn input_shape(&self) -> Vec<usize> {
        vec![self.metadata.input_size]
    }

    /// Get output shape
    pub fn output_shape(&self) -> Vec<usize> {
        vec![self.metadata.output_size]
    }
}

/// Create a sample model for testing
pub fn create_sample_model() -> FurnaceResult<BurnModelContainer> {
    let config = MlpConfig::new(784, 128, 10); // MNIST-like model
    let container = BurnModelContainer::new(config, "sample_mnist_model".to_string());
    Ok(container)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use tempfile::tempdir;

    #[test]
    fn test_model_creation() {
        let container = create_sample_model().unwrap();
        assert_eq!(container.metadata.input_size, 784);
        assert_eq!(container.metadata.output_size, 10);
        assert_eq!(container.metadata.model_type, "mlp");
    }

    #[test]
    fn test_model_forward() {
        let container = create_sample_model().unwrap();
        let device = <B as Backend>::Device::default();

        // Create dummy input
        let input_data = vec![0.1; 784];
        let input = Tensor::from_data(TensorData::new(input_data, [1, 784]), &device);

        let output = container.predict(input);
        let output_shape = output.shape();

        assert_eq!(output_shape.dims, [1, 10]);
    }

    #[test]
    fn test_model_save_load() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model");

        // Create and save model
        let original_container = create_sample_model().unwrap();
        original_container.save(&model_path).unwrap();

        // Load model
        let loaded_container = BurnModelContainer::load(&model_path).unwrap();

        // Verify metadata
        assert_eq!(
            loaded_container.metadata.name,
            original_container.metadata.name
        );
        assert_eq!(
            loaded_container.metadata.input_size,
            original_container.metadata.input_size
        );
        assert_eq!(
            loaded_container.metadata.output_size,
            original_container.metadata.output_size
        );

        // Test inference
        let device = <B as Backend>::Device::default();
        let input_data = vec![0.1; 784];
        let input = Tensor::from_data(TensorData::new(input_data, [1, 784]), &device);

        let output = loaded_container.predict(input);
        assert_eq!(output.shape().dims, [1, 10]);
    }
}
