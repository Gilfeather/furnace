// Generated ONNX models following Burn documentation
// These modules include the Rust code generated from ONNX files

#[cfg(feature = "burn-import")]
pub mod resnet18 {
    // Include the generated ResNet18 model code
    include!(concat!(env!("OUT_DIR"), "/models/resnet18.rs"));
}

// Only include if the model was successfully generated
#[cfg(all(feature = "burn-import", model_gptneox_opset18))]
pub mod gptneox_opset18 {
    // Include the generated GPT-NeoX model code
    include!(concat!(env!("OUT_DIR"), "/models/gptneox_Opset18.rs"));
}

// Re-export for easier access
use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
#[cfg(all(feature = "burn-import", model_gptneox_opset18))]
pub use gptneox_opset18::Model as GptNeoxModel;
#[cfg(feature = "burn-import")]
pub use resnet18::Model as ResNet18Model;

use crate::error::{ModelError, Result};
use crate::model::{BurnModel, OptimizationInfo};

type Backend = NdArray<f32>;

/// Simple wrapper for ResNet18 to avoid Sync issues
#[cfg(feature = "burn-import")]
#[derive(Debug)]
pub struct SimpleResNet18ModelWrapper {
    model: std::sync::Arc<std::sync::Mutex<ResNet18Model<Backend>>>,
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

/// Simple wrapper for GPT-NeoX to avoid Sync issues
#[cfg(all(feature = "burn-import", model_gptneox_opset18))]
#[derive(Debug)]
pub struct SimpleGptNeoxModelWrapper {
    model: std::sync::Arc<std::sync::Mutex<GptNeoxModel<Backend>>>,
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

/// BurnModel implementation for the simple ResNet18 wrapper
#[cfg(feature = "burn-import")]
impl BurnModel for SimpleResNet18ModelWrapper {
    fn predict(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        let [batch_size, input_size] = input.dims();

        // Validate input size
        let expected_input_size: usize = self.input_shape.iter().product();
        if input_size != expected_input_size {
            return Err(ModelError::InputValidation {
                expected: self.input_shape.clone(),
                actual: vec![input_size],
            }
            .into());
        }

        // Convert 2D input to 4D for ResNet (assuming image input)
        let input_4d = input.reshape([
            batch_size,
            self.input_shape[1], // channels
            self.input_shape[2], // height
            self.input_shape[3], // width
        ]);

        // Run inference using the ResNet18 model
        let model = self.model.lock().unwrap();
        let output = model.forward(input_4d);

        // Convert output back to 2D format
        let output_2d = if output.dims().len() > 2 {
            let output_dims = output.dims();
            let output_size = output_dims[1..].iter().product::<usize>();
            output.reshape([output_dims[0], output_size])
        } else {
            output
        };

        Ok(output_2d)
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_backend_info(&self) -> String {
        "burn-resnet18".to_string()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        OptimizationInfo {
            kernel_fusion: false,
            autotuning_cache: false,
            backend_type: self.get_backend_info(),
        }
    }
}

/// BurnModel implementation for the GPT-NeoX wrapper
#[cfg(all(feature = "burn-import", model_gptneox_opset18))]
impl BurnModel for SimpleGptNeoxModelWrapper {
    fn predict(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        let [batch_size, input_size] = input.dims();

        // Validate input size
        let expected_input_size: usize = self.input_shape.iter().product();
        if input_size != expected_input_size {
            return Err(ModelError::InputValidation {
                expected: self.input_shape.clone(),
                actual: vec![input_size],
            }
            .into());
        }

        // For GPT-NeoX, input is typically [batch_size, seq_length]
        // We may need to adjust based on the actual model requirements
        let model = self.model.lock().unwrap();
        let output = model.forward(input);

        Ok(output)
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_backend_info(&self) -> String {
        "burn-gptneox".to_string()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        OptimizationInfo {
            kernel_fusion: false,
            autotuning_cache: false,
            backend_type: self.get_backend_info(),
        }
    }
}

/// Enum for available built-in models
/// Add new models here and implement the corresponding cases below
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltInModel {
    ResNet18,
    #[cfg(model_gptneox_opset18)]
    GptNeox,
}

impl BuiltInModel {
    /// Create a BuiltInModel from a string name
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "resnet18" => Ok(Self::ResNet18),
            #[cfg(model_gptneox_opset18)]
            "gptneox" | "gpt-neox" => Ok(Self::GptNeox),
            _ => {
                let available_models = Self::available_models().join(", ");
                Err(ModelError::InvalidArgument {
                    arg: "model-name".to_string(),
                    value: name.to_string(),
                    reason: format!(
                        "unknown built-in model. Available models: {available_models}"
                    ),
                }
                .into())
            }
        }
    }

    /// Get list of available built-in models
    pub fn available_models() -> Vec<&'static str> {
        let models = vec!["resnet18"];
        #[cfg(model_gptneox_opset18)]
        models.push("gptneox");
        models
    }

    /// Get the display name of the model
    pub fn name(&self) -> &'static str {
        match self {
            Self::ResNet18 => "resnet18",
            #[cfg(model_gptneox_opset18)]
            Self::GptNeox => "gptneox",
        }
    }

    /// Create a model instance for this built-in model
    pub fn create_model(&self) -> Result<Box<dyn BurnModel>> {
        match self {
            Self::ResNet18 => {
                #[cfg(feature = "burn-import")]
                {
                    // Create the generated ResNet18 model using default method
                    let model = ResNet18Model::<Backend>::default();

                    // Create a simplified wrapper with Arc<Mutex<>> for thread safety
                    Ok(Box::new(SimpleResNet18ModelWrapper {
                        model: std::sync::Arc::new(std::sync::Mutex::new(model)),
                        name: "resnet18".to_string(),
                        input_shape: vec![1, 3, 224, 224],
                        output_shape: vec![1000],
                    }))
                }
                #[cfg(not(feature = "burn-import"))]
                {
                    Err(
                        ModelError::InvalidFormat("burn-import feature not enabled".to_string())
                            .into(),
                    )
                }
            }
            #[cfg(model_gptneox_opset18)]
            Self::GptNeox => {
                #[cfg(all(feature = "burn-import", model_gptneox_opset18))]
                {
                    // Create the generated GPT-NeoX model using default method
                    let model = GptNeoxModel::<Backend>::default();

                    // Create a simplified wrapper with Arc<Mutex<>> for thread safety
                    Ok(Box::new(SimpleGptNeoxModelWrapper {
                        model: std::sync::Arc::new(std::sync::Mutex::new(model)),
                        name: "gptneox".to_string(),
                        input_shape: vec![1, 512],
                        output_shape: vec![50257],
                    }))
                }
                #[cfg(not(all(feature = "burn-import", model_gptneox_opset18)))]
                {
                    Err(
                        ModelError::InvalidFormat("GPT-NeoX model not available (generation failed or burn-import not enabled)".to_string())
                            .into(),
                    )
                }
            }
        }
    }

    /// Get the expected input shape for this model
    pub fn input_shape(&self) -> Vec<usize> {
        match self {
            Self::ResNet18 => vec![1, 3, 224, 224],
            #[cfg(model_gptneox_opset18)]
            Self::GptNeox => vec![1, 512], // Example: sequence length 512
        }
    }

    /// Get the expected output shape for this model
    pub fn output_shape(&self) -> Vec<usize> {
        match self {
            Self::ResNet18 => vec![1000],
            #[cfg(model_gptneox_opset18)]
            Self::GptNeox => vec![50257], // GPT-NeoX vocab size
        }
    }
}
