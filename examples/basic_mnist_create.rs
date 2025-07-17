use burn::backend::ndarray::NdArray;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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

/// Model metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub model_type: String,
    pub created_at: String,
    pub description: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Creating MNIST-like MLP model example...");

    // Create output directory
    let output_dir = Path::new("examples/basic_mnist");
    fs::create_dir_all(output_dir)?;

    // Model configuration
    let config = MlpConfig::new(784, 128, 10); // MNIST: 28x28 -> 128 -> 10 classes
    let device = <B as Backend>::Device::default();

    // Create model
    let model: Mlp<B> = Mlp::new(&config, &device);

    // Create some dummy training to initialize weights properly
    println!("📊 Initializing model with dummy forward pass...");
    let dummy_input = Tensor::random(
        [1, 784],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let _output = model.forward(dummy_input);

    // Save model
    let recorder = CompactRecorder::new();
    let model_path = output_dir.join("model");

    println!("💾 Saving model to: {}.mpk", model_path.display());
    model.clone().save_file(&model_path, &recorder)?;

    // Create metadata
    let metadata = ModelMetadata {
        name: "mnist_mlp_example".to_string(),
        version: "1.0.0".to_string(),
        input_size: config.input_size,
        hidden_size: config.hidden_size,
        output_size: config.output_size,
        model_type: "mlp".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        description: "Simple 3-layer MLP for MNIST-like classification (784->128->10)".to_string(),
    };

    // Save metadata
    let metadata_path = output_dir.join("model.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;

    println!("📋 Model metadata saved to: {}", metadata_path.display());

    // Test inference
    println!("🧪 Testing inference...");
    let test_input = Tensor::zeros([1, 784], &device);
    let output = model.forward(test_input);
    let output_shape = output.shape();

    println!("✅ Model created successfully!");
    println!("   📁 Model file: {}.mpk", model_path.display());
    println!("   📄 Metadata: {}", metadata_path.display());
    println!("   📊 Input shape: [batch_size, {}]", config.input_size);
    println!("   📊 Output shape: {:?}", output_shape.dims);
    println!("   🏷️  Model type: {}", metadata.model_type);

    println!("\n🚀 Next steps:");
    println!(
        "   1. Start the server: cargo run --bin furnace -- --model-path {} --log-level debug",
        model_path.with_extension("mpk").display()
    );
    println!("   2. Test inference: cargo run --example basic_mnist_test");

    Ok(())
}
