//! Create an MNIST-shaped MLP (784 -> 128 -> 10) and save it in the format
//! `furnace --model-path` loads: `model.bpk` (Burnpack weights) + `model.json` (metadata).

use furnace::BackendKind;
use furnace::burn_model::{BurnModelContainer, MlpConfig};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Creating MNIST-like MLP model example...");

    let output_dir = Path::new("examples/basic_mnist");
    fs::create_dir_all(output_dir)?;

    let config = MlpConfig::new(784, 128, 10); // 28x28 -> 128 -> 10 classes
    let container =
        BurnModelContainer::new(config, "mnist_mlp_example".to_string(), BackendKind::Cpu);

    let model_path = output_dir.join("model");
    container.save(&model_path)?;

    println!("✅ Model created successfully!");
    println!(
        "   📁 Weights:  {}",
        model_path.with_extension("bpk").display()
    );
    println!(
        "   📄 Metadata: {}",
        model_path.with_extension("json").display()
    );
    println!(
        "   📊 Input:  [batch_size, {}]",
        container.metadata.input_size
    );
    println!(
        "   📊 Output: [batch_size, {}]",
        container.metadata.output_size
    );

    println!("\n🚀 Next steps:");
    println!(
        "   1. Start the server: cargo run --release --bin furnace -- --model-path {}",
        model_path.display()
    );
    println!("   2. Test inference:   cargo run --example basic_mnist_test");

    Ok(())
}
