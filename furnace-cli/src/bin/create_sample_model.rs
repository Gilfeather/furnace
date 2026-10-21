use burn::tensor::{Tensor, TensorData};
use furnace_burn::mlp::create_sample_model;
use std::path::PathBuf;
use tracing::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Creating sample Burn model...");
    let container = create_sample_model()?;

    let model_path = PathBuf::from("sample_model");
    container.save(&model_path)?;

    info!("Sample model saved to: sample_model.bpk and sample_model.json");
    info!("  Name: {}", container.metadata.name);
    info!(
        "  Layers: {} -> {} -> {}",
        container.metadata.input_size,
        container.metadata.hidden_size,
        container.metadata.output_size
    );

    // Smoke-test inference
    let input = Tensor::from_data(
        TensorData::new(
            vec![0.1f32; container.metadata.input_size],
            [1, container.metadata.input_size],
        ),
        container.device(),
    );
    let output = container.predict(input);
    info!("Test inference output shape: {:?}", output.dims());

    Ok(())
}
