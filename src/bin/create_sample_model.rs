use furnace::burn_model::create_sample_model;
use std::path::PathBuf;
use tracing::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Creating sample Burn model...");

    // Create sample model
    let container = create_sample_model()?;

    // Save to file
    let model_path = PathBuf::from("sample_model");
    container.save(&model_path)?;

    info!("Sample model saved to: sample_model.mpk and sample_model.json");
    info!("Model details:");
    info!("  Name: {}", container.metadata.name);
    info!("  Input size: {}", container.metadata.input_size);
    info!("  Hidden size: {}", container.metadata.hidden_size);
    info!("  Output size: {}", container.metadata.output_size);
    info!("  Type: {}", container.metadata.model_type);

    // Test inference
    use burn::backend::ndarray::NdArray;
    use burn::prelude::Backend;
    use burn::tensor::{Tensor, TensorData};

    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();

    let input_data = vec![0.1; container.metadata.input_size];
    let input = Tensor::from_data(
        TensorData::new(input_data, [1, container.metadata.input_size]),
        &device,
    );

    let output = container.predict(input);
    let output_shape = output.shape();

    info!("Test inference completed:");
    info!("  Input shape: [1, {}]", container.metadata.input_size);
    info!("  Output shape: {:?}", output_shape.dims);
    info!("  Output tensor created successfully");

    Ok(())
}
