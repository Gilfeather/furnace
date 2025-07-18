use serde_json::json;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ResNet-18 expects input shape [3, 224, 224] = 150,528 values
    // Normalized RGB values typically range from 0.0 to 1.0
    let input_size = 3 * 224 * 224; // 150,528

    // Generate random-like sample data (using a simple pattern for reproducibility)
    let mut sample_input = Vec::with_capacity(input_size);
    for i in 0..input_size {
        // Create a simple pattern that looks like normalized image data
        let value = ((i as f32 * 0.001).sin() + 1.0) * 0.5; // Range 0.0 to 1.0
        sample_input.push(value);
    }

    // Create sample request for single prediction
    let single_request = json!({
        "input": sample_input
    });

    // Create sample request for batch prediction (3 images)
    let batch_request = json!({
        "inputs": [
            sample_input.clone(),
            sample_input.iter().map(|x| x * 0.8).collect::<Vec<f32>>(),
            sample_input.iter().map(|x| x * 1.2).collect::<Vec<f32>>()
        ]
    });

    // Save sample data files
    fs::write(
        "resnet18_single_sample.json",
        serde_json::to_string_pretty(&single_request)?,
    )?;
    fs::write(
        "resnet18_batch_sample.json",
        serde_json::to_string_pretty(&batch_request)?,
    )?;

    println!("✅ Generated ResNet-18 sample data:");
    println!("   - resnet18_single_sample.json (single image)");
    println!("   - resnet18_batch_sample.json (batch of 3 images)");
    println!("   - Input shape: [3, 224, 224] = {} values", input_size);

    Ok(())
}
