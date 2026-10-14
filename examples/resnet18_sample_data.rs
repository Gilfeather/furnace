//! Generate request bodies for the built-in ResNet-18:
//! `resnet18_single_sample.json` (one image) and `resnet18_batch_sample.json` (three images).

use serde_json::json;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ResNet-18 expects [3, 224, 224] = 150,528 values per image (normalized RGB, CHW order)
    let input_size = 3 * 224 * 224;

    // Deterministic pattern in [0, 1] that looks like normalized image data
    let sample: Vec<f32> = (0..input_size)
        .map(|i| ((i as f32 * 0.001).sin() + 1.0) * 0.5)
        .collect();

    let single_request = json!({ "input": sample });
    let batch_request = json!({
        "inputs": [
            sample,
            sample.iter().map(|x| x * 0.8).collect::<Vec<f32>>(),
            sample.iter().map(|x| (x * 1.2).min(1.0)).collect::<Vec<f32>>(),
        ]
    });

    // Compact JSON: pretty-printing 150k floats only bloats the request body
    fs::write(
        "resnet18_single_sample.json",
        serde_json::to_string(&single_request)?,
    )?;
    fs::write(
        "resnet18_batch_sample.json",
        serde_json::to_string(&batch_request)?,
    )?;

    println!("✅ Generated ResNet-18 sample data:");
    println!("   - resnet18_single_sample.json (single image)");
    println!("   - resnet18_batch_sample.json (batch of 3 images)");
    println!("   - Input shape: [3, 224, 224] = {input_size} values per image");

    Ok(())
}
