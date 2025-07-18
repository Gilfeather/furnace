use serde_json::json;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ResNet-18 expects input shape [3, 224, 224] = 150,528 values
    let input_size = 3 * 224 * 224;

    // Create a smaller sample for testing (first 1000 values)
    let small_input: Vec<f32> = (0..1000)
        .map(|i| (i as f32 * 0.001).sin() * 0.5 + 0.5)
        .collect();

    let small_request = json!({
        "input": small_input
    });

    // Create full size sample but with simpler data
    let full_input: Vec<f32> = vec![0.5; input_size];
    let full_request = json!({
        "input": full_input
    });

    fs::write(
        "resnet18_small_test.json",
        serde_json::to_string_pretty(&small_request)?,
    )?;
    fs::write(
        "resnet18_full_test.json",
        serde_json::to_string_pretty(&full_request)?,
    )?;

    println!("✅ Generated ResNet-18 test samples:");
    println!("   - resnet18_small_test.json (1000 values for testing)");
    println!("   - resnet18_full_test.json ({input_size} values, full size)");

    Ok(())
}
