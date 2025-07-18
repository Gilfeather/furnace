use serde_json::json;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a smaller test sample (just 10 values for testing)
    let small_sample = vec![0.5; 10];

    let small_request = json!({
        "input": small_sample
    });

    fs::write(
        "small_test_sample.json",
        serde_json::to_string_pretty(&small_request)?,
    )?;

    println!("✅ Generated small test sample: small_test_sample.json");

    Ok(())
}
