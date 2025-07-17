
use serde_json::{json, Value};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Furnace inference server...");

    let base_url = "http://127.0.0.1:3000";
    let client = reqwest::Client::new();

    // Test 1: Health check
    println!("\n1️⃣ Testing health check...");
    let start = Instant::now();
    let response = client.get(format!("{base_url}/healthz")).send().await?;
    let duration = start.elapsed();

    if response.status().is_success() {
        let health: Value = response.json().await?;
        println!(
            "✅ Health check passed ({:.2}ms)",
            duration.as_secs_f64() * 1000.0
        );
        println!(
            "   Status: {}",
            health["status"].as_str().unwrap_or("unknown")
        );
        println!(
            "   Model loaded: {}",
            health["model_loaded"].as_bool().unwrap_or(false)
        );
    } else {
        println!("❌ Health check failed: {}", response.status());
        return Ok(());
    }

    // Test 2: Model info
    println!("\n2️⃣ Testing model info...");
    let start = Instant::now();
    let response = client
        .get(format!("{base_url}/model/info"))
        .send()
        .await?;
    let duration = start.elapsed();

    if response.status().is_success() {
        let info: Value = response.json().await?;
        println!(
            "✅ Model info retrieved ({:.2}ms)",
            duration.as_secs_f64() * 1000.0
        );
        if let Some(model_info) = info["model_info"].as_object() {
            println!(
                "   Name: {}",
                model_info["name"].as_str().unwrap_or("unknown")
            );
            println!("   Input shape: {:?}", model_info["input_spec"]["shape"]);
            println!("   Output shape: {:?}", model_info["output_spec"]["shape"]);
            println!(
                "   Backend: {}",
                model_info["backend"].as_str().unwrap_or("unknown")
            );
        }
    } else {
        println!("❌ Model info failed: {}", response.status());
    }

    // Test 3: Inference with random data
    println!("\n3️⃣ Testing inference with random MNIST-like data...");

    // Generate random 28x28 image data (flattened to 784)
    let random_image: Vec<f32> = (0..784).map(|_| rand::random::<f32>()).collect();

    let request_body = json!({
        "input": random_image
    });

    let start = Instant::now();
    let response = client
        .post(format!("{base_url}/predict"))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?;
    let duration = start.elapsed();

    if response.status().is_success() {
        let result: Value = response.json().await?;
        println!(
            "✅ Inference successful ({:.2}ms)",
            duration.as_secs_f64() * 1000.0
        );

        if let Some(output) = result["output"].as_array() {
            println!("   Output length: {}", output.len());
            println!("   Sample outputs: {:?}", &output[..3.min(output.len())]);

            // Find the predicted class (highest probability)
            let mut max_idx = 0;
            let mut max_val = -f64::INFINITY;
            for (i, val) in output.iter().enumerate() {
                if let Some(v) = val.as_f64() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }
            }
            println!(
                "   Predicted class: {max_idx} (confidence: {max_val:.4})"
            );
        }

        if let Some(inference_time) = result["inference_time_ms"].as_f64() {
            println!("   Server inference time: {inference_time:.2}ms");
        }
    } else {
        println!("❌ Inference failed: {}", response.status());
        let error_text = response.text().await?;
        println!("   Error: {error_text}");
    }

    // Test 4: Multiple rapid requests
    println!("\n4️⃣ Testing multiple rapid requests...");
    let num_requests = 5;
    let mut total_time = 0.0;

    for i in 1..=num_requests {
        let random_image: Vec<f32> = (0..784).map(|_| rand::random::<f32>()).collect();
        let request_body = json!({ "input": random_image });

        let start = Instant::now();
        let response = client
            .post(format!("{base_url}/predict"))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        total_time += duration;

        if response.status().is_success() {
            println!("   Request {i}: ✅ {duration:.2}ms");
        } else {
            println!("   Request {}: ❌ {}", i, response.status());
        }
    }

    println!("   Average time: {:.2}ms", total_time / num_requests as f64);

    // Test 5: Error handling
    println!("\n5️⃣ Testing error handling...");
    let invalid_request = json!({
        "input": [1.0, 2.0, 3.0] // Wrong size (should be 784)
    });

    let response = client
        .post(format!("{base_url}/predict"))
        .header("Content-Type", "application/json")
        .json(&invalid_request)
        .send()
        .await?;

    if response.status().is_client_error() {
        println!("✅ Error handling works correctly");
        println!("   Status: {}", response.status());
        let error: Value = response.json().await?;
        println!(
            "   Error message: {}",
            error["error"].as_str().unwrap_or("unknown")
        );
    } else {
        println!("❌ Expected error response, got: {}", response.status());
    }

    println!("\n🎉 All tests completed!");
    println!("\n💡 Tips:");
    println!("   - Try different input sizes to test validation");
    println!("   - Monitor server logs for structured logging output");
    println!("   - Use FURNACE_ENV=production for JSON logs");

    Ok(())
}
