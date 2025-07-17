use reqwest;
use serde_json::{json, Value};
use std::time::Instant;

/// Generate synthetic image data that looks more realistic
fn generate_synthetic_mnist_data() -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0; 784];
    
    // Create a simple pattern - a rough circle in the center
    for y in 0..28 {
        for x in 0..28 {
            let idx = y * 28 + x;
            let center_x = 14.0;
            let center_y = 14.0;
            let distance = ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt();
            
            // Create a circle with some noise
            if distance < 8.0 {
                data[idx] = rng.gen_range(0.7..1.0);
            } else if distance < 10.0 {
                data[idx] = rng.gen_range(0.3..0.7);
            } else {
                data[idx] = rng.gen_range(0.0..0.2);
            }
        }
    }
    
    data
}

/// Generate edge-case test data
fn generate_edge_cases() -> Vec<(String, Vec<f32>)> {
    vec![
        ("all_zeros".to_string(), vec![0.0; 784]),
        ("all_ones".to_string(), vec![1.0; 784]),
        ("alternating".to_string(), (0..784).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect()),
        ("gradient".to_string(), (0..784).map(|i| i as f32 / 784.0).collect()),
        ("random_sparse".to_string(), {
            let mut data = vec![0.0; 784];
            for i in (0..784).step_by(10) {
                data[i] = rand::random::<f32>();
            }
            data
        }),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Advanced Furnace inference testing with custom data...");

    let base_url = "http://127.0.0.1:3000";
    let client = reqwest::Client::new();

    // Test 1: Verify server is running
    println!("\n1️⃣ Verifying server connection...");
    match client.get(&format!("{}/healthz", base_url)).send().await {
        Ok(response) if response.status().is_success() => {
            let health: Value = response.json().await?;
            println!("✅ Server is running");
            println!("   Model loaded: {}", health["model_loaded"].as_bool().unwrap_or(false));
        }
        Ok(response) => {
            println!("❌ Server responded with error: {}", response.status());
            return Ok(());
        }
        Err(_) => {
            println!("❌ Cannot connect to server. Make sure it's running:");
            println!("   cargo run --bin furnace -- --model-path examples/basic_mnist/model.mpk");
            return Ok(());
        }
    }

    // Test 2: Synthetic realistic data
    println!("\n2️⃣ Testing with synthetic MNIST-like data...");
    for i in 1..=3 {
        let synthetic_data = generate_synthetic_mnist_data();
        let request_body = json!({ "input": synthetic_data });

        let start = Instant::now();
        let response = client
            .post(&format!("{}/predict", base_url))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        let duration = start.elapsed();

        if response.status().is_success() {
            let result: Value = response.json().await?;
            if let Some(output) = result["output"].as_array() {
                let max_idx = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.as_f64().partial_cmp(&b.as_f64()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                println!("   Sample {}: ✅ Predicted class {} ({:.2}ms)", 
                    i, max_idx, duration.as_secs_f64() * 1000.0);
            }
        } else {
            println!("   Sample {}: ❌ {}", i, response.status());
        }
    }

    // Test 3: Edge cases
    println!("\n3️⃣ Testing edge cases...");
    let edge_cases = generate_edge_cases();
    
    for (name, data) in edge_cases {
        let request_body = json!({ "input": data });
        
        let start = Instant::now();
        let response = client
            .post(&format!("{}/predict", base_url))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        let duration = start.elapsed();

        if response.status().is_success() {
            let result: Value = response.json().await?;
            if let Some(output) = result["output"].as_array() {
                let max_idx = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.as_f64().partial_cmp(&b.as_f64()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                println!("   {}: ✅ Class {} ({:.2}ms)", name, max_idx, duration.as_secs_f64() * 1000.0);
            }
        } else {
            println!("   {}: ❌ {}", name, response.status());
        }
    }

    // Test 4: Batch-like processing simulation
    println!("\n4️⃣ Simulating batch processing...");
    let batch_size = 10;
    let mut total_time = 0.0;
    let mut successful_requests = 0;

    let start_batch = Instant::now();
    for i in 1..=batch_size {
        let data = generate_synthetic_mnist_data();
        let request_body = json!({ "input": data });

        let start = Instant::now();
        let response = client
            .post(&format!("{}/predict", base_url))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        total_time += duration;

        if response.status().is_success() {
            successful_requests += 1;
            if i % 3 == 0 {
                println!("   Processed {}/{} requests...", i, batch_size);
            }
        }
    }
    let batch_duration = start_batch.elapsed().as_secs_f64() * 1000.0;

    println!("   ✅ Batch completed: {}/{} successful", successful_requests, batch_size);
    println!("   Average request time: {:.2}ms", total_time / batch_size as f64);
    println!("   Total batch time: {:.2}ms", batch_duration);
    println!("   Throughput: {:.1} requests/second", batch_size as f64 / (batch_duration / 1000.0));

    // Test 5: Error scenarios
    println!("\n5️⃣ Testing error handling...");
    
    let error_cases = vec![
        ("empty_input", json!({ "input": [] })),
        ("wrong_size", json!({ "input": vec![1.0; 100] })),
        ("missing_input", json!({ "data": vec![1.0; 784] })),
        ("invalid_type", json!({ "input": "not_an_array" })),
        ("null_input", json!({ "input": serde_json::Value::Null })),
    ];

    for (name, request_body) in error_cases {
        let response = client
            .post(&format!("{}/predict", base_url))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if response.status().is_client_error() {
            println!("   {}: ✅ Correctly rejected ({})", name, response.status());
        } else {
            println!("   {}: ❌ Unexpected response ({})", name, response.status());
        }
    }

    println!("\n🎉 Advanced testing completed!");
    println!("\n📊 Summary:");
    println!("   - Synthetic data generation and testing");
    println!("   - Edge case validation");
    println!("   - Batch processing simulation");
    println!("   - Comprehensive error handling");
    println!("   - Performance measurement");

    Ok(())
}