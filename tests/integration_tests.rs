use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use reqwest::Client;
use serde_json::json;
use furnace::model::load_model;
use furnace::api::start_server;

#[tokio::test]
async fn test_end_to_end_inference() {
    // Start server in background
    let model_path = PathBuf::from("test_model.burn");
    let model = load_model(&model_path).expect("Failed to load test model");
    
    let server_handle = tokio::spawn(async move {
        start_server("127.0.0.1", 3001, model).await.unwrap();
    });
    
    // Wait for server to start
    sleep(Duration::from_millis(500)).await;
    
    let client = Client::new();
    let base_url = "http://127.0.0.1:3001";
    
    // Test health check
    let health_response = client
        .get(&format!("{}/healthz", base_url))
        .send()
        .await
        .expect("Failed to send health check request");
    
    assert_eq!(health_response.status(), 200);
    let health_json: serde_json::Value = health_response.json().await.unwrap();
    assert_eq!(health_json["status"], "healthy");
    assert_eq!(health_json["model_loaded"], true);
    
    // Test model info
    let info_response = client
        .get(&format!("{}/model/info", base_url))
        .send()
        .await
        .expect("Failed to send model info request");
    
    assert_eq!(info_response.status(), 200);
    let info_json: serde_json::Value = info_response.json().await.unwrap();
    assert_eq!(info_json["model_info"]["input_spec"]["shape"], json!([784]));
    assert_eq!(info_json["model_info"]["output_spec"]["shape"], json!([10]));
    
    // Test single prediction
    let predict_payload = json!({
        "input": vec![0.5; 784]
    });
    
    let predict_response = client
        .post(&format!("{}/predict", base_url))
        .json(&predict_payload)
        .send()
        .await
        .expect("Failed to send prediction request");
    
    assert_eq!(predict_response.status(), 200);
    let predict_json: serde_json::Value = predict_response.json().await.unwrap();
    assert_eq!(predict_json["status"], "success");
    assert_eq!(predict_json["batch_size"], 1);
    assert!(predict_json["output"].is_array());
    assert_eq!(predict_json["output"].as_array().unwrap().len(), 10);
    
    // Test batch prediction
    let batch_payload = json!({
        "inputs": vec![
            vec![0.5; 784],
            vec![0.3; 784],
            vec![0.7; 784]
        ]
    });
    
    let batch_response = client
        .post(&format!("{}/predict", base_url))
        .json(&batch_payload)
        .send()
        .await
        .expect("Failed to send batch prediction request");
    
    assert_eq!(batch_response.status(), 200);
    let batch_json: serde_json::Value = batch_response.json().await.unwrap();
    assert_eq!(batch_json["status"], "success");
    assert_eq!(batch_json["batch_size"], 3);
    assert!(batch_json["outputs"].is_array());
    assert_eq!(batch_json["outputs"].as_array().unwrap().len(), 3);
    
    // Test invalid input
    let invalid_payload = json!({
        "input": vec![0.5; 100] // Wrong size
    });
    
    let invalid_response = client
        .post(&format!("{}/predict", base_url))
        .json(&invalid_payload)
        .send()
        .await
        .expect("Failed to send invalid prediction request");
    
    assert_eq!(invalid_response.status(), 400);
    let invalid_json: serde_json::Value = invalid_response.json().await.unwrap();
    assert_eq!(invalid_json["code"], "INPUT_VALIDATION_FAILED");
    
    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_server_startup_with_invalid_model() {
    let model_path = PathBuf::from("nonexistent_model.burn");
    let result = load_model(&model_path);
    
    assert!(result.is_err());
    // Should gracefully handle model loading failure
}

#[tokio::test]
async fn test_concurrent_requests() {
    // Start server in background
    let model_path = PathBuf::from("test_model.burn");
    let model = load_model(&model_path).expect("Failed to load test model");
    
    let server_handle = tokio::spawn(async move {
        start_server("127.0.0.1", 3002, model).await.unwrap();
    });
    
    // Wait for server to start
    sleep(Duration::from_millis(500)).await;
    
    let client = Client::new();
    let base_url = "http://127.0.0.1:3002";
    
    // Send multiple concurrent requests
    let mut handles = Vec::new();
    for _i in 0..10 {
        let client = client.clone();
        let base_url = base_url.to_string();
        let handle = tokio::spawn(async move {
            let predict_payload = json!({
                "input": vec![0.5; 784]
            });
            
            let response = client
                .post(&format!("{}/predict", base_url))
                .json(&predict_payload)
                .send()
                .await
                .expect("Failed to send prediction request");
            
            assert_eq!(response.status(), 200);
            let json: serde_json::Value = response.json().await.unwrap();
            assert_eq!(json["status"], "success");
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Clean up
    server_handle.abort();
}