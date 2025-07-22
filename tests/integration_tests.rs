use furnace::api::start_server;
use furnace::model::load_built_in_model;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_end_to_end_inference() {
    // Start server in background with built-in model
    let model = load_built_in_model("resnet18").expect("Failed to load built-in model");

    let server_handle = tokio::spawn(async move {
        start_server("127.0.0.1", 3001, model).await.unwrap();
    });

    // Wait for server to start
    sleep(Duration::from_millis(500)).await;

    let client = Client::new();
    let base_url = "http://127.0.0.1:3001";

    // Test health check
    let health_response = client
        .get(format!("{base_url}/healthz"))
        .send()
        .await
        .expect("Failed to send health check request");

    assert_eq!(health_response.status(), 200);
    let health_json: serde_json::Value = health_response.json().await.unwrap();
    assert_eq!(health_json["status"], "healthy");
    assert_eq!(health_json["model_loaded"], true);

    // Test model info
    let info_response = client
        .get(format!("{base_url}/model/info"))
        .send()
        .await
        .expect("Failed to send model info request");

    assert_eq!(info_response.status(), 200);
    let info_json: serde_json::Value = info_response.json().await.unwrap();
    assert_eq!(
        info_json["model_info"]["input_spec"]["shape"],
        json!([1, 3, 224, 224])
    );
    assert_eq!(
        info_json["model_info"]["output_spec"]["shape"],
        json!([1000])
    );

    // Test single prediction (ResNet18 input size: 3 * 224 * 224 = 150,528)
    let predict_payload = json!({
        "inputs": [vec![0.5; 150528]]
    });

    let predict_response = client
        .post(format!("{base_url}/predict"))
        .json(&predict_payload)
        .send()
        .await
        .expect("Failed to send prediction request");

    assert_eq!(predict_response.status(), 200);
    let predict_json: serde_json::Value = predict_response.json().await.unwrap();
    assert_eq!(predict_json["status"], "success");
    assert_eq!(predict_json["batch_size"], 1);
    assert!(predict_json["output"].is_array());
    assert_eq!(predict_json["output"].as_array().unwrap().len(), 1000);

    // Test batch prediction (ResNet18 input size: 3 * 224 * 224 = 150,528)
    let batch_payload = json!({
        "inputs": vec![
            vec![0.5; 150528],
            vec![0.3; 150528],
            vec![0.7; 150528]
        ]
    });

    let batch_response = client
        .post(format!("{base_url}/predict"))
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
    // Each output should have 1000 elements (ImageNet classes)
    assert_eq!(batch_json["outputs"][0].as_array().unwrap().len(), 1000);

    // Test invalid input (wrong size for ResNet18)
    let invalid_payload = json!({
        "inputs": [vec![0.5; 100]] // Wrong size - should be 150,528
    });

    let invalid_response = client
        .post(format!("{base_url}/predict"))
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
    let result = load_built_in_model("nonexistent_model");

    assert!(result.is_err());
    // Should gracefully handle model loading failure
}

#[tokio::test]
async fn test_concurrent_requests() {
    // Start server in background with built-in model
    let model = load_built_in_model("resnet18").expect("Failed to load built-in model");

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
                "inputs": [vec![0.5; 150528]] // ResNet18 input size: 3 * 224 * 224 = 150,528
            });

            let response = client
                .post(format!("{base_url}/predict"))
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
