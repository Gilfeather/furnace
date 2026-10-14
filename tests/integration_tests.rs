//! End-to-end tests against a real HTTP server running the built-in ResNet-18.
//!
//! Requires `models/resnet18.onnx` at build time (see README).

use furnace::api::{ServerConfig, router};
use furnace::model::load_built_in_model;
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::OnceLock;

const RESNET_INPUT: usize = 3 * 224 * 224;

/// Start a server on an ephemeral port and return its base URL
async fn spawn_server(config: ServerConfig) -> String {
    let model = tokio::task::spawn_blocking(|| load_built_in_model("resnet18"))
        .await
        .unwrap()
        .expect("resnet18 must be compiled in (place models/resnet18.onnx before building)");
    let app = router(model, &config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn image(value: f32) -> Vec<f32> {
    vec![value; RESNET_INPUT]
}

fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(Client::new)
}

#[tokio::test]
async fn test_end_to_end_inference() {
    let base_url = spawn_server(ServerConfig::default()).await;

    // Health check
    let health: Value = client()
        .get(format!("{base_url}/healthz"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "healthy");
    assert_eq!(health["model_loaded"], true);
    assert_eq!(health["backend"], "cpu");

    // Model info
    let info: Value = client()
        .get(format!("{base_url}/model/info"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        info["model_info"]["input_spec"]["shape"],
        json!([3, 224, 224])
    );
    assert_eq!(info["model_info"]["output_spec"]["shape"], json!([1000]));
    assert_eq!(info["model_info"]["model_type"], "onnx");

    // Single prediction: a full ResNet input is well over axum's 2 MB default body limit
    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "input": image(0.5) }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let single: Value = response.json().await.unwrap();
    assert_eq!(single["status"], "success");
    assert_eq!(single["batch_size"], 1);
    let logits = single["output"].as_array().unwrap();
    assert_eq!(logits.len(), 1000);
    // Real model, not a constant dummy output
    assert!(logits.iter().any(|v| v != &logits[0]));

    // Batch prediction: different images give different logits
    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "inputs": [image(0.5), image(0.3), image(0.7)] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let batch: Value = response.json().await.unwrap();
    assert_eq!(batch["batch_size"], 3);
    let outputs = batch["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 3);
    assert!(outputs.iter().all(|o| o.as_array().unwrap().len() == 1000));
    assert_ne!(outputs[0], outputs[1]);

    // Batching must not change results: batch item 0 equals the single prediction
    let single_vals: Vec<f64> = logits.iter().map(|v| v.as_f64().unwrap()).collect();
    let batch_vals: Vec<f64> = outputs[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(
        single_vals
            .iter()
            .zip(&batch_vals)
            .all(|(a, b)| (a - b).abs() < 1e-3)
    );

    // Wrong input size
    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "inputs": [vec![0.5; 100]] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let invalid: Value = response.json().await.unwrap();
    assert_eq!(invalid["code"], "INPUT_VALIDATION_FAILED");
}

#[tokio::test]
async fn test_invalid_model_name() {
    assert!(load_built_in_model("nonexistent_model").is_err());
}

#[tokio::test]
async fn test_concurrent_requests() {
    let base_url = spawn_server(ServerConfig::default()).await;

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let url = format!("{base_url}/predict");
            tokio::spawn(async move {
                let response = client()
                    .post(url)
                    .json(&json!({ "input": image(i as f32 / 10.0) }))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), 200);
                let body: Value = response.json().await.unwrap();
                assert_eq!(body["status"], "success");
            })
        })
        .collect();

    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_backpressure_returns_503() {
    let base_url = spawn_server(ServerConfig {
        max_concurrent_requests: 1,
        ..ServerConfig::default()
    })
    .await;

    // Fire several requests at once against a single inference slot
    let handles: Vec<_> = (0..6)
        .map(|_| {
            let url = format!("{base_url}/predict");
            tokio::spawn(async move {
                client()
                    .post(url)
                    .json(&json!({ "inputs": [image(0.5), image(0.5), image(0.5), image(0.5)] }))
                    .send()
                    .await
                    .unwrap()
                    .status()
                    .as_u16()
            })
        })
        .collect();

    let mut statuses = Vec::new();
    for handle in handles {
        statuses.push(handle.await.unwrap());
    }
    assert!(statuses.contains(&200), "statuses: {statuses:?}");
    assert!(statuses.contains(&503), "statuses: {statuses:?}");
    assert!(statuses.iter().all(|s| *s == 200 || *s == 503));
}
