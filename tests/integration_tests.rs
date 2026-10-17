//! End-to-end tests against a real HTTP server running the built-in models.
//!
//! Requires `models/resnet18.onnx` and `models/mnist.onnx` at build time (see README).

use furnace::api::{ServerConfig, router};
use furnace::model::load_built_in_model;
use reqwest::Client;
use serde_json::{Value, json};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const RESNET_INPUT: usize = 3 * 224 * 224;
const MNIST_INPUT: usize = 28 * 28;

/// Start a ResNet-18 server on an ephemeral port and return its base URL
async fn spawn_server(config: ServerConfig) -> String {
    spawn_model_server("resnet18", config).await
}

/// Start a server for built-in model `name` on an ephemeral port and return its base URL
async fn spawn_model_server(name: &'static str, config: ServerConfig) -> String {
    let model = tokio::task::spawn_blocking(move || load_built_in_model(name))
        .await
        .unwrap()
        .unwrap_or_else(|e| {
            panic!("{name} must be compiled in (place models/{name}.onnx before building): {e}")
        });
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

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn json_floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3));
}

#[tokio::test]
async fn test_end_to_end_binary_input() {
    let base_url = spawn_server(ServerConfig::default()).await;
    // A non-constant image so a mis-ordered decode would change the result
    let ramp: Vec<f32> = (0..RESNET_INPUT)
        .map(|i| (i % 224) as f32 / 112.0 - 1.0)
        .collect();

    // Reference: the same image through the JSON API
    let reference: Value = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "input": ramp }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let reference = json_floats(&reference["output"]);

    // Binary in, JSON out: same response shape as `{"input": ...}`
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "application/octet-stream")
        .body(le_bytes(&ramp))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let single: Value = response.json().await.unwrap();
    assert_eq!(single["batch_size"], 1);
    assert_close(&json_floats(&single["output"]), &reference);

    // Binary in, binary out, batch of 2
    let mut body = le_bytes(&image(0.5));
    body.extend(le_bytes(&ramp));
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "application/octet-stream")
        .header("accept", "application/octet-stream")
        .header("x-batch-size", "2")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["content-type"],
        "application/octet-stream"
    );
    assert_eq!(response.headers()["x-batch-size"], "2");
    assert_eq!(response.headers()["x-output-shape"], "1000");
    let bytes = response.bytes().await.unwrap();
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    let outputs: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
    assert_eq!(outputs.len(), 2 * 1000);
    assert_ne!(outputs[..1000], outputs[1000..]);
    assert_close(&outputs[1000..], &reference);

    // Size mismatch against X-Batch-Size
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "application/octet-stream")
        .header("x-batch-size", "2")
        .body(le_bytes(&ramp))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let invalid: Value = response.json().await.unwrap();
    assert_eq!(invalid["code"], "INPUT_VALIDATION_FAILED");
    assert_eq!(invalid["details"]["received_bytes"], RESNET_INPUT * 4);
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

/// A distinct, non-constant image per `seed`, so mixed-up outputs would show
fn ramp(seed: usize) -> Vec<f32> {
    (0..RESNET_INPUT)
        .map(|i| ((i * (seed + 1)) % 224) as f32 / 112.0 - 1.0)
        .collect()
}

/// Concurrent requests batched into shared forward passes get the same outputs, in the
/// same order, as each image run on its own, and a bad request fails alone
#[tokio::test]
async fn test_dynamic_batching_matches_unbatched() {
    use furnace::api::BatchConfig;

    let unbatched = spawn_server(ServerConfig {
        batching: BatchConfig {
            max_batch_size: 1,
            ..BatchConfig::default()
        },
        ..ServerConfig::default()
    })
    .await;
    let batched = spawn_server(ServerConfig {
        batching: BatchConfig {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(5),
            ..BatchConfig::default()
        },
        ..ServerConfig::default()
    })
    .await;

    let mut reference = Vec::new();
    for seed in 0..10 {
        let response: Value = client()
            .post(format!("{unbatched}/predict"))
            .header("content-type", "application/octet-stream")
            .body(le_bytes(&ramp(seed)))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        reference.push(json_floats(&response["output"]));
    }

    let mut nan = ramp(0);
    nan[123] = f32::NAN;
    let url = format!("{batched}/predict");
    let requests = vec![
        // (request, seeds of its items; empty for the invalid one)
        (
            client().post(&url).json(&json!({ "input": ramp(0) })),
            vec![0],
        ),
        (
            client()
                .post(&url)
                .header("content-type", "application/octet-stream")
                .body(le_bytes(&ramp(1))),
            vec![1],
        ),
        (
            client()
                .post(&url)
                .json(&json!({ "inputs": [ramp(2), ramp(3)] })),
            vec![2, 3],
        ),
        (
            client()
                .post(&url)
                .header("content-type", "application/octet-stream")
                .body(le_bytes(&nan)),
            vec![],
        ),
        (
            client()
                .post(&url)
                .header("content-type", "application/octet-stream")
                .header("x-batch-size", "2")
                .header("accept", "application/octet-stream")
                .body(le_bytes(&[ramp(4), ramp(5)].concat())),
            vec![4, 5],
        ),
        (
            client().post(&url).json(&json!({ "input": ramp(6) })),
            vec![6],
        ),
        (
            client().post(&url).json(&json!({ "input": ramp(7) })),
            vec![7],
        ),
        (
            client().post(&url).json(&json!({ "input": ramp(8) })),
            vec![8],
        ),
        (
            client().post(&url).json(&json!({ "input": ramp(9) })),
            vec![9],
        ),
    ];

    let handles: Vec<_> = requests
        .into_iter()
        .map(|(request, seeds)| {
            tokio::spawn(async move {
                let response = request.send().await.unwrap();
                let status = response.status().as_u16();
                let binary = response.headers()["content-type"] == "application/octet-stream";
                let body = response.bytes().await.unwrap();
                (seeds, status, binary, body)
            })
        })
        .collect();

    for handle in handles {
        let (seeds, status, binary, body) = handle.await.unwrap();
        if seeds.is_empty() {
            assert_eq!(status, 400);
            let error: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(error["code"], "INVALID_INPUT_VALUE");
            continue;
        }
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
        let outputs: Vec<Vec<f32>> = if binary {
            let (values, _) = body.as_chunks::<4>();
            let values: Vec<f32> = values.iter().copied().map(f32::from_le_bytes).collect();
            values.chunks(1000).map(<[f32]>::to_vec).collect()
        } else {
            let doc: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(doc["batch_size"], seeds.len());
            match doc.get("output") {
                Some(output) => vec![json_floats(output)],
                None => doc["outputs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(json_floats)
                    .collect(),
            }
        };
        assert_eq!(outputs.len(), seeds.len());
        for (output, seed) in outputs.iter().zip(&seeds) {
            assert_close(output, &reference[*seed]);
        }
    }

    let metrics: Value = client()
        .get(format!("{batched}/metrics"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Every request reached the model; the valid ones shared forward passes
    assert_eq!(metrics["total_inferences"], 9);
    assert_eq!(metrics["max_batch_size"], 8);
    let batches = metrics["total_batches"].as_u64().unwrap();
    assert!((1..8).contains(&batches), "{metrics}");
    assert!(metrics["average_batch_size"].as_f64().unwrap() > 1.0);
    assert_eq!(metrics["in_flight_inferences"], 0);
}

/// A second built-in model with different shapes, generated from models/mnist.onnx with no
/// model-specific code
#[tokio::test]
async fn test_end_to_end_second_model() {
    let base_url = spawn_model_server("mnist", ServerConfig::default()).await;

    let info: Value = client()
        .get(format!("{base_url}/model/info"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(info["model_info"]["name"], "mnist");
    assert_eq!(
        info["model_info"]["input_spec"]["shape"],
        json!([1, 28, 28])
    );
    assert_eq!(info["model_info"]["output_spec"]["shape"], json!([10]));

    // A vertical bar and a horizontal bar should not look alike to the classifier
    let vertical: Vec<f32> = (0..MNIST_INPUT)
        .map(|i| {
            if (12..16).contains(&(i % 28)) {
                255.0
            } else {
                0.0
            }
        })
        .collect();
    let horizontal: Vec<f32> = (0..MNIST_INPUT)
        .map(|i| {
            if (12..16).contains(&(i / 28)) {
                255.0
            } else {
                0.0
            }
        })
        .collect();
    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "inputs": [vertical, horizontal] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let batch: Value = response.json().await.unwrap();
    let outputs = batch["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 2);
    let (a, b) = (json_floats(&outputs[0]), json_floats(&outputs[1]));
    assert_eq!(a.len(), 10);
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    assert_ne!(a, b);

    // Binary in, binary out uses the model's own output shape
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "application/octet-stream")
        .header("accept", "application/octet-stream")
        .body(le_bytes(&vertical))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-output-shape"], "10");
    let bytes = response.bytes().await.unwrap();
    assert_eq!(bytes.len(), 10 * 4);

    // A ResNet-sized input is rejected
    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "input": image(0.5) }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

/// Kills the server process when the test ends, even on panic
struct ServerProcess(Child);

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn furnace() -> Command {
    Command::new(env!("CARGO_BIN_EXE_furnace"))
}

/// `furnace --model-name mnist` serves the model, through the real binary and CLI
#[tokio::test]
async fn test_cli_serves_model_by_name() {
    // Grab a free port, then hand it to the server
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let _server = ServerProcess(
        furnace()
            .args(["--model-name", "MNIST", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );

    let base_url = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(60);
    let info: Value = loop {
        if let Ok(response) = client().get(format!("{base_url}/model/info")).send().await {
            break response.json().await.unwrap();
        }
        assert!(Instant::now() < deadline, "server did not start");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(info["model_info"]["name"], "mnist");

    let response = client()
        .post(format!("{base_url}/predict"))
        .json(&json!({ "input": vec![0.0; MNIST_INPUT] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[test]
fn test_cli_lists_available_models() {
    let help = furnace().arg("--help").output().unwrap();
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("Available models: mnist, resnet18"), "{help}");

    let unknown = furnace()
        .args(["--model-name", "no-such-model"])
        .output()
        .unwrap();
    assert!(!unknown.status.success());
    let stderr = String::from_utf8(unknown.stderr).unwrap();
    assert!(stderr.contains("unknown built-in model"), "{stderr}");
    assert!(stderr.contains("mnist, resnet18"), "{stderr}");
}

/// Image input: the server's preprocessing must give the same logits as sending the
/// preprocessed floats, and JPEG input must work end to end
#[cfg(feature = "image")]
#[tokio::test]
async fn test_end_to_end_image_input() {
    use furnace::preprocess::{ImageKind, preprocess_image};
    use image::codecs::jpeg::JpegEncoder;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder, Rgb, RgbImage};

    let base_url = spawn_server(ServerConfig::default()).await;

    // A 320x240 synthetic photo-like image: gradients plus a bright disc off centre
    let image = RgbImage::from_fn(320, 240, |x, y| {
        let (dx, dy) = (x as i32 - 190, y as i32 - 110);
        if dx * dx + dy * dy < 50 * 50 {
            Rgb([240, 220, 60])
        } else {
            Rgb([(x * 255 / 320) as u8, (y * 255 / 240) as u8, 90])
        }
    });
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(image.as_raw(), 320, 240, ExtendedColorType::Rgb8)
        .unwrap();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 90)
        .write_image(image.as_raw(), 320, 240, ExtendedColorType::Rgb8)
        .unwrap();

    // Reference: preprocess on the client and send floats
    let floats = preprocess_image(&png, ImageKind::Png, 224, 224).unwrap();
    assert_eq!(floats.len(), RESNET_INPUT);
    let reference: Value = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "application/octet-stream")
        .body(le_bytes(&floats))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let reference = json_floats(&reference["output"]);

    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "image/png")
        .body(png)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let from_png: Value = response.json().await.unwrap();
    assert_eq!(from_png["batch_size"], 1);
    assert_close(&json_floats(&from_png["output"]), &reference);

    // JPEG is lossy, so only check that it decodes to the same top-1 class
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "image/jpeg")
        .body(jpeg)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let from_jpeg: Value = response.json().await.unwrap();
    let argmax = |logits: &[f32]| {
        (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap()
    };
    assert_eq!(
        argmax(&json_floats(&from_jpeg["output"])),
        argmax(&reference)
    );

    // Not an image
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "image/jpeg")
        .body("hello")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let invalid: Value = response.json().await.unwrap();
    assert_eq!(invalid["code"], "INVALID_IMAGE");
}

/// Image input needs a `[3, H, W]` model: MNIST (`[1, 28, 28]`) rejects it, with or without
/// the `image` feature
#[tokio::test]
async fn test_image_input_rejected_for_non_rgb_model() {
    let base_url = spawn_model_server("mnist", ServerConfig::default()).await;
    // A PNG signature is enough: the shape (or content type) is checked before decoding
    let response = client()
        .post(format!("{base_url}/predict"))
        .header("content-type", "image/png")
        .body(b"\x89PNG\r\n\x1a\n".to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 415);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["code"], "INVALID_REQUEST");
    #[cfg(feature = "image")]
    assert_eq!(error["details"]["expected_shape"], json!([1, 28, 28]));
}
