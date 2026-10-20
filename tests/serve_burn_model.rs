//! Serving models written in Burn through the public library API (`furnace::Server`,
//! `Model::from_fn`, `BurnModel`): the server's answers must match calling `forward` directly.

#[path = "../examples/serve_burn_model/models.rs"]
mod models;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use burn::tensor::{Int, Tensor, TensorData};
use furnace::{BackendKind, Model, Server};
use models::{DigitNet, Sentiment, SentimentNet};
use tower::ServiceExt;

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, serde_json::Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap_or_default())
}

fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .uri(path)
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|v| match v.as_array() {
            Some(row) => row.iter().map(|x| x.as_f64().unwrap() as f32).collect(),
            None => vec![v.as_f64().unwrap() as f32],
        })
        .collect()
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-5 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

/// Both example models behind one router, and the nets themselves for reference outputs
fn app() -> (axum::Router, DigitNet, SentimentNet) {
    let backend = BackendKind::Cpu;
    let device = backend.device();
    let digits = DigitNet::new(&device);
    let sentiment = SentimentNet::new(&device);
    let served_digits = digits.clone();
    let router = Server::new()
        .model(
            Model::from_fn(
                "digits",
                backend,
                &[1, 28, 28],
                &[10],
                move |images: Tensor<4>| served_digits.forward(images),
            )
            .unwrap(),
        )
        .model(Model::custom(Sentiment {
            net: sentiment.clone(),
            backend,
            device,
            seq_len: 8,
        }))
        .router()
        .unwrap();
    (router, digits, sentiment)
}

#[tokio::test]
async fn a_burn_forward_function_serves_predict_and_v2() {
    let (app, digits, _) = app();
    let device = BackendKind::Cpu.device();
    let images: Vec<f32> = (0..2 * 28 * 28).map(|i| (i % 255) as f32 / 255.0).collect();
    let expected = digits
        .forward(Tensor::from_data(
            TensorData::new(images.clone(), [2, 1, 28, 28]),
            &device,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();

    // /predict uses the first model, with the usual flat JSON
    let (status, body) = send(
        &app,
        post(
            "/predict",
            serde_json::json!({ "inputs": [&images[..784], &images[784..]] }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    close(&floats(&body["outputs"]), &expected);

    // The Open Inference Protocol, as Triton clients send it
    let (status, body) = send(
        &app,
        post(
            "/v2/models/digits/infer",
            serde_json::json!({
                "inputs": [{ "name": "input", "datatype": "FP32", "shape": [2, 1, 28, 28], "data": images }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outputs"][0]["shape"], serde_json::json!([2, 10]));
    close(&floats(&body["outputs"][0]["data"]), &expected);
}

#[tokio::test]
async fn a_burn_model_with_named_integer_tensors() {
    let (app, _, sentiment) = app();
    let device = BackendKind::Cpu.device();

    let (status, body) = send(
        &app,
        Request::builder()
            .uri("/v2/models/sentiment")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["inputs"][0]["name"], "input_ids");
    assert_eq!(body["inputs"][0]["datatype"], "INT64");
    assert_eq!(body["inputs"][0]["shape"], serde_json::json!([-1, 8]));
    assert_eq!(body["outputs"][0]["name"], "logits");

    let ids = [
        101i64, 7, 42, 9, 102, 0, 0, 0, 101, 2000, 3000, 102, 0, 0, 0, 0,
    ];
    let mask: Vec<i64> = ids.iter().map(|&id| i64::from(id != 0)).collect();
    let expected = sentiment
        .forward(
            Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [2, 8]), &device),
            Tensor::<2, Int>::from_data(TensorData::new(mask.clone(), [2, 8]), &device),
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();

    let (status, body) = send(
        &app,
        post(
            "/v2/models/sentiment/infer",
            serde_json::json!({
                "inputs": [
                    { "name": "attention_mask", "datatype": "INT64", "shape": [2, 8], "data": mask },
                    { "name": "input_ids", "datatype": "INT64", "shape": [2, 8], "data": ids },
                ],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outputs"][0]["name"], "logits");
    assert_eq!(body["outputs"][0]["shape"], serde_json::json!([2, 2]));
    close(&floats(&body["outputs"][0]["data"]), &expected);

    // A sequence of the wrong length is refused before it reaches the model
    let (status, body) = send(
        &app,
        post(
            "/v2/models/sentiment/infer",
            serde_json::json!({
                "inputs": [
                    { "name": "input_ids", "datatype": "INT64", "shape": [1, 7], "data": [1, 2, 3, 4, 5, 6, 7] },
                    { "name": "attention_mask", "datatype": "INT64", "shape": [1, 7], "data": [1, 1, 1, 1, 1, 1, 1] },
                ],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"].as_str().unwrap().contains("input_ids"),
        "{body}"
    );
}

#[test]
fn from_fn_checks_the_shapes_against_the_ranks() {
    let err = Model::from_fn(
        "digits",
        BackendKind::Cpu,
        &[28, 28],
        &[10],
        |images: Tensor<4>| images.flatten::<2>(1, 3),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("rank 4"), "{err}");
}

#[tokio::test]
async fn serving_needs_a_model() {
    let err = Server::new().serve().await.unwrap_err().to_string();
    assert!(err.contains("no model"), "{err}");
}
