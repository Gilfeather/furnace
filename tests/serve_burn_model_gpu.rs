//! The library API on a GPU (feature `wgpu`): the same Burn models, with the same weights, served
//! on the wgpu backend must answer what they compute on the CPU.
#![cfg(feature = "wgpu")]

#[path = "../examples/serve_burn_model/models.rs"]
mod models;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use furnace::{BackendKind, Model, Server};
use models::{DigitNet, Sentiment, SentimentNet};
use tower::ServiceExt;

async fn post(app: &axum::Router, path: &str, body: serde_json::Value) -> serde_json::Value {
    let request = Request::builder()
        .uri(path)
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    body
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

/// GPU and CPU float results agree up to the GPU's accumulation order
fn close(gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len());
    for (x, y) in gpu.iter().zip(cpu) {
        assert!(
            (x - y).abs() <= 1e-3 * (1.0 + y.abs()),
            "gpu {x} vs cpu {y}"
        );
    }
}

/// `module`'s weights, saved on the CPU and loaded into a copy built on `device`
fn to_device<M: Module + ModuleSnapshot>(module: &M, mut copy: M, dir: &std::path::Path) -> M {
    let path = dir.join("weights.bpk");
    module
        .save_into(&mut BurnpackStore::from_file(&path).overwrite(true))
        .unwrap();
    furnace::weights::load_burnpack(&mut copy, &path).unwrap();
    copy
}

#[tokio::test]
async fn burn_models_on_wgpu_match_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let cpu = BackendKind::Cpu.device();
    let gpu = BackendKind::Wgpu.device();
    let digits = DigitNet::new(&cpu);
    let sentiment = SentimentNet::new(&cpu);
    let digits_gpu = to_device(&digits, DigitNet::new(&gpu), dir.path());
    let sentiment_gpu = to_device(&sentiment, SentimentNet::new(&gpu), dir.path());

    let app = Server::new()
        .model(
            Model::from_fn(
                "digits",
                BackendKind::Wgpu,
                &[1, 28, 28],
                &[10],
                move |images: Tensor<4>| digits_gpu.forward(images),
            )
            .unwrap(),
        )
        .model(Model::custom(Sentiment {
            net: sentiment_gpu,
            backend: BackendKind::Wgpu,
            device: gpu,
            seq_len: 8,
        }))
        .router()
        .unwrap();

    // One float tensor, through /predict and its batching
    let images: Vec<f32> = (0..3 * 784).map(|i| (i % 255) as f32 / 255.0).collect();
    let expected = digits
        .forward(Tensor::from_data(
            TensorData::new(images.clone(), [3, 1, 28, 28]),
            &cpu,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let items: Vec<&[f32]> = images.chunks(784).collect();
    let body = post(&app, "/predict", serde_json::json!({ "inputs": items })).await;
    close(&floats(&body["outputs"]), &expected);

    // Named INT64 tensors, through v2
    let ids = [101i64, 7, 42, 9, 102, 0, 0, 0];
    let mask: Vec<i64> = ids.iter().map(|&id| i64::from(id != 0)).collect();
    let expected = sentiment
        .forward(
            Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, 8]), &cpu),
            Tensor::<2, Int>::from_data(TensorData::new(mask.clone(), [1, 8]), &cpu),
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let body = post(
        &app,
        "/v2/models/sentiment/infer",
        serde_json::json!({
            "inputs": [
                { "name": "input_ids", "datatype": "INT64", "shape": [1, 8], "data": ids },
                { "name": "attention_mask", "datatype": "INT64", "shape": [1, 8], "data": mask },
            ],
        }),
    )
    .await;
    close(&floats(&body["outputs"][0]["data"]), &expected);
}
