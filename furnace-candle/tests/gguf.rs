//! Real GGUF files, compared with candle's own quantized models, and served end to end. They
//! need the files, so they are ignored by default:
//!
//! ```text
//! FURNACE_QWEN2_GGUF=qwen2.5-0.5b-instruct-q4_k_m.gguf \
//! FURNACE_LLAMA_GGUF=SmolLM2-135M-Instruct-Q4_K_M.gguf \
//!   cargo test --release -p furnace-candle -- --ignored
//! ```

use std::path::PathBuf;

use axum::body::Body;
use axum::http::Request;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen2};
use furnace::Server;
use furnace::llm::{Message, StepModel};
use furnace_candle::Gguf;
use tower::ServiceExt;

fn file(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {variable} to a GGUF file"))
}

/// candle's model of the file, as the reference
enum Reference {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
}

impl Reference {
    fn load(path: &PathBuf) -> Self {
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let architecture = content.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        match architecture.as_str() {
            "llama" => Self::Llama(
                quantized_llama::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                    .unwrap(),
            ),
            "qwen2" => Self::Qwen2(
                quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                    .unwrap(),
            ),
            other => panic!("no reference for {other}"),
        }
    }

    fn forward(&mut self, tokens: &[u32], position: usize) -> Vec<f32> {
        let input = Tensor::new(tokens, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = match self {
            Self::Llama(model) => model.forward(&input, position),
            Self::Qwen2(model) => model.forward(&input, position),
        };
        logits.unwrap().squeeze(0).unwrap().to_vec1().unwrap()
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

/// furnace-candle's forward pass gives candle's logits, prefill and decode, with two sequences
/// interleaved on the same weights
fn matches_candle(variable: &str, knows_paris: bool) {
    let path = file(variable);
    let (model, pipeline) = Gguf::new(&path).load_parts().unwrap();
    let prompt = pipeline
        .encode(
            &pipeline
                .apply_chat_template(&[Message::new("user", "What is the capital of France?")])
                .unwrap(),
            false,
        )
        .unwrap();
    let mut reference = Reference::load(&path);

    let (mut sequence, logits) = model.prefill(&prompt).unwrap();
    // A second sequence in between must not disturb the first: each has its own cache
    let (mut other, _) = model.prefill(&prompt[..prompt.len() / 2]).unwrap();
    let mut expected = reference.forward(&prompt, 0);
    let mut logits = logits.0;
    let mut generated = Vec::new();
    for step in 0..12 {
        let scale = expected.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let difference = logits
            .iter()
            .zip(&expected)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            difference <= 1e-3 * scale.max(1.0),
            "step {step}: logits differ by {difference} (largest {scale})"
        );
        let token = argmax(&expected);
        assert_eq!(argmax(&logits), token, "step {step}");
        generated.push(token);
        logits = model.decode(&mut sequence, token).unwrap().0;
        model.decode(&mut other, token).unwrap();
        expected = reference.forward(&[token], prompt.len() + step);
    }
    let text = pipeline.tokenizer().decode(&generated, true).unwrap();
    println!("{variable}: {text:?}");
    assert!(!knows_paris || text.contains("Paris"), "{text:?}");
}

#[test]
#[ignore = "needs FURNACE_QWEN2_GGUF"]
fn qwen2_matches_candle() {
    matches_candle("FURNACE_QWEN2_GGUF", true);
}

#[test]
#[ignore = "needs FURNACE_LLAMA_GGUF"]
fn llama_matches_candle() {
    // SmolLM2-135M is too small to know; the logits matching candle is the test
    matches_candle("FURNACE_LLAMA_GGUF", false);
}

/// The whole path: a chat request to the server, the GGUF file's chat template and tokenizer,
/// furnace's scheduler and sampling, the answer back
async fn serves_a_chat(variable: &str, knows_paris: bool) {
    let engine = Gguf::new(file(variable)).name("model").load().unwrap();
    let app = Server::new().llm(engine).router();
    let body = serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
        "temperature": 0,
        "max_tokens": 16,
    });
    let request = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), 200);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    println!("{variable}: {content:?}, {}", body["usage"]);
    assert!(!content.trim().is_empty(), "{body}");
    assert!(!knows_paris || content.contains("Paris"), "{body}");
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 10);
}

#[tokio::test]
#[ignore = "needs FURNACE_QWEN2_GGUF"]
async fn qwen2_serves_a_chat() {
    serves_a_chat("FURNACE_QWEN2_GGUF", true).await;
}

#[tokio::test]
#[ignore = "needs FURNACE_LLAMA_GGUF"]
async fn llama_serves_a_chat() {
    serves_a_chat("FURNACE_LLAMA_GGUF", false).await;
}
