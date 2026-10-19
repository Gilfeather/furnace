//! Serve a GGUF file with the OpenAI API on http://127.0.0.1:8080/v1, for instance
//! Qwen2.5-0.5B-Instruct at Q4_K_M:
//!
//! ```text
//! cargo run --release -p furnace-candle --example serve_gguf -- qwen2.5-0.5b-instruct-q4_k_m.gguf
//!
//! curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
//!   -d '{"model": "qwen2.5-0.5b-instruct-q4_k_m", "stream": true,
//!        "messages": [{"role": "user", "content": "Name three Rust web frameworks."}]}'
//! ```
//!
//! Any OpenAI client works with `base_url = "http://127.0.0.1:8080/v1"`. The Prometheus metrics
//! (time to first token, inter-token latency, tokens) are on `GET /metrics` with
//! `Accept: text/plain`.

use furnace::Server;
use furnace::llm::BoxError;
use furnace_candle::Gguf;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: serve_gguf <model.gguf> [port]")?;
    let port = match std::env::args().nth(2) {
        Some(port) => port.parse()?,
        None => 8080,
    };
    let engine = Gguf::new(path).load()?;
    Server::new().port(port).llm(engine).serve().await?;
    Ok(())
}
