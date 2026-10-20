//! The LLM layer through the public API (feature `llm`): the OpenAI routes and their streaming
//! format, stop strings, cancellation on disconnect, backpressure and metrics, with the mock
//! engines.

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use furnace::llm::mock::{self, MockEngine, MockStepModel};
use furnace::llm::{
    Event, FinishReason, GenerateRequest, LlmEngine, Message, SamplingParams, Scheduled,
    SchedulerConfig,
};
use furnace::model::DummyModel;
use furnace::{Model, Server};
use serde_json::{Value, json};
use tower::ServiceExt;

const REPLY: &str = "Hello, wörld! ✓ done.";

fn scheduled(model: MockStepModel) -> Scheduled<MockStepModel> {
    Scheduled::new(model, mock::byte_pipeline())
}

fn app_with(model: MockStepModel) -> axum::Router {
    Server::new().llm(scheduled(model)).router()
}

fn app() -> axum::Router {
    app_with(MockStepModel::new("mock", REPLY))
}

fn post(path: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .uri(path)
        .method("POST")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::builder().uri(path).body(Body::empty()).unwrap()
}

struct Answer {
    status: StatusCode,
    content_type: String,
    body: String,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.body))
    }

    /// The `data:` payloads of an SSE body
    fn events(&self) -> Vec<String> {
        self.body
            .split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }

    /// The payloads before `[DONE]`, parsed
    fn chunks(&self) -> Vec<Value> {
        let events = self.events();
        assert_eq!(
            events.last().map(String::as_str),
            Some("[DONE]"),
            "{}",
            self.body
        );
        events[..events.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect()
    }
}

async fn send(app: &axum::Router, request: Request<Body>) -> Answer {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    Answer {
        status,
        content_type,
        body: String::from_utf8(body.to_vec()).unwrap(),
    }
}

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "Say hello"}],
        "temperature": 0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// An OpenAI error: `{"error": {"message", "type", "param", "code"}}`
fn assert_openai_error(answer: &Answer, status: StatusCode, kind: &str) -> Value {
    assert_eq!(answer.status, status, "{}", answer.body);
    let error = answer.json()["error"].clone();
    assert!(
        error["message"].as_str().is_some_and(|m| !m.is_empty()),
        "{error}"
    );
    assert_eq!(error["type"], kind, "{error}");
    assert!(
        error.get("param").is_some() && error.get("code").is_some(),
        "{error}"
    );
    error
}

#[tokio::test]
async fn chat_streaming_follows_the_openai_format() {
    let answer = send(
        &app(),
        post(
            "/v1/chat/completions",
            &chat(json!({"stream": true, "stream_options": {"include_usage": true}})),
        ),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    assert!(
        answer.content_type.starts_with("text/event-stream"),
        "{}",
        answer.content_type
    );
    let chunks = answer.chunks();

    let id = chunks[0]["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-"), "{id}");
    for chunk in &chunks {
        assert_eq!(chunk["id"], id);
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["model"], "mock");
        assert!(chunk["created"].as_u64().unwrap() > 1_700_000_000);
    }
    // The role first, with empty content
    let first = &chunks[0]["choices"][0];
    assert_eq!(first["index"], 0);
    assert_eq!(first["delta"], json!({"role": "assistant", "content": ""}));
    assert_eq!(first["finish_reason"], Value::Null);
    // Then the content, whole characters only
    let deltas = &chunks[1..chunks.len() - 2];
    assert!(deltas.len() > 1, "streamed in pieces");
    let mut content = String::new();
    for chunk in deltas {
        let choice = &chunk["choices"][0];
        assert_eq!(choice["finish_reason"], Value::Null);
        assert!(choice["delta"].get("role").is_none());
        let piece = choice["delta"]["content"].as_str().unwrap();
        assert!(
            !piece.is_empty() && !piece.contains('\u{FFFD}'),
            "{piece:?}"
        );
        content.push_str(piece);
    }
    assert_eq!(content, REPLY);
    // The finish chunk with an empty delta
    let finish = &chunks[chunks.len() - 2]["choices"][0];
    assert_eq!(finish["delta"], json!({}));
    assert_eq!(finish["finish_reason"], "stop");
    // With include_usage: the usage chunk last, without choices; `usage: null` everywhere else
    let usage_chunk = chunks.last().unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    let usage = &usage_chunk["usage"];
    let prompt_tokens = "<|user|>Say hello\n<|assistant|>".len() as u64;
    assert_eq!(usage["prompt_tokens"], prompt_tokens);
    assert_eq!(usage["completion_tokens"], REPLY.len() as u64);
    assert_eq!(usage["total_tokens"], prompt_tokens + REPLY.len() as u64);
    for chunk in &chunks[..chunks.len() - 1] {
        assert_eq!(chunk["usage"], Value::Null);
        assert!(chunk.get("usage").is_some());
    }
}

#[tokio::test]
async fn without_include_usage_chunks_have_no_usage() {
    let answer = send(
        &app(),
        post("/v1/chat/completions", &chat(json!({"stream": true}))),
    )
    .await;
    let chunks = answer.chunks();
    assert!(chunks.iter().all(|c| c.get("usage").is_none()));
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn chat_without_streaming_answers_at_once() {
    let answer = send(&app(), post("/v1/chat/completions", &chat(json!({})))).await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.body);
    let body = answer.json();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(
        body["choices"][0]["message"],
        json!({"role": "assistant", "content": REPLY})
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], REPLY.len() as u64);
}

#[tokio::test]
async fn completions_stream_text_chunks() {
    let request = json!({"model": "mock", "prompt": "Once", "stream": true, "temperature": 0,
        "stream_options": {"include_usage": true}});
    let answer = send(&app(), post("/v1/completions", &request)).await;
    let chunks = answer.chunks();
    assert!(chunks.iter().all(|c| c["object"] == "text_completion"));
    assert!(chunks[0]["id"].as_str().unwrap().starts_with("cmpl-"));
    let text: String = chunks[..chunks.len() - 1]
        .iter()
        .map(|c| c["choices"][0]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text, REPLY);
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        "stop"
    );
    assert_eq!(chunks.last().unwrap()["usage"]["prompt_tokens"], 4);

    let answer = send(
        &app(),
        post(
            "/v1/completions",
            &json!({"prompt": ["Once"], "max_tokens": 5, "temperature": 0}),
        ),
    )
    .await;
    let body = answer.json();
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], &REPLY[..5]);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 5);
}

#[tokio::test]
async fn stop_strings_match_across_token_boundaries() {
    // One token per byte: every stop string spans several tokens, "ö" and "✓" are split too
    for (stop, expected) in [
        (json!("wörld"), "Hello, "),
        (json!([" done", "✓"]), "Hello, wörld! "),
        (json!("nowhere"), REPLY),
    ] {
        for stream in [false, true] {
            let answer = send(
                &app(),
                post(
                    "/v1/chat/completions",
                    &chat(json!({"stop": stop, "stream": stream})),
                ),
            )
            .await;
            let (content, reason) = if stream {
                let chunks = answer.chunks();
                let content: String = chunks
                    .iter()
                    .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
                    .collect();
                let reason = chunks.last().unwrap()["choices"][0]["finish_reason"].clone();
                (content, reason)
            } else {
                let body = answer.json();
                let content = body["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap()
                    .to_string();
                (content, body["choices"][0]["finish_reason"].clone())
            };
            assert_eq!(content, expected, "stop {stop}, stream {stream}");
            assert_eq!(reason, "stop");
        }
    }
}

#[tokio::test]
async fn max_tokens_ends_with_length() {
    for field in ["max_tokens", "max_completion_tokens"] {
        let answer = send(
            &app(),
            post("/v1/chat/completions", &chat(json!({field: 3}))),
        )
        .await;
        let body = answer.json();
        assert_eq!(body["choices"][0]["message"]["content"], "Hel");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }
}

#[tokio::test]
async fn llms_and_tensor_models_share_one_server() {
    let app = Server::new()
        .model(Model::new(
            Box::new(DummyModel::new("digits".to_string(), vec![4], vec![2])),
            "dummy",
            0,
        ))
        .llm(scheduled(MockStepModel::new("mock", REPLY)))
        .llm(MockEngine::new("other", &["a", "b"]))
        .router();

    let models = send(&app, get("/v1/models")).await.json();
    assert_eq!(models["object"], "list");
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["mock", "other", "digits"]);
    assert_eq!(models["data"][0]["object"], "model");
    assert_eq!(models["data"][0]["max_model_len"], 4096);

    let predict = send(
        &app,
        post("/predict", &json!({"input": [1.0, 2.0, 3.0, 4.0]})),
    )
    .await;
    assert_eq!(predict.status, StatusCode::OK, "{}", predict.body);

    let other = send(
        &app,
        post(
            "/v1/chat/completions",
            &json!({"model": "other",
        "messages": [{"role": "user", "content": "x"}]}),
        ),
    )
    .await;
    assert_eq!(other.json()["choices"][0]["message"]["content"], "ab");
    // Without `model`, the first LLM
    let first = send(
        &app,
        post(
            "/v1/chat/completions",
            &json!({
        "messages": [{"role": "user", "content": "x"}], "temperature": 0}),
        ),
    )
    .await;
    assert_eq!(first.json()["model"], "mock");

    let not_llm = send(
        &app,
        post(
            "/v1/chat/completions",
            &json!({"model": "digits",
        "messages": [{"role": "user", "content": "x"}]}),
        ),
    )
    .await;
    assert_openai_error(&not_llm, StatusCode::BAD_REQUEST, "invalid_request_error");
}

#[tokio::test]
async fn a_server_of_llms_alone_answers_its_probes() {
    let app = app();
    assert_eq!(send(&app, get("/healthz")).await.status, StatusCode::OK);
    assert_eq!(send(&app, get("/readyz")).await.status, StatusCode::OK);
    let models = send(&app, get("/models")).await.json();
    assert_eq!(models["default_model"], "");
    let predict = send(&app, post("/predict", &json!({"input": [1.0]}))).await;
    assert_eq!(predict.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn bad_requests_get_openai_errors() {
    let app = app();
    let cases = [
        (
            json!({"model": "gpt-4"}),
            StatusCode::NOT_FOUND,
            Some("model_not_found"),
        ),
        (json!({"n": 2}), StatusCode::BAD_REQUEST, None),
        (json!({"logprobs": true}), StatusCode::BAD_REQUEST, None),
        (
            json!({"tools": [{"type": "function"}]}),
            StatusCode::BAD_REQUEST,
            None,
        ),
        (
            json!({"response_format": {"type": "json_object"}}),
            StatusCode::BAD_REQUEST,
            None,
        ),
        (json!({"temperature": 3}), StatusCode::BAD_REQUEST, None),
        (
            json!({"stop": ["a", "b", "c", "d", "e"]}),
            StatusCode::BAD_REQUEST,
            None,
        ),
        (json!({"max_tokens": 0}), StatusCode::BAD_REQUEST, None),
        (json!({"messages": []}), StatusCode::BAD_REQUEST, None),
        (
            json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
            StatusCode::BAD_REQUEST,
            None,
        ),
    ];
    for (extra, status, code) in cases {
        let answer = send(&app, post("/v1/chat/completions", &chat(extra.clone()))).await;
        let kind = "invalid_request_error";
        let error = assert_openai_error(&answer, status, kind);
        if let Some(code) = code {
            assert_eq!(error["code"], code, "{extra}");
        }
    }
    let request = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    assert_openai_error(
        &send(&app, request).await,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    );

    // Text parts are joined; unknown fields are ignored
    let answer = send(&app, post("/v1/chat/completions", &chat(json!({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say "}, {"type": "text", "text": "hi"}]}],
        "user": "someone", "logit_bias": {}, "parallel_tool_calls": false,
    })))).await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.body);
}

#[tokio::test]
async fn a_prompt_longer_than_the_context_is_refused() {
    let app = app_with(MockStepModel::new("mock", REPLY).max_context(16));
    let answer = send(
        &app,
        post(
            "/v1/chat/completions",
            &chat(json!({
                "messages": [{"role": "user", "content": "far too long for sixteen tokens"}]
            })),
        ),
    )
    .await;
    let error = assert_openai_error(&answer, StatusCode::BAD_REQUEST, "invalid_request_error");
    assert_eq!(error["code"], "context_length_exceeded");
    assert_eq!(error["param"], "messages");
}

#[tokio::test]
async fn the_concurrency_limit_answers_503() {
    let app = Server::new()
        .configure(|c| c.max_concurrent_requests = 1)
        .llm(MockEngine::new("mock", &["a"; 50]).piece_delay(Duration::from_millis(20)))
        .router();
    // The first stream holds the only slot while its body is open
    let open = app
        .clone()
        .oneshot(post("/v1/chat/completions", &chat(json!({"stream": true}))))
        .await
        .unwrap();
    assert_eq!(open.status(), StatusCode::OK);
    let refused = send(&app, post("/v1/chat/completions", &chat(json!({})))).await;
    let error = assert_openai_error(&refused, StatusCode::SERVICE_UNAVAILABLE, "server_error");
    assert_eq!(error["code"], "server_busy");
    drop(open);
    // The slot is free again once the stream is gone
    let answer = send(
        &app,
        post("/v1/chat/completions", &chat(json!({"max_tokens": 1}))),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.body);

    let metrics = send(&app, get("/metrics")).await.json();
    assert_eq!(metrics["llm"][0]["requests_rejected"], 1);
}

#[tokio::test]
async fn a_full_scheduler_answers_503() {
    let model = MockStepModel::new("mock", &"x".repeat(200))
        .max_sequences(1)
        .token_delay(Duration::from_millis(10));
    let live = model.live();
    let engine = Scheduled::with_config(
        model,
        mock::byte_pipeline(),
        SchedulerConfig {
            max_waiting: 1,
            ..SchedulerConfig::default()
        },
    );
    let app = Server::new().llm(engine).router();
    let stream = || {
        app.clone()
            .oneshot(post("/v1/chat/completions", &chat(json!({"stream": true}))))
    };
    let running = stream().await.unwrap();
    wait_for(|| live.get() == 1).await;
    // A stream's status waits for its first event, so a queued one has no response yet
    let waiting = tokio::spawn(stream());
    let deadline = Instant::now() + Duration::from_secs(5);
    while send(&app, get("/metrics")).await.json()["llm"][0]["requests_waiting"] != 1 {
        assert!(Instant::now() < deadline, "the second request never queued");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let refused = send(&app, post("/v1/chat/completions", &chat(json!({})))).await;
    let error = assert_openai_error(&refused, StatusCode::SERVICE_UNAVAILABLE, "server_error");
    assert_eq!(error["code"], "server_busy");
    assert_eq!(running.status(), StatusCode::OK);
    // A waiting request whose client left frees its place in the queue, as soon as its task
    // has seen the client go: a new one is queued (no answer yet) rather than refused at once
    waiting.abort();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match tokio::time::timeout(Duration::from_millis(50), stream()).await {
            Err(_) => break,
            Ok(answer) => assert_eq!(answer.unwrap().status(), StatusCode::SERVICE_UNAVAILABLE),
        }
        assert!(Instant::now() < deadline, "the queue stayed full");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    drop(running);
}

async fn wait_for(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

#[tokio::test]
async fn dropping_the_stream_frees_the_sequence_within_one_step() {
    let step = Duration::from_millis(100);
    let model = MockStepModel::new("mock", &"x".repeat(100)).token_delay(step);
    let live = model.live();
    let decodes = model.decodes();
    let engine = scheduled(model);
    let mut events = engine
        .generate(
            GenerateRequest::chat(vec![Message::new("user", "hi")])
                .with_params(SamplingParams::greedy()),
        )
        .unwrap();
    assert!(matches!(events.recv().await, Some(Event::Started { .. })));
    for _ in 0..2 {
        assert!(matches!(events.recv().await, Some(Event::Text(_))));
    }
    assert_eq!(live.get(), 1);
    let before = decodes.get();
    drop(events);
    wait_for(|| live.get() == 0).await;
    // Counted in steps, not time: at most the step under way when the stream went
    let steps = decodes.get() - before;
    assert!(steps <= 1, "freed after {steps} more steps");
}

#[tokio::test]
async fn a_client_disconnect_frees_the_sequence() {
    let step = Duration::from_millis(50);
    let model = MockStepModel::new("mock", &"x".repeat(1000)).token_delay(step);
    let live = model.live();
    let decodes = model.decodes();
    let app = app_with(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = reqwest::Client::new();
    let mut response = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&chat(json!({"stream": true})))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    // The role chunk, then some content
    let mut received = String::new();
    while received.matches("\"content\":\"x\"").count() < 3 {
        received.push_str(&String::from_utf8_lossy(
            &response.chunk().await.unwrap().unwrap(),
        ));
    }
    assert_eq!(live.get(), 1);
    let before = decodes.get();
    drop(response);
    drop(client);
    wait_for(|| live.get() == 0).await;
    // The server learns of the disconnect when it writes the next token (a write to a closed
    // socket may still succeed once), and the worker drops the sequence at its next step
    let steps = decodes.get() - before;
    assert!(steps <= 3, "freed after {steps} more steps");
}

#[tokio::test]
async fn an_engine_with_its_own_loop_is_cancelled_by_dropping_too() {
    let engine = MockEngine::new("mock", &["a"; 1000]).piece_delay(Duration::from_millis(10));
    let live = engine.live();
    let app = Server::new().llm(engine).router();
    let response = app
        .clone()
        .oneshot(post("/v1/chat/completions", &chat(json!({"stream": true}))))
        .await
        .unwrap();
    let mut body = response.into_body().into_data_stream();
    use futures_util::StreamExt as _;
    let _ = body.next().await;
    assert_eq!(live.get(), 1);
    drop(body);
    wait_for(|| live.get() == 0).await;
    let metrics = send(&app, get("/metrics")).await.json();
    assert_eq!(metrics["llm"][0]["requests_finished"]["cancelled"], 1);
}

#[tokio::test]
async fn concurrent_streams_interleave_token_by_token() {
    let model = MockStepModel::new("mock", &"x".repeat(20)).token_delay(Duration::from_millis(5));
    let engine = scheduled(model);
    let request = || GenerateRequest::text("go").with_params(SamplingParams::greedy());
    let mut first = engine.generate(request()).unwrap();
    // The first stream is under way when the second arrives
    assert!(matches!(first.recv().await, Some(Event::Started { .. })));
    assert!(matches!(first.recv().await, Some(Event::Text(_))));
    let mut second = engine.generate(request()).unwrap();
    let mut second_started_text = false;
    loop {
        tokio::select! {
            event = second.recv() => {
                if matches!(event, Some(Event::Text(_))) {
                    second_started_text = true;
                    break;
                }
            }
            event = first.recv() => {
                if let Some(Event::Finished { reason, .. }) = event {
                    assert_eq!(reason, FinishReason::Stop);
                    break;
                }
            }
        }
    }
    assert!(
        second_started_text,
        "the second stream waited for the first to finish"
    );
}

#[tokio::test]
async fn a_burst_of_requests_starts_together() {
    let model = MockStepModel::new("mock", &"x".repeat(50)).token_delay(Duration::from_millis(100));
    let decodes = model.decodes();
    let engine = scheduled(model);
    let request = || GenerateRequest::text("go").with_params(SamplingParams::greedy());
    // A running stream keeps the worker in a decode step while four more requests arrive
    let mut running = engine.generate(request()).unwrap();
    assert!(matches!(running.recv().await, Some(Event::Started { .. })));
    assert!(matches!(running.recv().await, Some(Event::Text(_))));
    let mut burst: Vec<_> = (0..4)
        .map(|_| engine.generate(request()).unwrap())
        .collect();
    let before = decodes.get();
    for events in &mut burst {
        assert!(matches!(events.recv().await, Some(Event::Started { .. })));
        assert!(matches!(events.recv().await, Some(Event::Text(_))));
    }
    // All four were prefilled after the step under way, not one per step
    let steps = decodes.get() - before;
    assert!(steps <= 1, "the burst took {steps} steps to start");
}

#[tokio::test]
async fn metrics_report_time_to_first_token_and_tokens() {
    let model = MockStepModel::new("mock", "abcdef")
        .prefill_delay(Duration::from_millis(20))
        .token_delay(Duration::from_millis(2));
    let app = app_with(model);
    for _ in 0..3 {
        let answer = send(
            &app,
            post("/v1/chat/completions", &chat(json!({"stream": true}))),
        )
        .await;
        assert_eq!(answer.status, StatusCode::OK);
    }
    let metrics = send(&app, get("/metrics")).await.json();
    let llm = &metrics["llm"][0];
    assert_eq!(llm["name"], "mock");
    assert_eq!(llm["requests_finished"]["stop"], 3);
    assert_eq!(llm["generation_tokens"], 18);
    assert_eq!(llm["requests_running"], 0);
    assert_eq!(llm["requests_waiting"], 0);
    let ttft = llm["time_to_first_token_ms"]["p50"].as_f64().unwrap();
    assert!(ttft >= 15.0, "{ttft}");
    assert!(llm["inter_token_latency_ms"]["p50"].as_f64().is_some());
    assert!(llm["tokens_per_second"].as_f64().unwrap() > 0.0);
    assert!(llm["prefill_ms"]["p50"].as_f64().unwrap() >= 15.0);

    let request = Request::builder()
        .uri("/metrics")
        .header(header::ACCEPT, "text/plain")
        .body(Body::empty())
        .unwrap();
    let text = send(&app, request).await.body;
    for series in [
        "# TYPE furnace_llm_time_to_first_token_seconds histogram",
        "furnace_llm_time_to_first_token_seconds_count{model=\"mock\"} 3",
        "furnace_llm_inter_token_latency_seconds_bucket{model=\"mock\",le=\"+Inf\"}",
        "furnace_llm_queue_seconds_count{model=\"mock\"} 3",
        "furnace_llm_generation_tokens_total{model=\"mock\"} 18",
        "furnace_llm_requests_finished_total{model=\"mock\",reason=\"stop\"} 3",
        "furnace_llm_requests_running{model=\"mock\"} 0",
        "furnace_llm_prefill_seconds_count{model=\"mock\"} 3",
        "furnace_llm_decode_step_seconds_bucket{model=\"mock\",le=\"+Inf\"}",
    ] {
        assert!(text.contains(series), "{series} missing from\n{text}");
    }
}

#[tokio::test]
async fn api_keys_guard_the_llm_routes() {
    let app = Server::new()
        .configure(|c| c.api_keys = vec!["secret".to_string()])
        .llm(scheduled(MockStepModel::new("mock", REPLY)))
        .router();
    let refused = send(&app, post("/v1/chat/completions", &chat(json!({})))).await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);
    let mut request = post("/v1/chat/completions", &chat(json!({})));
    request
        .headers_mut()
        .insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
    assert_eq!(send(&app, request).await.status, StatusCode::OK);
}

#[tokio::test]
async fn an_answer_past_the_request_timeout_is_a_504() {
    let engine = MockEngine::new("mock", &["a"; 3]).piece_delay(Duration::from_secs(60));
    let live = engine.live();
    let app = Server::new()
        .configure(|c| c.request_timeout = Some(Duration::from_millis(50)))
        .llm(engine)
        .router();
    let answer = send(&app, post("/v1/chat/completions", &chat(json!({})))).await;
    let error = assert_openai_error(&answer, StatusCode::GATEWAY_TIMEOUT, "server_error");
    assert_eq!(error["code"], "timeout");
    // Giving up cancelled the generation
    wait_for(|| live.get() == 0).await;
}

#[tokio::test]
async fn a_stream_that_stalls_ends_with_an_error_and_done() {
    let engine = MockEngine::new("mock", &["a"; 3]).piece_delay(Duration::from_secs(60));
    let live = engine.live();
    let app = Server::new()
        .configure(|c| c.request_timeout = Some(Duration::from_millis(50)))
        .llm(engine)
        .router();
    let answer = send(
        &app,
        post("/v1/chat/completions", &chat(json!({"stream": true}))),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    let chunks = answer.chunks();
    // The role, the first piece, then the error in place of the second
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "a");
    let error = &chunks.last().unwrap()["error"];
    assert_eq!(error["code"], "timeout", "{error}");
    wait_for(|| live.get() == 0).await;
}

/// Starts, says one word, then fails
#[derive(Debug)]
struct FailsMidway;

impl LlmEngine for FailsMidway {
    fn name(&self) -> &str {
        "mock"
    }

    fn max_context(&self) -> usize {
        64
    }

    fn generate(
        &self,
        _request: GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Event>, furnace::llm::EngineError> {
        let (events, receiver) = tokio::sync::mpsc::channel(4);
        for event in [
            Event::Started { prompt_tokens: 1 },
            Event::Text("half".to_string()),
            Event::Error(furnace::llm::EngineError::Failed(
                "the model broke".to_string(),
            )),
        ] {
            events.try_send(event).unwrap();
        }
        Ok(receiver)
    }
}

#[tokio::test]
async fn an_error_mid_stream_is_a_payload_then_done() {
    let app = Server::new().llm(FailsMidway).router();
    let answer = send(
        &app,
        post("/v1/chat/completions", &chat(json!({"stream": true}))),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    let chunks = answer.chunks();
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "half");
    let error = &chunks.last().unwrap()["error"];
    assert_eq!(error["message"], "the model broke", "{error}");
    assert_eq!(error["type"], "server_error", "{error}");
    // Without streaming, the same failure is a 500
    let answer = send(&app, post("/v1/chat/completions", &chat(json!({})))).await;
    assert_openai_error(&answer, StatusCode::INTERNAL_SERVER_ERROR, "server_error");
}

/// Refuses only once it has looked at the request: `Error` comes in place of `Started`
#[derive(Debug)]
struct RefusesLate;

impl LlmEngine for RefusesLate {
    fn name(&self) -> &str {
        "mock"
    }

    fn max_context(&self) -> usize {
        8
    }

    fn generate(
        &self,
        _request: GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Event>, furnace::llm::EngineError> {
        let (events, receiver) = tokio::sync::mpsc::channel(1);
        events
            .try_send(Event::Error(furnace::llm::EngineError::TooLong {
                prompt_tokens: 12,
                max_context: 8,
            }))
            .unwrap();
        Ok(receiver)
    }
}

#[tokio::test]
async fn a_refusal_as_the_first_event_is_a_400_even_when_streaming() {
    let app = Server::new().llm(RefusesLate).router();
    for stream in [true, false] {
        let answer = send(
            &app,
            post("/v1/chat/completions", &chat(json!({"stream": stream}))),
        )
        .await;
        let error = assert_openai_error(&answer, StatusCode::BAD_REQUEST, "invalid_request_error");
        assert_eq!(error["code"], "context_length_exceeded", "{error}");
        assert_eq!(error["param"], "messages", "{error}");
    }
}
