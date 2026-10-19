//! The OpenAI-compatible routes: `POST /v1/chat/completions`, `POST /v1/completions` and
//! `GET /v1/models`, with SSE streaming, usage, and errors in OpenAI's shape.
//!
//! Cancellation is drop-based: when a streaming client disconnects, hyper drops the response
//! body, which drops the event receiver, and the engine stops generating for it. A
//! non-streaming request is cancelled the same way when its handler is dropped.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::metrics::{self, EdgeMetrics, Tracked};
use super::{
    EngineError, Event, FinishReason, GenerateRequest, Input, LlmEngine, LlmMetrics, Message,
    SamplingParams,
};

/// OpenAI's limit on stop strings
const MAX_STOP_STRINGS: usize = 4;

/// The LLMs of a server
pub(crate) struct LlmState {
    engines: Vec<Served>,
    /// The server's other models, listed by `/v1/models` too
    other_models: Vec<String>,
    /// The server's concurrency limit (`max_concurrent_requests`), shared with every route
    slots: Arc<Semaphore>,
    /// When the server started, as `created` in `/v1/models`
    created: u64,
}

struct Served {
    engine: Arc<dyn LlmEngine>,
    metrics: Arc<EdgeMetrics>,
}

impl LlmState {
    pub(crate) fn new(
        engines: Vec<Arc<dyn LlmEngine>>,
        other_models: Vec<String>,
        slots: Arc<Semaphore>,
    ) -> Self {
        for (i, engine) in engines.iter().enumerate() {
            let name = engine.name();
            assert!(
                !other_models.iter().any(|m| m == name)
                    && !engines[..i].iter().any(|e| e.name() == name),
                "two models are named '{name}'"
            );
        }
        Self {
            engines: engines
                .into_iter()
                .map(|engine| Served {
                    engine,
                    metrics: Arc::default(),
                })
                .collect(),
            other_models,
            slots,
            created: unix_seconds(),
        }
    }

    pub(crate) fn names(&self) -> Vec<&str> {
        self.engines.iter().map(|s| s.engine.name()).collect()
    }

    pub(crate) fn render_prometheus(&self, out: &mut String) {
        let engines: Vec<(&str, &EdgeMetrics, &dyn LlmEngine)> = self
            .engines
            .iter()
            .map(|s| (s.engine.name(), &*s.metrics, &*s.engine))
            .collect();
        metrics::render(out, &engines);
    }

    pub(crate) fn json_metrics(&self) -> Vec<LlmMetrics> {
        self.engines
            .iter()
            .map(|s| metrics::json(s.engine.name(), &s.metrics, &*s.engine))
            .collect()
    }

    /// The engine a request names, or the first one when it names none
    fn engine(&self, model: Option<&str>) -> Result<&Served, Box<Response>> {
        let Some(model) = model else {
            return Ok(&self.engines[0]);
        };
        if let Some(served) = self.engines.iter().find(|s| s.engine.name() == model) {
            return Ok(served);
        }
        let response = if self.other_models.iter().any(|m| m == model) {
            error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "The model '{model}' is not a language model; send it tensors on /v2/models/{model}/infer"
                ),
                "invalid_request_error",
                Some("model"),
                Some("model_not_supported"),
            )
        } else {
            error_response(
                StatusCode::NOT_FOUND,
                &format!(
                    "The model '{model}' does not exist; this server serves {}",
                    self.names().join(", ")
                ),
                "invalid_request_error",
                Some("model"),
                Some("model_not_found"),
            )
        };
        Err(Box::new(response))
    }
}

/// The routes, for the server's router (any state: they carry their own)
pub(crate) fn routes<S: Clone + Send + Sync + 'static>(state: Arc<LlmState>) -> Router<S> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
}

/// `stop`: one string or several
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// What chat and completion requests share. Fields OpenAI defines and furnace does not serve
/// are rejected; unknown fields are ignored, as other servers do, so SDKs that send extras work.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Options {
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    /// Not OpenAI, but common (vLLM, llama.cpp); 0 or -1 turn it off
    top_k: Option<i64>,
    seed: Option<i64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stop: Option<OneOrMany>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    n: Option<u64>,
    /// `true` for chat, a number for completions
    logprobs: Option<serde_json::Value>,
    tools: Option<Vec<serde_json::Value>>,
    response_format: Option<serde_json::Value>,
    echo: Option<bool>,
    best_of: Option<u64>,
    suffix: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(flatten)]
    options: Options,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    /// `null` for an assistant message with tool calls
    #[serde(default)]
    content: Option<Content>,
}

/// A message's content: text, or parts of which only text is served
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Prompt,
    #[serde(flatten)]
    options: Options,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Many(Vec<serde_json::Value>),
}

/// How the response goes out
struct Delivery {
    stream: bool,
    include_usage: bool,
}

fn invalid(message: &str, param: &str) -> Box<Response> {
    Box::new(error_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        Some(param),
        None,
    ))
}

impl Options {
    /// The sampling parameters and stop strings, or a 400 for what furnace does not serve
    fn check(self) -> Result<(SamplingParams, Vec<String>, Delivery), Box<Response>> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(invalid("Only n = 1 is supported", "n"));
        }
        if self.best_of.is_some_and(|n| n != 1) {
            return Err(invalid("Only best_of = 1 is supported", "best_of"));
        }
        let logprobs = match &self.logprobs {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(on)) => *on,
            Some(serde_json::Value::Number(n)) => n.as_u64() != Some(0),
            Some(_) => true,
        };
        if logprobs {
            return Err(invalid("logprobs are not supported", "logprobs"));
        }
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
            return Err(invalid("Tool calling is not supported", "tools"));
        }
        if let Some(format) = &self.response_format {
            let kind = format.get("type").and_then(|t| t.as_str());
            if kind.is_some_and(|k| k != "text") {
                return Err(invalid(
                    "Only the text response format is supported",
                    "response_format",
                ));
            }
        }
        if self.echo == Some(true) {
            return Err(invalid("echo is not supported", "echo"));
        }
        if self.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
            return Err(invalid("suffix is not supported", "suffix"));
        }
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(invalid(
                "temperature must be between 0 and 2",
                "temperature",
            ));
        }
        if let Some(p) = self.top_p
            && !(p > 0.0 && p <= 1.0)
        {
            return Err(invalid("top_p must be in (0, 1]", "top_p"));
        }
        for (penalty, name) in [
            (self.presence_penalty, "presence_penalty"),
            (self.frequency_penalty, "frequency_penalty"),
        ] {
            if penalty.is_some_and(|p| !(-2.0..=2.0).contains(&p)) {
                return Err(invalid(&format!("{name} must be between -2 and 2"), name));
            }
        }
        let max_tokens = self.max_completion_tokens.or(self.max_tokens);
        if max_tokens == Some(0) {
            return Err(invalid("max_tokens must be at least 1", "max_tokens"));
        }
        let stop = match self.stop {
            None => Vec::new(),
            Some(OneOrMany::One(stop)) => vec![stop],
            Some(OneOrMany::Many(stops)) => stops,
        };
        if stop.len() > MAX_STOP_STRINGS {
            return Err(invalid(
                &format!("At most {MAX_STOP_STRINGS} stop strings are supported"),
                "stop",
            ));
        }
        let params = SamplingParams {
            max_tokens: max_tokens.map(|m| usize::try_from(m).unwrap_or(usize::MAX)),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self
                .top_k
                .filter(|&k| k > 0)
                .map(|k| usize::try_from(k).unwrap_or(usize::MAX)),
            seed: self.seed.map(|s| s as u64),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
        };
        let stream = self.stream.unwrap_or(false);
        let delivery = Delivery {
            stream,
            include_usage: stream && self.stream_options.is_some_and(|o| o.include_usage),
        };
        Ok((params, stop, delivery))
    }
}

/// Parse a JSON body, with OpenAI's error for what doesn't parse
fn parse<T: serde::de::DeserializeOwned>(
    body: Result<Bytes, BytesRejection>,
) -> Result<T, Box<Response>> {
    let body = body.map_err(|rejection| {
        Box::new(error_response(
            rejection.status(),
            &rejection.body_text(),
            "invalid_request_error",
            None,
            None,
        ))
    })?;
    serde_json::from_slice(&body)
        .map_err(|e| invalid(&format!("Invalid request body: {e}"), "body"))
}

/// The two kinds of completion, which differ only in their JSON
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Chat,
    Text,
}

/// `POST /v1/chat/completions`
async fn chat_completions(
    State(state): State<Arc<LlmState>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let received = Instant::now();
    let request: ChatRequest = match parse(body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages {
        let content = match message.content {
            None => String::new(),
            Some(Content::Text(text)) => text,
            Some(Content::Parts(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match (part.kind.as_str(), part.text) {
                        ("text", Some(part)) => text.push_str(&part),
                        (kind, _) => {
                            return *invalid(
                                &format!("Content of type '{kind}' is not supported, only text"),
                                "messages",
                            );
                        }
                    }
                }
                text
            }
        };
        messages.push(Message {
            role: message.role,
            content,
        });
    }
    if messages.is_empty() {
        return *invalid("messages must not be empty", "messages");
    }
    generate(
        &state,
        Kind::Chat,
        request.model.as_deref(),
        Input::Chat(messages),
        request.options,
        received,
    )
    .await
}

/// `POST /v1/completions`
async fn completions(
    State(state): State<Arc<LlmState>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let received = Instant::now();
    let request: CompletionRequest = match parse(body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let prompt = match request.prompt {
        Prompt::Text(text) => text,
        Prompt::Many(mut prompts) => match (prompts.len(), prompts.pop()) {
            (1, Some(serde_json::Value::String(text))) => text,
            _ => {
                return *invalid(
                    "Only one prompt, as a string, is supported per request",
                    "prompt",
                );
            }
        },
    };
    generate(
        &state,
        Kind::Text,
        request.model.as_deref(),
        Input::Text(prompt),
        request.options,
        received,
    )
    .await
}

async fn generate(
    state: &LlmState,
    kind: Kind,
    model: Option<&str>,
    input: Input,
    options: Options,
    received: Instant,
) -> Response {
    let served = match state.engine(model) {
        Ok(served) => served,
        Err(response) => return *response,
    };
    let (params, stop, delivery) = match options.check() {
        Ok(checked) => checked,
        Err(response) => return *response,
    };
    // Backpressure, as for every other route: refuse at once when all slots are taken
    let Ok(permit) = state.slots.clone().try_acquire_owned() else {
        served.metrics.reject();
        return busy_response("Too many concurrent requests, retry later");
    };
    let request = GenerateRequest {
        input,
        params,
        stop,
    };
    let events = match served.engine.generate(request) {
        Ok(events) => events,
        Err(e) => {
            if matches!(e, EngineError::Busy(_)) {
                served.metrics.reject();
            }
            return engine_error_response(&e, kind);
        }
    };
    let tracked = Tracked::new(events, served.metrics.clone(), received);
    let reply = Reply {
        kind,
        id: completion_id(kind),
        created: unix_seconds(),
        model: served.engine.name().to_string(),
    };
    if delivery.stream {
        stream(reply, tracked, delivery.include_usage, permit)
    } else {
        collect(reply, tracked, permit).await
    }
}

/// What every chunk of a response repeats
struct Reply {
    kind: Kind,
    id: String,
    created: u64,
    model: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl Usage {
    fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Serialize)]
struct Chunk<'a, C> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: &'a [C],
    /// Absent without `include_usage`; with it, `null` but in the last chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Option<Usage>>,
}

#[derive(Serialize)]
struct ChatChoice<'a> {
    index: u32,
    delta: Delta<'a>,
    logprobs: Option<()>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

#[derive(Serialize)]
struct TextChoice<'a> {
    index: u32,
    text: &'a str,
    logprobs: Option<()>,
    finish_reason: Option<&'static str>,
}

impl Reply {
    fn object(&self, stream: bool) -> &'static str {
        match (self.kind, stream) {
            (Kind::Chat, true) => "chat.completion.chunk",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Text, _) => "text_completion",
        }
    }

    /// One `data:` payload: `role` and `content` for chat, `content` alone as text for
    /// completions; no choice at all with `usage`
    fn chunk(
        &self,
        role: Option<&'static str>,
        content: Option<&str>,
        finish_reason: Option<FinishReason>,
        usage: Option<Option<Usage>>,
        last: bool,
    ) -> String {
        let finish_reason = finish_reason.map(FinishReason::as_str);
        let serialized = match self.kind {
            Kind::Chat => {
                let choice = [ChatChoice {
                    index: 0,
                    delta: Delta { role, content },
                    logprobs: None,
                    finish_reason,
                }];
                serde_json::to_string(&Chunk {
                    id: &self.id,
                    object: self.object(true),
                    created: self.created,
                    model: &self.model,
                    choices: if last { &[] } else { &choice },
                    usage,
                })
            }
            Kind::Text => {
                let choice = [TextChoice {
                    index: 0,
                    text: content.unwrap_or(""),
                    logprobs: None,
                    finish_reason,
                }];
                serde_json::to_string(&Chunk {
                    id: &self.id,
                    object: self.object(true),
                    created: self.created,
                    model: &self.model,
                    choices: if last { &[] } else { &choice },
                    usage,
                })
            }
        };
        serialized.expect("chunks serialize")
    }
}

/// A streaming response in progress
struct Streamed {
    reply: Reply,
    tracked: Tracked,
    include_usage: bool,
    /// Payloads ready to go out
    pending: VecDeque<String>,
    done: bool,
    completion_tokens: usize,
    _permit: OwnedSemaphorePermit,
}

impl Streamed {
    fn usage_field(&self) -> Option<Option<Usage>> {
        self.include_usage.then_some(None)
    }

    /// Turn an event into payloads
    fn on(&mut self, event: Option<Event>) {
        match event {
            Some(Event::Started { .. }) => {}
            Some(Event::Text(text)) => {
                let chunk = self
                    .reply
                    .chunk(None, Some(&text), None, self.usage_field(), false);
                self.pending.push_back(chunk);
            }
            Some(Event::Finished {
                reason,
                completion_tokens,
            }) => {
                self.completion_tokens = completion_tokens;
                let content = (self.reply.kind == Kind::Text).then_some("");
                let chunk =
                    self.reply
                        .chunk(None, content, Some(reason), self.usage_field(), false);
                self.pending.push_back(chunk);
                if self.include_usage {
                    let usage = Usage::new(self.tracked.prompt_tokens, completion_tokens);
                    let chunk = self.reply.chunk(None, None, None, Some(Some(usage)), true);
                    self.pending.push_back(chunk);
                }
                self.end();
            }
            Some(Event::Error(e)) => {
                self.pending.push_back(error_body(&e).to_string());
                self.end();
            }
            None => {
                let e = EngineError::Failed("the engine ended the request early".to_string());
                self.pending.push_back(error_body(&e).to_string());
                self.end();
            }
        }
    }

    fn end(&mut self) {
        self.pending.push_back("[DONE]".to_string());
        self.done = true;
    }
}

/// Server-sent events: chunks as the engine produces them, then `[DONE]`. Even an error after
/// the stream started ends with `[DONE]`, after an error payload, as vLLM does.
fn stream(
    reply: Reply,
    tracked: Tracked,
    include_usage: bool,
    permit: OwnedSemaphorePermit,
) -> Response {
    let mut streamed = Streamed {
        reply,
        tracked,
        include_usage,
        pending: VecDeque::new(),
        done: false,
        completion_tokens: 0,
        _permit: permit,
    };
    if streamed.reply.kind == Kind::Chat {
        let chunk = streamed.reply.chunk(
            Some("assistant"),
            Some(""),
            None,
            streamed.usage_field(),
            false,
        );
        streamed.pending.push_back(chunk);
    }
    let events = futures_util::stream::unfold(streamed, |mut streamed| async move {
        loop {
            if let Some(data) = streamed.pending.pop_front() {
                return Some((
                    Ok::<_, Infallible>(SseEvent::default().data(data)),
                    streamed,
                ));
            }
            if streamed.done {
                return None;
            }
            let event = streamed.tracked.next().await;
            streamed.on(event);
        }
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// A whole response at once
async fn collect(reply: Reply, mut tracked: Tracked, _permit: OwnedSemaphorePermit) -> Response {
    let mut text = String::new();
    let (reason, completion_tokens) = loop {
        match tracked.next().await {
            Some(Event::Started { .. }) => {}
            Some(Event::Text(piece)) => text.push_str(&piece),
            Some(Event::Finished {
                reason,
                completion_tokens,
            }) => break (reason, completion_tokens),
            Some(Event::Error(e)) => return engine_error_response(&e, reply.kind),
            None => {
                let e = EngineError::Failed("the engine ended the request early".to_string());
                return engine_error_response(&e, reply.kind);
            }
        }
    };
    let choice = match reply.kind {
        Kind::Chat => serde_json::json!({
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": null,
            "finish_reason": reason.as_str(),
        }),
        Kind::Text => serde_json::json!({
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": reason.as_str(),
        }),
    };
    Json(serde_json::json!({
        "id": reply.id,
        "object": reply.object(false),
        "created": reply.created,
        "model": reply.model,
        "choices": [choice],
        "usage": Usage::new(tracked.prompt_tokens, completion_tokens),
    }))
    .into_response()
}

/// `GET /v1/models`: the LLMs, then the server's other models
async fn list_models(State(state): State<Arc<LlmState>>) -> Json<serde_json::Value> {
    let llms = state.engines.iter().map(|s| {
        serde_json::json!({
            "id": s.engine.name(),
            "object": "model",
            "created": state.created,
            "owned_by": "furnace",
            "max_model_len": s.engine.max_context(),
        })
    });
    let others = state.other_models.iter().map(|name| {
        serde_json::json!({
            "id": name,
            "object": "model",
            "created": state.created,
            "owned_by": "furnace",
        })
    });
    Json(serde_json::json!({
        "object": "list",
        "data": llms.chain(others).collect::<Vec<_>>(),
    }))
}

/// OpenAI's error body: `{"error": {"message", "type", "param", "code"}}`
fn error_value(
    message: &str,
    kind: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "error": {"message": message, "type": kind, "param": param, "code": code}
    })
}

fn error_response(
    status: StatusCode,
    message: &str,
    kind: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    (status, Json(error_value(message, kind, param, code))).into_response()
}

fn busy_response(message: &str) -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        message,
        "server_error",
        None,
        Some("server_busy"),
    )
}

/// The status, type and code of an engine's error
fn classify(e: &EngineError) -> (StatusCode, &'static str, Option<&'static str>) {
    match e {
        EngineError::TooLong { .. } => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded"),
        ),
        EngineError::Busy(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            Some("server_busy"),
        ),
        EngineError::Invalid(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
        EngineError::Failed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None),
    }
}

fn error_body(e: &EngineError) -> serde_json::Value {
    let (_, kind, code) = classify(e);
    error_value(&e.to_string(), kind, None, code)
}

fn engine_error_response(e: &EngineError, kind: Kind) -> Response {
    let (status, error_kind, code) = classify(e);
    let param = matches!(e, EngineError::TooLong { .. }).then_some(match kind {
        Kind::Chat => "messages",
        Kind::Text => "prompt",
    });
    error_response(status, &e.to_string(), error_kind, param, code)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `chatcmpl-…` or `cmpl-…`: a random per-process value plus a counter, unique without asking
/// the OS for randomness per request
fn completion_id(kind: Kind) -> String {
    static SEED: std::sync::LazyLock<u64> =
        std::sync::LazyLock::new(|| uuid::Uuid::new_v4().as_u64_pair().0);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let prefix = match kind {
        Kind::Chat => "chatcmpl",
        Kind::Text => "cmpl",
    };
    format!(
        "{prefix}-{:016x}{:08x}",
        *SEED,
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
