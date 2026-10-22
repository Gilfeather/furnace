//! Overload and failure on the server: the concurrency limit refuses at once, a full batching
//! queue answers 503, and a brain that hangs, is slow or panics ends its requests while the
//! rest of the server goes on answering.
//!
//! Nothing here waits on the clock: blocked work is released by the test, and the one timeout
//! that has to pass (a request waiting for a hung brain) passes on Tokio's paused clock.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode};
use furnace::batching::BatchConfig;
use furnace::nerve::effector::Respond;
use furnace::nerve::receptor::Http;
use furnace::nerve::{Fabric, FnNeuron, NeuronKind, Summation};
use furnace::{HostTensor, IoSpec, Model, Server, ServerConfig, TensorModel, Values};
use tokio::sync::mpsc;
use tower::ServiceExt;

const WIDTH: usize = 4;

/// A gate the test opens: whatever waits at it blocks until then
#[derive(Debug, Default)]
struct Gate {
    open: Mutex<bool>,
    opened: Condvar,
}

impl Gate {
    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }

    fn pass(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
    }
}

/// Echoes its input. A forward pass of anything but zeros (the warm-up) tells the test it
/// started, then waits at the gate.
#[derive(Debug)]
struct Gated {
    gate: Arc<Gate>,
    started: mpsc::UnboundedSender<()>,
}

impl TensorModel for Gated {
    fn name(&self) -> &str {
        "gated"
    }
    fn backend(&self) -> &str {
        "cpu"
    }
    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &[WIDTH])]
    }
    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &[WIDTH])]
    }
    fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
        let input = inputs.into_iter().next().expect("one input");
        let Values::F32(values) = &input.values else {
            panic!("f32 input");
        };
        if values.iter().any(|&v| v != 0.0) {
            let _ = self.started.send(());
            self.gate.pass();
        }
        Ok(vec![input])
    }
}

struct Setup {
    app: Router,
    gate: Arc<Gate>,
    started: mpsc::UnboundedReceiver<()>,
}

fn gated(config: ServerConfig) -> Setup {
    let gate = Arc::new(Gate::default());
    let (started, started_rx) = mpsc::unbounded_channel();
    let model = Model::new(
        Box::new(Gated {
            gate: gate.clone(),
            started,
        }),
        "test",
        0,
    );
    let app = Server::with_config(config).model(model).router().unwrap();
    Setup {
        app,
        gate,
        started: started_rx,
    }
}

impl Setup {
    /// Wait until a forward pass has started and is held at the gate; fails the test after
    /// 10 seconds rather than hanging
    async fn started(&mut self) {
        tokio::time::timeout(Duration::from_secs(10), self.started.recv())
            .await
            .expect("a forward pass started within 10 s")
            .expect("the model is alive");
    }
}

fn predict(body: &str) -> Request<Body> {
    Request::post("/predict")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn ones() -> Request<Body> {
    predict(&serde_json::json!({ "input": vec![1.0; WIDTH] }).to_string())
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Bytes) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, body)
}

/// `send`, but failing the test after 10 seconds: for a request that must be answered while
/// others are held, rather than queue behind them
async fn send_now(app: &Router, request: Request<Body>) -> (StatusCode, Bytes) {
    tokio::time::timeout(Duration::from_secs(10), send(app, request))
        .await
        .expect("answered while the other requests are held")
}

fn spawn(app: &Router, request: Request<Body>) -> tokio::task::JoinHandle<(StatusCode, Bytes)> {
    let app = app.clone();
    tokio::spawn(async move { send(&app, request).await })
}

fn code(body: &Bytes) -> String {
    let error: serde_json::Value = serde_json::from_slice(body).unwrap();
    error["code"].as_str().unwrap_or_default().to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn past_the_concurrency_limit_requests_are_refused_at_once() {
    let mut s = gated(ServerConfig {
        max_concurrent_requests: 2,
        ..ServerConfig::default()
    });
    let held: Vec<_> = (0..2).map(|_| spawn(&s.app, ones())).collect();
    s.started().await;
    s.started().await;

    // Both slots are taken and their passes are held: the next requests are refused at once,
    // while the first two are still running
    for _ in 0..5 {
        let (status, body) = send_now(&s.app, ones()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(code(&body), "SERVER_BUSY");
    }
    // Routes that run no model go on answering
    let health = Request::get("/healthz").body(Body::empty()).unwrap();
    assert_eq!(send_now(&s.app, health).await.0, StatusCode::OK);

    s.gate.open();
    for request in held {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
    // The slots are free again
    assert_eq!(send(&s.app, ones()).await.0, StatusCode::OK);
    let metrics = Request::get("/metrics").body(Body::empty()).unwrap();
    let (_, body) = send(&s.app, metrics).await;
    let metrics: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(metrics["rejected_busy"], 5, "{metrics}");
}

/// The batching queue holds as many requests as may run at once, so it fills only with them;
/// a request under the limit is queued even after others gave up while the worker was busy
#[tokio::test(flavor = "multi_thread")]
async fn the_batching_queue_refuses_only_past_the_limit() {
    let mut s = gated(ServerConfig {
        max_concurrent_requests: 2,
        batching: BatchConfig {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(1),
            workers: 1,
        },
        ..ServerConfig::default()
    });
    let first = spawn(&s.app, ones());
    s.started().await;
    // Requests that give up before they are queued: each was announced to the batching worker
    // and withdrawn
    for _ in 0..4 {
        assert_eq!(
            send_now(&s.app, predict("{")).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    // One more fits under the limit, and waits for the worker
    let second = spawn(&s.app, ones());
    until_waiting(&s.app, 1).await;
    // Past the limit: refused at once
    let (status, body) = send_now(&s.app, ones()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(code(&body), "SERVER_BUSY");

    s.gate.open();
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    assert_eq!(second.await.unwrap().0, StatusCode::OK);
}

/// Wait until `waiting` requests are queued for a batching worker, as `/metrics` reports;
/// fails the test after 10 seconds
async fn until_waiting(app: &Router, waiting: u64) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let metrics = Request::get("/metrics").body(Body::empty()).unwrap();
        let (_, body) = send(app, metrics).await;
        let metrics: serde_json::Value = serde_json::from_slice(&body).unwrap();
        if metrics["nerve"]["waiting"] == waiting {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "not {waiting} waiting after 10 s: {}",
            metrics["nerve"]
        );
        tokio::task::yield_now().await;
    }
}

/// POST /ask goes to a brain behind a deliberative pathway, which `respond_within` waits for;
/// POST /ping is a reflex that answers at once. The brain waits at the gate on `slow`, panics
/// on `boom`, and answers anything else.
fn nerve_app(wait: Duration) -> (Router, Arc<Gate>, mpsc::UnboundedReceiver<()>) {
    let gate = Arc::new(Gate::default());
    let (thinking, thinking_rx) = mpsc::unbounded_channel();
    let mut b = Fabric::builder();
    let ask = b.receptor(Http::new("ask", "/ask").text().respond_within(wait));
    let ping = b.receptor(Http::new("ping", "/ping"));
    let brain = {
        let gate = gate.clone();
        b.neuron(
            FnNeuron::reflex("brain", move |signal, _| {
                let text = signal.payload.as_text().unwrap_or_default().to_string();
                match text.as_str() {
                    "boom" => panic!("the brain panicked"),
                    "slow" => {
                        let _ = thinking.send(());
                        gate.pass();
                    }
                    _ => {}
                }
                signal.command(
                    "respond",
                    serde_json::json!({ "status": 200, "body": format!("thought: {text}") }),
                )
            })
            .with_kind(NeuronKind::Brain),
        )
    };
    let reflex = b.neuron(FnNeuron::reflex("reflex", |signal, _| {
        signal.command(
            "respond",
            serde_json::json!({ "status": 200, "body": "pong" }),
        )
    }));
    b.effector(Respond::new("respond"));
    b.deliberative(ask, brain)
        .summation(Summation::none().workers(1));
    b.reflex(ping, reflex);
    let app = Server::new().nerve(b).router().unwrap();
    (app, gate, thinking_rx)
}

fn post(path: &str, body: &str) -> Request<Body> {
    Request::post(path)
        .header("content-type", "text/plain")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// A brain that never answers: the request waiting for it ends when its wait is over, here on
/// the paused clock, and the reflex goes on answering meanwhile
#[tokio::test(start_paused = true)]
async fn a_hung_brain_lets_its_requests_go_and_the_reflex_answers() {
    let (app, gate, mut thinking) = nerve_app(Duration::from_secs(30));
    let started = tokio::time::Instant::now();
    let (status, _) = send(&app, post("/ask", "slow")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert!(started.elapsed() >= Duration::from_secs(30));
    thinking.recv().await.expect("the brain is thinking");

    // Requests queued behind the hung one end the same way; nothing else is held up
    for _ in 0..3 {
        assert_eq!(
            send(&app, post("/ask", "next")).await.0,
            StatusCode::ACCEPTED
        );
        let (status, body) = send(&app, post("/ping", "")).await;
        assert_eq!((status, &body[..]), (StatusCode::OK, &b"pong"[..]));
    }
    let nerve = Request::get("/nerve").body(Body::empty()).unwrap();
    assert_eq!(send(&app, nerve).await.0, StatusCode::OK);
    gate.open();
}

/// A slow brain is waited for while the reflex answers; a panicking one loses only its own
/// request, and the brain answers the next
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_or_panicking_brain_fails_alone() {
    let (app, gate, mut thinking) = nerve_app(Duration::from_secs(60));
    let slow = spawn(&app, post("/ask", "slow"));
    tokio::time::timeout(Duration::from_secs(10), thinking.recv())
        .await
        .expect("the brain started within 10 s");
    let (status, body) = send_now(&app, post("/ping", "")).await;
    assert_eq!((status, &body[..]), (StatusCode::OK, &b"pong"[..]));
    gate.open();
    let (status, body) = slow.await.unwrap();
    assert_eq!((status, &body[..]), (StatusCode::OK, &b"thought: slow"[..]));

    // The panic drops the request's signal, so its request is not answered by the brain: it
    // ends at once rather than at the end of its wait
    let (status, _) = send_now(&app, post("/ask", "boom")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let (status, body) = send_now(&app, post("/ask", "again")).await;
    assert_eq!(
        (status, &body[..]),
        (StatusCode::OK, &b"thought: again"[..])
    );
}
