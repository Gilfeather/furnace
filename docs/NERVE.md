# The nervous system

Inference getting faster is worth nothing if the layer that moves signals to and from the
models is the bottleneck. With the `nerve` feature, Furnace is that layer: not a server that
loads models and answers requests, but the nervous system between sensors, small fast models,
the large slow model (the brain, an LLM) and actuators. Text, images, tensors and model outputs
are one kind of signal. Most signals never reach the brain. The receptive side and the motor
side are separate systems. The mechanisms are a neuron's.

```toml
[dependencies]
furnace-server = { version = "0.9", default-features = false, features = ["nerve"] }
```

Everything is pure Rust and adds no dependency: `nerve` uses what the crate already has, plus
a 300-line HTTP/1.1 client over Tokio sockets for the outbound side (`furnace::nerve::wire`).

## Why a fly dodges a hand

A fly takes off within milliseconds of its escape neuron firing, far inside a human reaction
time: about 1 ms separates the giant fibre's spike from the jump muscle's activation (the
giant fibre itself fires once the looming object has grown to a threshold size). Nothing in
the explanation is "the fly's brain is fast":

- The retina does not recognise a hand; a few neurons (LPLC2) detect *expansion* and the
  image goes no further. **Receptors emit features, not data.**
- One thick descending fibre per side runs from those neurons to the jump motor neuron over
  mixed electrical-chemical synapses; the fibre integrates size and velocity against a
  threshold and then fires, with no deliberation behind it. **Reflex pathways have no queue
  and no scheduler.**
- A fly has a short, rough escape and a longer, better coordinated one; when the giant fibre
  spikes relative to the parallel circuits decides which happens, and faster looms bias it
  to the short one. **Timing selects the pathway, not a classifier.**
- Before a takeoff, a fly repositions its legs so that the jump goes away from the threat, a
  preparation that depends on its posture and the stimulus. **Slow signals prepare.**
- Repeated looms produce weaker and rarer escapes, and the response recovers with time;
  in the fly this habituation sits upstream of the reflex, in the afferents to the giant
  fibre. **Habituation lives before the pathway**: in Furnace, in the receptor, as a design
  choice.
- In flight, the halteres drive an equilibrium reflex that corrects posture without the brain.
  **Effectors have their own local loops.**
- The fly's eye resolves flicker at about 100 Hz, against about 50 to 90 Hz for a human.
  **Receptors sample faster than the world changes.**

Every inference server today is the opposite: every input goes to the brain, the brain decides
everything, and the body waits for the brain's reply.

## Concepts

```text
world ──▶ RECEPTORS ──▶ PATHWAYS ──▶ EFFECTORS ──▶ world
          (afferent)    ├─ reflex: a direct call, microseconds
                        └─ deliberative: a queue to the BRAIN (an LLM, a large model) → commands
```

**Signal.** One type for everything that moves (`furnace::nerve::Signal`): a `Feature` (a
64-bit sketch plus a kind, what habituation and thresholds compare), an `Urgency` (a deadline
and the rate the receptor saw), an `intensity` in 0..1, a lazily shared `Payload` (bytes, text,
JSON, tensors or a `Command`), a `Trace` of the neurons it passed, and, for a signal that
started as a request, the way back to it. Signals are owned Rust values: no serialization and
no copy between neurons.

**Neuron.** Anything that takes a signal and may fire: `fn stimulate(&self, Signal, &Cx) ->
Firing`, where `Firing` is `Quiet`, `Fire(Vec<Signal>)` or `Compute(closure)` for blocking
work. Four kinds: receptors (world → signal), reflexes (feature → command, in process), brains
(the slow model; its output is commands) and effectors (command → world).

**Mechanisms.** Every neuron is added with `Mechanisms`, each optional and each a few words of
state, run by the fabric before `stimulate`:

- *Threshold*: fires only if `intensity >= threshold`.
- *Habituation*: a bounded, set-associative table keyed by feature. Each repeat multiplies the
  intensity by `1 - a` and raises `a` toward 1 by `gain`; `a` decays as `exp(-dt / tau)`. A
  repeated stimulus dies here, below the threshold, and recovers with time. This is caching as
  a property of every neuron, not a layer.
- *Refractory period*: after firing, the neuron ignores signals for a while. Backpressure
  without counters.

**Pathway.** A directed connection with a **measured** rolling p50/p99 latency, never
configured. Reflex pathways are direct calls on the firing thread: `Receptor::stimulate` →
`Reflex::stimulate` → `Effector::stimulate`, no channel between them. Deliberative pathways
queue to a worker thread that sums the signals of a short window into one firing (temporal
summation); the window is learned from the pathway's latency (an eighth of its p50, between
200 µs and 20 ms).

**Selection.** From a neuron that fired, the candidate pathways are those whose p99 fits in
the signal's remaining deadline; among them the heaviest weight wins; with no deadline every
pathway is eligible. Give the brain's pathway the heavier weight and the reflex arc is what
happens when there is no time for the brain, like the fly's short-mode escape. A command
(`Payload::Command`) names its effector and goes there.

**Preparation.** A signal with time to spare that takes a deliberative pathway calls
`prepare()` on the neurons it will probably reach (the brain, and the effectors after it), at
most once a second each: a webhook effector opens its connection in advance.

**Afferent and efferent.** Receptors sample on their own threads or tasks and never wait on
the motor side. Effectors act on their own, on the fabric's efferent runtime, with their own
loops: retry with backoff, a circuit breaker, a rate limit, a refractory period. The brain is
never in a retry loop. Acting on the world is the point: an HTTP response is one effector
among the webhook, the local command, the journal file, the in-process channel and the
Server-Sent-Events push.

## The API

```rust
use std::time::Duration;
use furnace::nerve::brain::Llm;
use furnace::nerve::effector::{Effector, FileAppend, Respond, Webhook};
use furnace::nerve::receptor::Http;
use furnace::nerve::{Fabric, Firing, FnNeuron, Habituation, Mechanisms, Summation, Urgency};
use furnace::Server;

let mut b = Fabric::builder();

// A receptor: POST /events. The feature is a 64-bit sketch of the body; repeats habituate
// here and never reach anything.
let events = b.receptor(
    Http::new("events", "/events").text(),
    Mechanisms::NONE
        .threshold(0.6)
        .habituation(Habituation::new(Duration::from_secs(30))),
);

// A reflex: in process, no blocking. Alerts become commands with a 2 ms deadline, which the
// brain's pathway (hundreds of milliseconds) can never meet; the rest pass on.
let triage = b.neuron(
    FnNeuron::reflex("triage", |signal, _cx| {
        if signal.payload.as_text().is_some_and(|t| t.contains("ALERT")) {
            Firing::Fire(vec![
                signal
                    .command("alarm", serde_json::json!({"text": signal.payload.as_text()}))
                    .with_urgency(Urgency::within(Duration::from_millis(2))),
                signal.command("respond", serde_json::json!({"status": 200, "body": "alarm"})),
            ])
        } else {
            Firing::one(signal)
        }
    }),
    Mechanisms::NONE,
);

// The brain: any LlmEngine (a Scheduled Burn model, the candle adapter, a mock). Its answer
// is parsed as commands: {"effector": "...", "args": {...}}, arrays of them, or tool calls.
let brain = b.neuron(Llm::new("brain", engine), Mechanisms::NONE);

// Effectors, by name, each with its local loop
let alarm = b.neuron(
    Effector::new(Webhook::new("alarm", "http://127.0.0.1:9000/alarm"))
        .refractory(Duration::from_millis(100)),
    Mechanisms::NONE,
);
b.neuron(Effector::new(Webhook::new("webhook", "http://127.0.0.1:9000/hook")), Mechanisms::NONE);
b.neuron(Effector::new(FileAppend::new("log", "journal.log")), Mechanisms::NONE);
b.neuron(Effector::new(Respond::new("respond")), Mechanisms::NONE);

// Pathways: the brain's weighs more, so it wins whenever the deadline allows it
b.reflex(events, triage, 1.0);
b.reflex(triage, alarm, 1.0);
b.deliberative(triage, brain, 2.0, Summation::default());

Server::new().port(3000).nerve(b.build()).serve().await?;
```

`Server::nerve` mounts the HTTP receptors as routes, starts the other receptors, serves
`GET /nerve` (neurons, mechanisms, pathways and their measured latencies) and
`GET /nerve/effectors/{name}/events` (an SSE effector's commands), and adds the fabric's
metrics to `/metrics`, next to whatever else the server serves. A request to a receptor is
answered by a `Respond` effector on its chain; otherwise it gets `202 Accepted` when its signal
was queued to the brain, or `204 No Content`, with what happened in the `x-furnace-nerve`
header (`quiet:habituation`, `acted:1`, `queued`). A fabric can also be driven without a
server: `fabric.stimulate(receptor, signal)` runs the whole synchronous chain before it
returns.

### Receptors (`furnace::nerve::receptor`)

| Receptor | Source | Feature | Urgency |
|---|---|---|---|
| `Http` | a route, `POST` or `GET` | sketch of the body, or your own `fn(&Ingest) -> Feature` | `x-deadline-ms` header, else the route's `UrgencyRule` |
| `Channel` | your code: `ChannelSender::emit` runs the chain inline, `send` queues to the receptor's thread | whatever you give the signal | the rule |
| `Timer` | one signal per period | the tick, or one fixed feature | the rule |
| `Lines` | stdin, a TCP connection, or a TCP port it listens on | sketch of the line | the rule |
| `Sse` | an upstream `text/event-stream`, with reconnects and `Last-Event-ID` | sketch of the event's data | the rule |

An `UrgencyRule` is a fixed deadline (`within`) or a deadline derived from the stream's
measured rate (`before_next(1.0)`: act before the next sample). Every receptor measures its
rate and stamps it on the signal.

### Effectors (`furnace::nerve::effector`)

An `Effector` wraps an `Act` and supplies the loop: `Retry` (3 attempts from 100 ms backoff by
default), `Breaker` (open after 5 consecutive failures for 10 s, then one trial), `RateLimit`
(a token bucket that paces or drops) and `refractory`. Acts that finish at once run inline on
the reflex arc; the rest run on the efferent runtime. Implement `Act` for your own device.

| Act | Does |
|---|---|
| `Respond` | answers the request that started the chain: `{"status", "content_type", "body"}` or the signal's payload |
| `Webhook` | `POST url` with the command's args as JSON, or the payload as it is; 5xx and 429 are retried |
| `Process` | runs a local command with fixed arguments plus `{"args": [...], "stdin": "..."}` |
| `FileAppend` | appends `args.line`, or a JSON line with the time, the signal id and the args |
| `Channel` | hands `Delivery { command, signal }` to the code holding the receiver |
| `Sse` | pushes `{"signal", "effector", "args"}` to subscribers of `GET /nerve/effectors/{name}/events` |

### Brains (`furnace::nerve::brain`)

`Llm::new(name, engine)` takes any `LlmEngine` (feature `llm`): the effectors' names go into
the system prompt (`{effectors}` in your own prompt), the signal's payload is the user message,
and the answer is parsed as commands wherever JSON appears in it: `{"effector": "...", "args":
...}`, `{"name": ..., "arguments": ...}`, `{"tool_calls": [...]}`, or arrays of those. The
signals of one summation window are generated at once, so an engine with a scheduler batches
them. `ModelNeuron::brain(name, model, input, output)` does the same for any `Model`, with
your functions from signal to input tensors and from outputs to commands, one forward pass per
window; `ModelNeuron::reflex` is the same model inline on a reflex arc (a small classifier).

### Metrics

In `/metrics` (`nerve` in the JSON, `furnace_nerve_*` in Prometheus) and `GET /nerve`:

- per neuron: `received`, `fired`, `quiet_by_threshold`, `quiet_by_habituation`,
  `quiet_by_refractory`, `quiet`; effectors add `acted`, `failed`, `retries`, `dropped_open`,
  `dropped_rate`, the breaker's state and the latency to the world; brains add generations,
  commands and the generation latency
- per pathway: signals, p50 / p99 / mean / max latency from signal-in to signal-out, how many
  signals it was too slow for, and the current summation window
- `brain_fraction`: signals that reached a brain over signals received
- per reflex arc (receptor → effector, never queued): stimulus to effector call, p50 / p99

## The demo

```bash
cargo run --release --features nerve,llm --example nerve
```

[`examples/nerve`](../examples/nerve/main.rs) runs, in one process, a tiny webhook receiver
that prints the commands it gets, the server with the fabric above (plus an in-process channel
of 28×28 frames through a small Burn classifier whose verdict is the frame's brightness), and
a demo stream: 200 text events and 200 frames, half of them repeats, 5% urgent. Repeated
events die at the receptor; urgent ones reach the alarm webhook over the reflex arc in a few
microseconds; novel ones go to a mock brain that takes 100 ms to answer with three commands.
The summary comes from `/metrics`:

```text
signals received 400, reached the brain 190: brain fraction 0.475
events            receptor  fired 105  quiet: habituation 95
frames            receptor  fired 105  quiet: habituation 95
...
pathway triage -> alarm (reflex): 10 signals, p50 13.9 µs, p99 23.1 µs
pathway triage -> brain (deliberative): 95 signals, p50 779387.4 µs, p99 1314082.6 µs
reflex arc events -> alarm: stimulus to effector call p50 3.2 µs, p99 3.3 µs (10 times)
```

The deliberative pathway's seconds are the queue: 190 novel signals at a few hundred per
second against a brain that answers 16 at a time every 100 ms. That is the point of the
numbers: the brain is the bottleneck, and the fabric keeps everything else away from it.
`--serve` keeps the server up to poke at with `curl` (`POST /events`, `GET /nerve`).

## Measured

[`benches/nerve`](../benches/nerve) measures the fabric against a hand-written axum server
doing the same work in a handler (a sketch of the body, the same habituation table, the same
triage) and against FastAPI where Python can do it. Windows 11, one machine, otherwise idle,
medians of 3 runs:

```bash
cd benches/nerve && cargo build --release && target/release/furnace-nerve-bench
```

| Metric | Target | Measured | Compared with |
|---|---|---|---|
| Propagation cost per hop, no work | ≤ 1 µs | 146 ns | a chain of boxed closures: 4 ns |
| Reflex arc, stimulus to effector call, in process | ≤ 10 µs | 0.8 µs p50, 5.0 µs p99 (the fabric's own arc measurement: 0.4 µs) | the same work as a function: 0.1 µs |
| Brain fraction on a stream with 50% repeats | ≤ 0.5, falling | 0.496 after 1,000 signals, 0.472 after 2,000, 0.459 after 5,000, 0.454 after 10,000 | a plain request/response server: 1.0 |
| Memory of the fabric at 1k signals/s | ≤ 10 MB over the base | 0.4 MB (all of it the idle fabric: two runtimes and their threads; the signals add nothing measurable) | — |
| Stimulus to webhook arrival, reflex arc, localhost | ≤ 100 µs added over the raw call | 47 µs added (raw 108 µs, fabric 155 µs, p50) | the raw HTTP call with the same client |
| Throughput at 16 clients on the demo | ≥ the hand-written server | **missed**: 84,987 req/s, 11% below | hand-written 95,691 req/s; FastAPI 10,484 req/s |

The demo over HTTP, `POST /events`, N clients on keep-alive connections, 2,000 requests each:

| Server | Clients | Ready | Requests/s | p50 | p95 | Peak private memory |
|---|---|---|---|---|---|---|
| furnace (`Server::nerve`) | 1 | 32 ms | 14,376 | 0.067 ms | 0.098 ms | 9.5 MB |
| | 4 | | 51,201 | 0.071 ms | 0.130 ms | |
| | 16 | | 84,987 | 0.181 ms | 0.279 ms | |
| hand-written axum | 1 | 42 ms | 17,294 | 0.055 ms | 0.088 ms | 3.8 MB |
| | 4 | | 55,584 | 0.066 ms | 0.121 ms | |
| | 16 | | 95,691 | 0.162 ms | 0.241 ms | |
| FastAPI (uvicorn, 1 worker) | 1 | 833 ms | 5,147 | 0.192 ms | 0.252 ms | 41.2 MB |
| | 4 | | 9,671 | 0.399 ms | 0.513 ms | |
| | 16 | | 10,484 | 1.453 ms | 1.952 ms | |

Of 32,000 requests at 16 clients, about 15,750 were quiet (204, habituated at the receptor),
1,600 alarms (200, the reflex arc) and 14,650 went to the brain (202), the same on every
server.

Notes on the numbers:

- The per-hop cost includes two clock readings, the gate, the trace, the pathway's latency
  sample and the counters; a chain of closures does none of that. The no-fire path (a signal
  that dies of habituation at the receptor) is one clock reading, one table lookup and a
  counter.
- The throughput target is missed: the fabric server serves 11% fewer requests per second at
  16 clients and answers 12 µs later at one client. The work in the handler is the same; the
  difference is Furnace's request layer (a request id, CORS and security headers, the outcome
  and signal headers, the reply channel for a `Respond` effector) that a bare handler does not
  have, and that benches/burn measures at about the same gap for `/predict`. The fabric's own
  cost is the 0.8 µs of the arc row. Whether that layer should get thinner is a question for
  the server, not the nervous system.
- The webhook row's raw call takes about 108 µs here because every variant is paced at one
  call per millisecond, so each pays the wake-up of idle runtime threads (most of the time on
  Windows). Back to back, the raw call takes about 25 µs and the fabric's path about 190 µs:
  the motor side's thread is asleep when a sporadic signal arrives, and waking it is what the
  hop costs. A hand-written handler that sends the webhook from its own task does not pay that
  hop; the afferent/efferent split does, by design.
- FastAPI runs one uvicorn worker; its habituation table is an `OrderedDict`.

## What it is not, and limits

- Not a workflow engine or a DAG DSL: pathways and neurons are code.
- Not an agent framework: the brain's commands go to effectors; planning loops belong to the
  brain.
- Not a new engine: reflex models and brains are Burn, candle or anything behind `Model` or
  `LlmEngine`.
- The deliberative pathway has its own summation worker rather than the server's `Batcher`:
  the batcher is tied to `Model` and to a request/reply shape, and a signal that is summed
  has no caller to reply to. The algorithm is the same (a window from the first signal, a
  bound on the batch) with the window learned instead of configured.
- "Pinned" reflex threads are the receptors' own threads (a timer's, a line reader's, the
  request handler's for an HTTP receptor); there is no CPU affinity. A reflex that runs a
  model blocks that thread for the forward pass, which is what makes it a reflex; keep such
  models small or put them behind a deliberative pathway.
- A brain on a reflex pathway would block the firing thread for the whole generation; the
  fabric allows it and nothing good comes of it.
- The outbound client speaks plain `http://` with keep-alive; HTTPS webhooks are not yet
  supported.
- Receptors on `Server::nerve` run the synchronous chain inside the request handler, on a
  Tokio worker thread: a few microseconds for a reflex arc, the model's time for a reflex
  model.
