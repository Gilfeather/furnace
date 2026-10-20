# The nervous system

Inference getting faster is worth nothing if the layer that moves signals to and from the
models is the bottleneck. Furnace is that layer: not a server that
loads models and answers requests, but the nervous system between sensors, small fast models,
the large slow model (the brain, an LLM) and actuators. Text, images, tensors and model outputs
are one kind of signal. Most signals never reach the brain. The receptive side and the motor
side are separate systems. The mechanisms are a neuron's.

```toml
[dependencies]
furnace-server = { version = "0.10", default-features = false }
```

The nervous system is always built: the server's own routes run on it. The `nerve` feature is
still accepted and changes nothing. Everything is pure Rust and adds no dependency: `nerve`
uses what the crate already has, plus a small HTTP/1.1 client over Tokio sockets for the
outbound side (`furnace::nerve::wire`).

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
- *Refractory period*: after firing, the neuron ignores signals for a while. Of signals
  arriving together, one is admitted. Backpressure without counters.

**Pathway.** A directed connection with a **measured** rolling p50/p99 latency, never
configured. Reflex pathways are direct calls on the firing thread: `Receptor::stimulate` →
`Reflex::stimulate` → `Effector::stimulate`, no channel between them. Deliberative pathways
queue to a worker thread that sums the signals of a short window into one firing (temporal
summation); the window is learned from the pathway's latency (an eighth of its p50, between
200 µs and 20 ms). The queue and the worker are the server's batching engine (the one behind
`--max-batch-size`), run with one worker per pathway, a bounded queue and the learned window
in place of the configured timeout.

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

Server::new().port(3000).nerve(b).serve().await?;
```

`Server::nerve` takes the builder, adds the server's models and LLMs to it (see below), mounts the HTTP receptors as routes, starts the other receptors, serves
`GET /nerve` (neurons, mechanisms, pathways and their measured latencies) and
`GET /nerve/effectors/{name}/events` (an SSE effector's commands), and adds the fabric's
metrics to `/metrics`, next to whatever else the server serves. A request to a receptor is
answered by a `Respond` effector on its chain; otherwise it gets `202 Accepted` when its signal
was queued to the brain, or `204 No Content`, with what happened in the `x-furnace-nerve`
header (`quiet:habituation`, `acted:1`, `queued`). A fabric can also be driven without a
server: `fabric.stimulate(receptor, signal)` runs the whole synchronous chain before it
returns.

## The server runs on it

The server's own routes are receptors and effectors on the same fabric. Each model has an
entrance `in:{model}`, its neuron `model:{model}` and the effector `serve:respond`: `/predict`,
`/models/{model}/predict`, v2 over HTTP and gRPC stimulate the entrance, the model fires its
output tensors, and `serve:respond` hands them back to the route, which encodes them for its
protocol. With batching off (the default) the model's pathway is a reflex, and the forward
pass runs on a blocking thread of its own, as a hand-written server would; with
`--max-batch-size` above 1 it is a deliberative pathway, whose summation is the server's
dynamic batching. Each LLM is a brain, `llm:{llm}` behind `in:{llm}`, whose stream of tokens
goes straight back to the OpenAI route. `GET /nerve` shows all of them with their measured
latencies, next to your own neurons; the names `in:*`, `model:*`, `llm:*` and `serve:respond`
are the server's.

Measured on the same laptop as below, before and after the routes moved onto the fabric: the
server's own work for a request to a trivial model went from about 17 µs to about 21 µs (three
alternating runs each); over HTTP to the digits model, 1 and 16 clients, the two are the same
within noise.

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
| `Respond` | answers the request that started the chain: `{"status", "content_type", "body"}` or the signal's payload; with nobody waiting any more it does nothing, which is not a failure |
| `Webhook` | `POST url` with the command's args as JSON, or the payload as it is; 5xx and 429 are retried; plain `http://`, no redirects; the timeout covers the whole call, response body included |
| `Process` | runs a local command with fixed arguments plus `{"args": [...], "stdin": "..."}`; writing stdin counts against the timeout; the args reach argv as they are, with no shell |
| `FileAppend` | appends `args.line`, or a JSON line with the time, the signal id and the args, on a thread of its own, in order |
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
  `quiet_by_refractory`, `quiet` (signals the neuron itself was not interested in, not a total;
  `reason="not_interested"` in Prometheus); effectors add `acted`, `failed`, `retries`,
  `dropped_open`, `dropped_rate`, `quiet_refractory`, commands in flight, the breaker's state
  and the latency to the world; brains add generations, commands and the generation latency
- per pathway: signals, p50 / p99 / mean / max latency from signal-in to signal-out, how many
  signals it was too slow for, and the current summation window
- `brain_fraction`: signals that reached a brain over signals received
- per reflex arc (receptor → effector, never queued): stimulus to effector call, p50 / p99

The JSON has all of it. Prometheus has the fabric's counters and `brain_fraction`, each
neuron's counters (`kind` and `reason` labels), each pathway's signals (`kind="reflex"` or
`"deliberative"`), the pathway and arc latencies as `quantile="0.5"` and `"0.99"` gauges in
seconds with a `_count`, and each effector's counters, `inflight` and `circuit_open`. What a
brain, a recall neuron or an expectations neuron adds, and an effector's latency, are in the
JSON only.

## The demo

```bash
cargo run --release --features llm --example nerve
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

## Consolidation

Consolidation level 1 is a **verified cache of the brain's decisions**, and the foundation for
what comes later (keys by similarity, expectations the brain writes into the periphery). The
brain's answers used to vanish: the thousandth signal of the same kind was asked of the brain
again, at the full price. With consolidation (`furnace::nerve::consolidation`) the fabric
watches what the brain decides for each key. Once it has decided the same thing for the same
key a few times in a row, the decision becomes an *engram* and the next such signal is answered
by a reflex in microseconds, without the brain. A sample of the reflex's answers keeps being
checked against the brain; an engram the brain disagrees with is dissolved.

```text
receptor ──► Recall ──(consolidated engram)──────────────────────────► effectors
               │
               └──(miss, or shadow check)──► Record(brain) ──(commands)──► effectors
                    deliberative pathway          │
                                                  ▼
                                        Memory (EngramStore)
changes ──► Inhibit ──(tags, dependency versions)──┘
```

```rust
use furnace::nerve::consolidation::{Consolidation, EngramStore, Inhibit};

let engrams = EngramStore::new(
    Consolidation::new()
        .effectors(["page", "ticket", "ignore"]) // what may become a reflex
        .after(3)                                 // agreeing decisions to consolidate
        .shadow(0.05)                             // fraction of answers re-checked
        .volatile(["reason"])                     // argument fields not compared
        .depends_on(["customers"])                // external state the brain consults
        .tags(|s| customer_of(s).map(|c| vec![format!("customer:{c}")]).unwrap_or_default())
        .forget_after(Duration::from_secs(3600)),
);
// Effector failures reach the engrams they acted for
b.neuron(Effector::new(engrams.watch(Webhook::new("page", url))), Mechanisms::NONE);
let (recall, _brain) = b.consolidate(brain, Mechanisms::NONE, Summation::default(), &engrams);
b.reflex(triage, recall, 1.0); // whatever fed the brain now feeds `recall`
// Change notifications: {"inhibit": ["customer:42"]} or {"advance": ["customers"]}
let changes = b.neuron(Inhibit::new("changes", &engrams), Mechanisms::NONE);
b.reflex(db_changes, changes, 1.0);
```

- **Key.** Level 1 is exact: the signal's `Feature` by default, or `Consolidation::key`. What
  makes it useful is the receptor or a reflex in front of `recall` putting the right thing in
  the key: a log line with its numbers and ids masked, a quantized sensor state, a normalized
  request. A fired decision carries its key as its feature.
- **`Recall`** (a reflex in front of the brain, named `<brain>-recall`) looks the key up. A
  consolidated engram fires its commands at once; anything else passes on to the brain's
  deliberative pathway, as before.
- **`Record`** wraps the brain: a brain's commands are routed by name straight to their
  effectors, so a neuron on a pathway behind the brain would never see them. Wrapping it sees
  each firing next to the signal that caused it, which is the decision for that key.
- **`Memory`** is the trait `Recall`, `Record`, `Inhibit` and `Watched` talk to: what a key is,
  what is stored and how it is looked up. `EngramStore`, the one built in, is a bounded
  set-associative table like the habituation table, holding commands; another memory (other
  keys, other contents) can take its place without touching the neurons.
- **Consolidated** after `after` agreeing brain decisions in a row (3); a different decision
  starts the count over. After an engram has been dissolved by the brain or by failures, the
  same key needs `after` more agreeing decisions for each dissolution.
- **Shadow checks.** A fraction of the reflex's answers (5%, and always the first 3 of every
  engram, chosen deterministically from the signal's id) also go to the brain, after the reflex
  acted and without the request's reply. The brain's answer is compared and dropped. A
  disagreement dissolves the engram.
- **Policy.** Opt-in per effector: a decision with a command for an effector not listed is never
  answered from an engram (`delete`, `pay`). Two decisions agree when their commands are equal
  with the `volatile` argument fields left out; the engram fires the latest brain decision as
  it was made.
- **Effector failures.** `engrams.watch(act)` wraps an effector's act; `failures` failures in a
  row (3, every attempt counts) for a key dissolve its engram.

### What a decision may depend on

A brain declared with no `depends_on` is taken to **decide from its input alone**: the same key
always gets the same answer. That holds for a log triage on the line's text; it does not hold
for a brain that looks up a customer's plan before deciding. Such a brain declares what it
consults, and every change of that state has to reach the store:

- **Tags (inhibition).** `Consolidation::tags` labels each decision with what it is about
  (`customer:42`). A change notification with `{"inhibit": ["customer:42"]}` reaching an
  `Inhibit` neuron (from a receptor fed by the database's change stream, a webhook, a queue)
  dissolves every engram carrying the tag at once, and the next such signal goes to the brain.
  A brain decision for a stimulus seen before the change, still on its way back, is not
  recorded. No penalty: the engram may consolidate again as soon as the brain agrees with
  itself under the new state.
- **Versions.** Every declared dependency has a version. `{"advance": ["customers"]}` (or
  `Memory::advance`) raises it, and every engram made under the old version is void, as are
  decisions made under it that come back later. This is the sure way when changes cannot be
  tagged (a bulk import, a migration).
- **Forgetting and shadow checks are the last line, not the mechanism.** An engram unused for
  `forget_after` (an hour) is forgotten, and the shadow sample catches a reflex the brain no
  longer agrees with, but only on that sample and only after the fact. A dependency nobody
  declared is a bug in the brain's declaration; these two bound how long it can last.

### Metrics of consolidation

On the recall neuron in `GET /nerve` (with the policy, the dependencies' versions and the most
used engrams) and in the JSON of `/metrics`: signals, sent to the brain, hits, shadow checks,
agreements, disagreements, consolidated, dissolved (by failures), inhibited, stale, forgotten,
decisions refused by the policy, `brain_fraction` and `brain_load` (shadow checks included),
the brain fraction per window of 1,000 signals, the brain's signal-to-decision p50 and the
brain time the hits saved at that p50. `Inhibit` counts the tags and versions it applied.

Not built yet: keys by similarity (locality-sensitive or perceptual-hash keys) and distilling
the engrams into a small model. Expectations the brain writes into the periphery, of which this
cache is the simplest kind (an exact match that acts), are
[below](#expectations-the-brain-programs-the-periphery).

### Measured: log triage

[`benches/nerve`](../benches/nerve/src/consolidation.rs) replays two public logs from
[Loghub](https://github.com/logpai/loghub), `HDFS_2k.log` and `BGL_2k.log` (2,000 lines each;
BGL's label column dropped), through a `Lines` receptor over TCP at 50 lines per second. A
reflex makes the feature each line's template (every token holding a digit masked). The brain
is a mock LLM: a summed window of up to 8 lines takes 400 ms plus 20 ms per further line, 4
generations at once. It decides on the raw line: `page` for errors and failures, `ticket` for
warnings and for HDFS blocks stored short of the 64 MB block size, `ignore` otherwise, with a `reason`
quoting the line (volatile). The block size is masked out of the template, so that template's
decision depends on what the key does not see: the case consolidation must not get wrong.
One run each, consolidation off and on, Windows 11:

```bash
cd benches/nerve && cargo build --release
target/release/furnace-nerve-bench consolidation --logs DIR
```

| Log (templates) | Consolidation | Brain calls | Brain calls per 200 lines, in order | Line to action p50 | p99 | Actions equal to the brain's for that line | Shadow agreement |
|---|---|---|---|---|---|---|---|
| HDFS (16) | off | 2,000 | 200 every time | 727 ms | 954 ms | 100% | — |
| | on | 328 | 127, 46, 58, 25, 14, 13, 4, 12, 15, 14 | 0.04 ms | 809 ms | 98.4% | 97.1% (136 of 140) |
| BGL (142) | off | 2,000 | 200 every time | 723 ms | 949 ms | 100% | — |
| | on | 666 | 127, 113, 39, 10, 17, 37, 130, 59, 50, 84 | 0.03 ms | 881 ms | 100% | 100% (124 of 124) |

With consolidation on, the reflex's own answers took 0.04 ms (HDFS) and 0.03 ms (BGL) at p50
and under 0.1 ms at p99, against about 0.55 to 0.6 s through the brain; the p99 rows are the
lines that still went to the brain. HDFS: 14 consolidations, 1,810 reflex answers, 139 shadow
checks, 4 dissolutions; the brain time the hits saved, at its p50, was about 1,060 s over a
40-second replay. BGL: 61 consolidations, 1,458 reflex answers, 124 checks, no dissolution,
about 810 s saved.

What the numbers say, and what they do not:

- The brain's load falls where traffic repeats: by 6× on HDFS (16 templates) and 3× on BGL
  (142 templates). Not by an order of magnitude on 2,000 lines: every template costs at least
  `after` brain decisions plus its first `first_shadows` checks, and BGL keeps showing new
  templates (the rise at lines 1,200 to 1,400 is a burst of templates not seen before). The
  share falls further the longer the stream runs on known templates.
- The HDFS trap worked as intended and shows the risk. The brain tickets a block stored short
  of 64 MB (35 lines in the sample) and ignores a full one; the template masks the size, so
  both share one key. The engram for that template consolidated on `ignore`, then the shadow
  checks caught the disagreement and dissolved it 4 times, each time needing more agreeing
  decisions to come back. In between, 33 partial-block lines were answered `ignore` by the
  reflex: 1.6% of all actions differ from what the brain would have done. A key that hides
  what the decision depends on gives wrong reflexes until a check catches them. The remedy is
  the key (keep the field the brain looks at), not more checks; the dissolution count in
  `GET /nerve` is how to find such keys.
- The brain here is a mock that decides from the line alone (no `depends_on`), at a fixed
  latency, so the numbers measure the mechanism, not an LLM's answers. A real GGUF is the next
  measurement.
- One run per mode, on one idle Windows 11 machine, 50 lines per second. The JSON report has
  every count (`--json PATH`).

## Expectations: the brain programs the periphery

Consolidation remembers the brain's past answers, and the brain is still asked first about
anything new. Expectations (`furnace::nerve::expect`) turn that around, the way predictive
coding describes the cortex: the brain does not only answer signals, it writes down how the
stream should behave from now on, and those rules run next to the receptor in microseconds.
A signal that fits is absorbed (or acted on by the rule itself) and goes no further. Only what
breaks a rule, what no rule covers, and what was expected and **did not come** reach the brain,
as a *surprise* with its context. The brain's work goes into writing the rules that keep the
rest away from it.

```text
             top-down: expectations, through the `expect` effector
       ┌──────────────────────────────────────────────────────────────┐
       ▼                                                              │
receptor ─► Expectations ──matched: absorb / the rule's commands──► effectors
               │   ▲ deadlines (absences)                             │
               └──surprise──► brain ──commands──► effectors           │
                                   └──expect commands────────────────┘
```

```rust
use furnace::nerve::expect::{ExpectConfig, Expectations};

let expectations = Expectations::new(
    "logs-expect",
    ExpectConfig::new().effectors(["ticket"]), // what a rule's own commands may reach
);
let expect = b.receptor(expectations.clone(), Mechanisms::NONE); // its timer raises absences
b.reflex(logs, expect, 1.0);
b.neuron(Effector::new(expectations.effector("expect")), Mechanisms::NONE); // top-down
b.deliberative(expect, brain, 1.0, Summation::default());
```

The brain installs rules by commanding the `expect` effector, which goes through the same
command parsing, policy and metrics as any other effector:

```json
{"effector": "expect", "args": {"add": {
  "id": "hb-a", "kind": "every", "period_ms": 5000, "tolerance_ms": 1000,
  "selector": {"template": "heartbeat from svc-a"},
  "learn": {"period_ms": [4000, 8000]},
  "provenance": {"surprise": "unexpected heartbeat from svc-a"}
}}}
```

### The language (v1)

Declarative and schema-checked; unknown fields are refused. Each kind compiles to a matcher
with a few words of state.

| Kind | Holds when | Example |
|---|---|---|
| `match` | a matching signal arrives; with `on_match` commands, act | "this template → ticket" (the cache case) |
| `rate` | matching signals per `window_ms` stay within `[min, max]` | "`WARN disk <*>` at most 3 a minute" |
| `every` | one matching signal at least every `period_ms` ± `tolerance_ms` | "heartbeat from svc-a every 5 s" |
| `range` | the numeric `field` stays within `[min, max]` | "temperature 18 to 27" |
| `then` | a match is followed by one matching `then` within `within_ms` | "job start → job end within 10 min" |
| `never` | no matching signal at all | "no `auth failure for root`" |

Every expectation also has a `selector` (a template and field equalities), `on_match`
(`"absorb"`, the default, or `{"act": [commands]}`), `on_violation` (`"escalate"`, the default,
or `{"act": [commands]}` and escalate), `learn` (parameters the periphery may tune itself, each
within `[low, high]` the brain sets), `ttl_s`, `depends_on` and `provenance`. Besides `add`, the
effector takes `replace`, `retire`, `verdict` and `changed`.

**Fields.** The extraction is part of v1: a log line's template is the line with every token
holding a digit masked as `<*>`, and the masked tokens are the fields `v0`, `v1`, ...; a
`key=value` token is also the field `key`. A JSON body's fields are its members, nested objects
joined with dots, and its template is the sorted list of its field names.

**Surprises** are JSON signals under `"surprise"`: the reason (`unexpected`, `violation`,
`absence`, `shadow`), what broke (expectation, kind, observed and expected), the observation
(template, fields, text), up to `context` (8) recent signals of the same template, and the
expectations installed for that template now. An absence comes from the neuron's timer (a
deadline heap on a thread of its own, woken when an earlier deadline is armed) and is reported
once per silence; the next match arms it again.

### Keeping it honest

- **Shadow checks.** Every match of a young expectation (the first 5) and a sample of the rest
  (1%) also go to the brain as a `shadow` surprise, after the rule acted and without the
  request's reply. The brain answers `{"verdict": {"id": ..., "correct": false}}` when it would
  not have absorbed it; `retire_after` (2) such answers retire the rule.
- **False alarms.** `{"verdict": {"id": ..., "normal": true}}` answers a surprise the brain finds
  normal: a parameter the rule lets the periphery `learn` moves to cover what was seen, within
  the brain's bounds. An `every` whose period is learnable also follows the observed interval.
- **Scope limits**, because log content reaches an LLM and an attacker can try to make it write
  "absorb everything": an absorbing rule must name a template or fields, at least as
  specifically as the surprise it answers (no wildcard absorb); a rule's commands only reach
  the effectors in `ExpectConfig::effectors`; at most `max` (256) rules and `writes_per_minute`
  (60) writes, refused ones included; with `approval`, new absorbing rules from the brain wait
  for `Expectations::approve`, which is the operator's and not a command.
- **Forgetting.** A rule not matched for its `ttl_s` (or an hour) expires; rules that declare
  `depends_on` retire when that state changes (`Expectations::changed`, or `{"changed": [...]}`).

### Metrics of expectations

On the neuron in `GET /nerve` (the rules with their counters and provenance, those waiting for
approval, the last retirements) and in the JSON of `/metrics`: signals, absorbed, acted,
unexpected, violations by kind, absences, surprises, shadow checks and wrong verdicts, false
alarms, parameters learned, installed, replaced, retired, expired, refused, the surprise rate
and its timeline per 1,000 signals.

### An LLM that writes expectations

`furnace::nerve::expect::llm::brain(name, engine)` (feature `llm`) is an `Llm` brain over any
`LlmEngine` that knows the language: its system prompt teaches the kinds, the selector, the
scope rules ("an absorbing expectation must name the template of the surprise it answers";
"never absorb errors, failures or security events") and the `expect` commands; its user message
renders each surprise with the signal, its template and fields, what broke, the recent signals of
the template and the expectations installed for it now, and says which answer the surprise
wants: an expectation for a template that is normal, an action, or a `verdict` (`correct: false`
on a shadow check, `normal: true` on a false alarm). The answer is parsed as commands as for any
`Llm` brain, so the `expect` effector's scope limits and budgets apply to it unchanged.
`SYSTEM_PROMPT` and `surprise_prompt` are public, to use with an `Llm` of your own. Tested with a
scripted engine that writes a `match` for routine lines: after its first answer the template stops
reaching it. How well a real model writes rules is the experiments' question.

### Checked on a synthetic stream

The public datasets of the design (Loghub BGL with its anomaly labels, the Numenta Anomaly
Benchmark) are not downloaded yet. Until then a test runs ten simulated minutes, time passed in,
with a scripted brain that learns the shape of what is normal (a `match` per routine template,
an `every` for the heartbeat, a `range` for a sensor, a `never` after a root login) and pages
on what is not: 6,846 signals of five routine templates, a heartbeat every second that drops out
for ten seconds, a reading every two seconds that leaves its range three times, three kernel
panics and two root logins. Every one of the 8 anomalies reached the brain and was paged, the
dropout was one absence, nothing normal was paged, and the brain was called for 16 surprises,
21 checks of young rules and a 1% sample of the rest: about 110 calls, 1.6% of the stream. This
checks the mechanism; it measures nothing about an LLM writing the rules, which is the next
step (the LLM adapter), nor about real logs.

## What it is not, and limits

- Not a workflow engine or a DAG DSL: pathways and neurons are code.
- Not an agent framework: the brain's commands go to effectors; planning loops belong to the
  brain.
- Not a new engine: reflex models and brains are Burn, candle or anything behind `Model` or
  `LlmEngine`.
- A deliberative pathway is the server's dynamic batching (`src/batching`): a queue, workers
  that merge what is queued once they are all busy, a window from the first item, a bound on
  the batch. Your pathways run it with one worker, a bounded queue and the window learned
  instead of configured; a model's pathway with the configured workers and timeout.
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
