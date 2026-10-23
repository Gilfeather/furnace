# The nervous system

Inference getting faster is worth nothing if the layer that moves signals to and from the
models is the bottleneck. Furnace is that layer: not a server that
loads models and answers requests, but the nervous system between sensors, models and
actuators. Text, images, tensors and model outputs are one kind of signal. Signals run through
any number of junctions, and at each one they act now or pass on; a brain (an LLM, any large
model, a remote server) is one possible end, and most signals never reach it. Between the
brain and the body sits a spine: the brain says "get dressed" to it and nothing more, the
spine's motor junctions break that down, and a nerve per body part (the right arm, the left
arm) carries the commands to the hand and the fingers, which retry a slipped grip themselves
and report back up. The receptive side and the motor side are separate systems. The
mechanisms are a neuron's.

```toml
[dependencies]
# `llm` for an LLM brain (`nerve::Llm`); the rest of the nervous system needs no feature
furnace-server = { version = "0.15", features = ["llm"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
tracing-subscriber = "0.3"
```

This page follows `main`, which is released as 0.15; until 0.15 is on crates.io, use
`furnace-server = { git = "https://github.com/Gilfeather/furnace", features = ["llm"] }`.

The nervous system is always built: the server's own routes run on it. Everything is pure Rust
and adds no dependency: it uses what the crate already has, plus a small HTTP/1.1 client over
Tokio sockets for the outbound side (internal to the crate).

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Gilfeather/furnace/main/docs/images/nervous-system-dark.svg">
  <img alt="Receptors feed a network of junctions of any depth that branches and merges; at every junction a signal acts now through an effector or passes on to the next junction; an optional brain is one possible end. Below, the motor side: the brain talks to the spine only, in actions; the spine carries actions down and senses, questions and outcomes up; a nerve per body part carries its own load to its motor junctions and effectors, where retries and adjustments close" src="https://raw.githubusercontent.com/Gilfeather/furnace/main/docs/images/nervous-system-light.svg" width="900">
</picture>

A fabric is any number of junctions (neurons) wired as you like: a chain, a tree or a net,
branching and merging. Signals enter at receptors (the afferent side) and leave at effectors
(the efferent side). At every junction, every signal is routed on its own: it acts now (a
reflex) or passes on to the next junction (deliberation), chosen by its deadline and the
pathways' measured latency. Nothing central decides. Depth has no limit but 32 hops per
signal, a guard against wiring loops, not a limit on stages. A brain is one possible
end, not a requirement, and a reflex can be a small model, a rule or a function.

How this differs from a cache: only habituation, in a gate, is cache-like (a repeat is
attenuated instead of recomputed). The rest is routing at every junction and acting on the
world.

**Signal.** One type for everything that moves (`furnace::nerve::Signal`): a `Feature` (a
64-bit sketch plus a kind, what habituation and thresholds compare), an `Urgency` (a deadline
and the rate the receptor saw), an `intensity` in 0..1, a lazily shared `Payload` (bytes, text,
JSON, tensors or a `Command`), a `Trace` of the neurons it passed, and, for a signal that
started as a request, the way back to it. Signals are owned Rust values: no serialization and
no copy between neurons.

**Neuron.** Anything that takes a signal and may fire: `fn stimulate(&self, Signal, &Cx) ->
Firing`, where `Firing` is `Quiet`, `Fire(Vec<Signal>)` or `Compute(closure)` for blocking
work. Five kinds: receptors (world → signal), reflexes (feature → command, in process), brains
(an optional slow model; its output is commands), motor neurons (command → commands below
them, see [the motor side](#the-motor-side)) and effectors (command → world).

**Mechanisms.** A neuron has none until you give it some with `b.gate(id, Mechanisms)`. Each
is optional and a few words of state, run by the fabric before `stimulate`:

- *Threshold*: fires only if `intensity >= threshold`.
- *Habituation*: a bounded, set-associative table keyed by feature. Each repeat multiplies the
  intensity by `1 - a` and raises `a` toward 1 by `gain`; `a` decays as `exp(-dt / tau)`. A
  repeated stimulus dies here, below the threshold, and recovers with time. This is caching as
  a property of every neuron, not a layer.
- *Refractory period*: after firing, the neuron ignores signals for a while. Of signals
  arriving together, one is admitted. Backpressure without counters.

**Pathway.** A directed connection out of a junction, with a **measured** rolling p50/p99
latency, never configured. A junction can have any number of them, to other junctions or to
effectors, and any number can lead into one. Reflex pathways are direct calls on the firing
thread: `Receptor::stimulate` → `Reflex::stimulate` → `Effector::stimulate`, no channel between
them. Deliberative pathways queue to a worker thread that sums the signals of a short window
into one firing (temporal summation); the window is learned from the pathway's latency (an
eighth of its p50, between 200 µs and 20 ms). The queue and the worker are the server's
batching engine (the one behind `--max-batch-size`), run with one worker per pathway, a bounded
queue and the learned window in place of the configured timeout.

**Selection.** Made at every junction, for every signal it fires, with nothing central
deciding: of the pathways leaving that junction, the candidates are those whose p99 fits in the
signal's remaining deadline; among them the heaviest weight wins; with no deadline every
pathway is eligible. Latencies are measured as signals flow, so when the pathway onward gets
slow it drops out and the reflex takes over. Give the deliberative pathway the heavier weight
(the builder does: reflex 1, deliberative 2) and the reflex is what happens when there is no
time to pass on, like the fly's short-mode escape. A command (`Payload::Command`) names its
effector or motor neuron and goes there, from any junction.

**Preparation.** A signal with time to spare that takes a deliberative pathway calls
`prepare()` on the neurons it will probably reach (the brain, and the effectors after it), at
most once a second each: a webhook effector opens its connection in advance.

**Afferent and efferent.** Receptors sample on their own threads or tasks and never wait on
the motor side. Effectors act on their own, on the fabric's efferent runtime, with their own
loops: retry with backoff, a circuit breaker, a rate limit, a refractory period. The brain is
never in a retry loop. Acting on the world is the point: an HTTP response is one effector
among the webhook, the local command, the journal file, the in-process channel and the
Server-Sent-Events push. Between the brain and the effectors sit the motor neurons, described
next.

## The motor side

The receptive side is many junctions deep; so is the motor side. A brain should not have to
say "turn the right shoulder to 40°, then the elbow to 90°, then close the thumb": it says
"get dressed", and the junctions below it know how. Two pieces of skeleton hold them: the
spine between the brain and the body, and a nerve per body part
([below](#the-skeleton-spine-and-nerves)); motor neurons are the junctions in them.

**Motor neuron.** `b.motor(name, program)` adds one. A command addressed to its name (from a
brain, a reflex, or another motor neuron) starts its program, an async closure, on the motor
side's runtime, so the receptive side never waits for it. The program gets a `Motion` and
carries the command out:

- `m.command(name, args).await` commands a motor neuron or an effector below and waits for its
  outcome: the value the effector's act returned (what a device measured, a service's JSON
  answer) or why it failed after the effector's own retries; for a motor neuron, what its
  program returned. One after the other is `.await` in turn; `m.together([...]).await` sends
  several at once and waits for all, the first failure stopping the others. Any depth, any
  branching.
- `m.sense(name)` receives a copy of every signal that passes a neuron's gate from then on:
  wait until the eye sees the coat, or adjust a grip by the pressure the fingertips report.
  Together with the outcomes, that is the local loop: retry a slipped grip, squeeze harder, all
  without the brain.
- `m.ask(payload).await` fires a small question along the motor neuron's own pathways and
  waits for what the neuron there fires in answer. The pathway is selected as any signal's: by
  the deadline and the measured latencies, so with no time for the brain a quick reflex
  answers, and with no pathway in time the question fails and the program does without.
- The command's deadline bounds the program and everything it commands
  (`MotionError::TimedOut`); `m.within(duration, step)` bounds one step.

**Outcomes go up.** What a program returns goes back to the program that commanded it. When a
brain or a reflex sent the command, nobody waits for it: the outcome is fired along the motor
neuron's pathways as `{"action": "get_dressed", "status": "done", "result": ...}` (or
`"failed"`, `"timed_out"`, `"cancelled"` with a `"reason"`). Wire one back to the brain
(`b.deliberative(motor, brain)`, which also carries the questions) and the brain hears how its
action ended.

**Stopping.** One program runs per motor neuron, as one body part does one thing at a time: a
new command stops the program running at once (its commander hears `cancelled`) and starts its
own. `fabric.stop(name)` (or `m.stop(name)` from another
program, a `stop` action say) stops it. A stopped program stops everything it commanded and
still waits for, all the way down, and an effector whose commander stopped waiting gives up
its retries, as it does once the command's deadline has passed.

**Actions: what the brain sees.** `b.action(id, description, args)` offers a motor neuron or an
effector to brains, with what it does and the shape of its arguments. An LLM brain's prompt
lists the actions and nothing else, so the joints and fingers below them stay out of its sight
and out of its answers. A brain with effectors it should call directly declares those as
actions too.

```rust,no_run
use furnace::nerve::prelude::*;
use serde_json::{Value, json};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut b = Fabric::builder();
// What the brain may call: the hand, not its fingers
let hand = b.motor("grasp", |m| async move {
    let mut force = 0.4;
    for _ in 0..3 {
        // Three fingers at once, then the grip
        let fingers = ["thumb", "index", "middle"]
            .map(|f| Command::new(&format!("finger:{f}"), json!({"curl": 70})));
        m.together(fingers).await?;
        match m.command("grip", json!({"force": force})).await {
            // The fingertips' pressure, as the grip effector measured it
            Ok(felt) if felt["pressure"].as_f64() >= Some(0.6) => return Ok(felt),
            // Too light: squeeze harder, without asking anyone
            Ok(_) => force += 0.3,
            // Slipped: try again
            Err(_) => {}
        }
    }
    Err(MotionError::failed("could not get a grip"))
});
b.action(hand, "Grasp what is in front of the hand.", Value::Null);
for finger in ["thumb", "index", "middle"] {
    b.motor(&format!("finger:{finger}"), move |m| async move {
        let curl = m.args()["curl"].clone();
        m.command(&format!("knuckle:{finger}"), json!({"angle": curl})).await
    });
    b.effector(Webhook::new(&format!("knuckle:{finger}"), "http://127.0.0.1:9000/joint")?);
}
b.effector(Webhook::new("grip", "http://127.0.0.1:9000/grip")?); // answers {"pressure": ...}
let fabric = b.build()?;
# let _ = fabric;
# Ok(())
# }
```

### The skeleton: spine and nerves

**Spine.** `b.spine(brain)` is the brain's only way to the body. `.ascend(receptor)` is the
ascending tract: what passes the receptor's gate and reflexes goes up to the brain, summed
into its firings (a deliberative pathway). `.action(id, description, args)` is the descending
tract: the brain may call that motor neuron or effector and is told about it; a motor
neuron's questions and how its programs ended ascend to the brain. A brain's commands reach
its actions and nothing else: a command from a brain to a joint is unrouted, even if the
model names one. The motor neurons that are not in a nerve, and the reflexes, run in the
spine, never in the brain.

**Nerve.** `b.nerve("right_arm", |b| { ... })` is the trunk to one part of the body: every
neuron the closure adds (motor neurons, effectors, receptors, reflexes) belongs to it. Its
lane bounds the part's acts: every act of one of its effectors takes a place in the lane,
whoever commanded it, from inside the part or outside. At most `Nerve::in_flight` acts are
carried out at once (16 unless `b.nerve(Nerve::new("right_arm").in_flight(4), ..)`), each
until it ends, retries included, and the rest wait, the nearest deadline first. An idle lane
costs nothing; a part that is flooded makes its own acts wait and no other part's, so a jam in
the right arm does not stop the left. Motor programs never take a place and never wait in a
lane: one runs per motor neuron and a newer command stops it at once, full lane or not. An act
waits for no other command and always ends, so parts commanding each other, or motor neurons
commanding each other in a cycle, never deadlock, deadlines or not. A lane holds up to 1,024
waiting acts and then drops the least urgent, which fails: the program that commanded it hears
the failure, and from there it goes up to the brain as any outcome. `GET /metrics` shows each
nerve's lane (`in_flight`, `waiting`, `waited`, `dropped`).

The motor side's design, the nerves' lanes and how a signal is routed are specified in TLA+
and model-checked with TLC in CI: [`specs/`](../specs/README.md) lists the properties checked
and what the checks found.

```rust,no_run
use furnace::nerve::prelude::*;
use serde_json::{Value, json};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut b = Fabric::builder();
let thermometer = b.receptor(Http::new("thermometer", "/temperature").text());
let brain = b.neuron(FnNeuron::reflex("brain", |signal, _| {
    // Stands in for an LLM: it is told the actions and nothing below them
    signal.command("get_dressed", json!({"warmth": 0.9}))
}).with_kind(NeuronKind::Brain));
let dress = b.motor("get_dressed", |m| async move {
    m.command("grasp", json!({"item": "coat"})).await?;
    m.together([Command::new("sleeve:left", Value::Null), Command::new("sleeve:right", Value::Null)])
        .await?;
    Ok(json!({"wearing": "coat"}))
});
b.spine(brain)
    .ascend(thermometer)
    .action(dress, "Put on something warm.", json!({"warmth": "0 to 1"}));
b.nerve("right_arm", |b| -> Result<(), BuildError> {
    b.motor("grasp", |m| async move { m.command("grip", json!({"force": 0.7})).await });
    b.motor("sleeve:right", |m| async move { m.command("elbow:right", json!({"angle": 30})).await });
    b.effector(Webhook::new("grip", "http://127.0.0.1:9000/grip")?);
    b.effector(Webhook::new("elbow:right", "http://127.0.0.1:9000/elbow/right")?);
    Ok(())
})?;
b.nerve("left_arm", |b| -> Result<(), BuildError> {
    b.motor("sleeve:left", |m| async move { m.command("elbow:left", json!({"angle": 30})).await });
    b.effector(Webhook::new("elbow:left", "http://127.0.0.1:9000/elbow/left")?);
    Ok(())
})?;
let fabric = b.build()?;
# let _ = fabric;
# Ok(())
# }
```

[`examples/get_dressed.rs`](../examples/get_dressed.rs) runs the whole story on this skeleton
(the spine, a right-arm and a left-arm nerve) with stand-ins for the eyes, the joints and the
brain, and prints it as it happens:

```bash
cargo run --example get_dressed
```

The thermometer reads 4 °C and the brain, which knows only `get_dressed` and `stop`, answers
`get_dressed`. `get_dressed` looks for a coat in what the eyes stream for 300 ms; there is
none, so it asks the brain what to wear instead and waits for "the jacket, on the hook". It
commands `reach` (the right shoulder, then the elbow) and `grasp` (three fingers at once, then
the grip); the first grip slips and the hand tries again, then squeezes harder by the pressure
the fingertips report. Both sleeves go on at once, and the brain hears
`{"action": "get_dressed", "status": "done", "result": {"wearing": "jacket"}}`. The brain is
reached three times: the cold, the question and the report.

## The API

`use furnace::nerve::prelude::*` brings what building a fabric takes. The smallest fabric
that does something, a reflex arc (receive, decide in process, act):

```rust,no_run
use furnace::Server;
use furnace::nerve::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init(); // Furnace logs through `tracing`
    let mut b = Fabric::builder();
    let events = b.receptor(Http::new("events", "/events").text()); // POST /events
    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
        let text = signal.payload.as_text()?;
        text.contains("ALERT").then(|| signal.command("alarm", json!({"text": text})))
    }));
    b.effector(Webhook::new("alarm", "http://127.0.0.1:9000/alarm")?);
    b.reflex(events, triage);
    Server::new().port(3000).nerve(b).serve().await
}
```

`curl -i -X POST localhost:3000/events -d 'ALERT: smoke'` answers `204` (the
`x-furnace-nerve` header says what happened), and the webhook effector posts `{"text": "ALERT: smoke"}` to
whatever listens on port 9000 (retrying, then giving up, when nothing does).

The other common shape sends what comes in to the brain, which decides what to do (feature
`llm`):

```rust,no_run
use furnace::Server;
use furnace::llm::mock::MockEngine;
use furnace::nerve::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init();
    // Any LlmEngine: `Scheduled` over your Burn model, furnace-candle's `Gguf`, or this mock,
    // which always answers the same. The answer is parsed as commands:
    // {"effector": "...", "args": {...}}, arrays of them, or tool calls.
    let engine = MockEngine::new("mock", &[r#"{"effector": "log", "args": {"line": "seen"}}"#]);

    let mut b = Fabric::builder();
    let events = b.receptor(Http::new("events", "/events").text());
    let brain = b.neuron(Llm::new("brain", engine));
    b.deliberative(events, brain);
    let page = b.effector(Webhook::new("page", "http://127.0.0.1:9000/page")?);
    let log = b.effector(FileAppend::new("log", "journal.log"));
    // What the brain is told it may call
    b.action(page, "Page the operator.", json!({"text": "why"}));
    b.action(log, "Write a line to the journal.", json!({"line": "text"}));
    Server::new().port(3000).nerve(b).serve().await
}
```

What the builder does for you:

- `receptor`, `neuron` and `effector` add a neuron and return its id. `effector` takes an
  act (`Webhook`, `Process`, `FileAppend`, `Respond`, `ChannelOut`, `SseOut`, or your own
  `Act`) with the default local loop, or an `Effector` you configured
  (`Effector::new(act).refractory(..)`).
- `motor` adds a motor neuron with its program ([the motor side](#the-motor-side)), and
  `action` offers a motor neuron or an effector to brains, with a description and the shape of
  its arguments: an LLM brain is told about these and nothing else, and its commands reach
  nothing else. `spine(brain)` and `nerve(name, ..)` build the skeleton between the brain and
  the body ([spine and nerves](#the-skeleton-spine-and-nerves)).
- A command reaches the effector or motor neuron it names from any neuron, with or without a
  pathway to it: `reflex` and `deliberative` are for signals that are not commands yet.
- Neurons have no mechanisms until `b.gate(id, Mechanisms::NONE.threshold(..).habituation(..))`
  gives them some, so nothing is dropped that you did not ask to drop. Habituation attenuates
  a repeat; the threshold is where it dies.
- `reflex` pathways weigh 1 and `deliberative` ones 2, so a signal goes to the brain whenever
  its deadline allows it and takes the reflex arc when it does not. `.weight(w)` and
  `.summation(Summation::default().max_batch(16))` change a pathway as it is added.
- A reflex closure returns a `Firing`, a `Signal` (fire it), an `Option<Signal>` (fire it or
  stay quiet) or a `Vec<Signal>`.

Both together, with habituation at the receptor and a deadline that keeps alerts off the
brain's pathway:

```rust,no_run
use std::time::Duration;

use furnace::Server;
use furnace::llm::mock::MockEngine;
use furnace::nerve::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init();
    // The brain: a mock that takes 100 ms and always asks for the same two commands. Put a
    // real LlmEngine here.
    let answer = r#"[{"effector": "webhook", "args": {"action": "investigate"}},
                     {"effector": "log", "args": {"line": "brain: looked"}}]"#;
    let engine = MockEngine::new("mock", &[answer]).first_delay(Duration::from_millis(100));

    let mut b = Fabric::builder();

    // A receptor: POST /events. The feature is a 64-bit sketch of the body; repeats habituate
    // here and never reach anything.
    let events = b.receptor(Http::new("events", "/events").text());
    b.gate(
        events,
        Mechanisms::NONE.threshold(0.6).habituation(Habituation::new(Duration::from_secs(30))),
    );

    // A reflex: in process, no blocking. Alerts become commands with a 2 ms deadline, which
    // the brain's pathway (hundreds of milliseconds) can never meet; the rest pass on.
    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
        if signal.payload.as_text().is_some_and(|t| t.contains("ALERT")) {
            vec![
                signal
                    .command("alarm", json!({"text": signal.payload.as_text()}))
                    .with_urgency(Urgency::within(Duration::from_millis(2))),
                signal.command("respond", json!({"status": 200, "body": "alarm"})),
            ]
        } else {
            vec![signal]
        }
    }));
    let brain = b.neuron(Llm::new("brain", engine));

    // Effectors, by name, each with its local loop
    let alarm = b.effector(
        Effector::new(Webhook::new("alarm", "http://127.0.0.1:9000/alarm")?)
            .refractory(Duration::from_millis(100)),
    );
    let webhook = b.effector(Webhook::new("webhook", "http://127.0.0.1:9000/hook")?);
    let log = b.effector(FileAppend::new("log", "journal.log"));
    b.effector(Respond::new("respond"));
    // What the brain may call; the alarm and the response are the reflex's
    b.action(webhook, "Ask the operations service to look.", json!({"action": "investigate"}));
    b.action(log, "Write a line to the journal.", json!({"line": "text"}));

    // Pathways: the brain's weighs more, so it wins whenever the deadline allows it. The
    // reflex pathway to the alarm is optional (the command would find it); declared, it has
    // its own measured latency in `GET /nerve`.
    b.reflex(events, triage);
    b.reflex(triage, alarm);
    b.deliberative(triage, brain);

    Server::new().port(3000).nerve(b).serve().await
}
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
| `ChannelIn` | your code: `ChannelSender::emit` runs the chain inline, `send` queues to the receptor's thread | whatever you give the signal | the rule |
| `Timer` | one signal per period | the tick, or one fixed feature | the rule |
| `Lines` | stdin, a TCP connection, or a TCP port it listens on | sketch of the line | the rule |
| `SseIn` | an upstream `text/event-stream`, with reconnects and `Last-Event-ID` | sketch of the event's data | the rule |

An `UrgencyRule` is a fixed deadline (`within`) or a deadline derived from the stream's
measured rate (`before_next(1.0)`: act before the next sample). Every receptor measures its
rate and stamps it on the signal.

### Effectors (`furnace::nerve::effector`)

An `Effector` wraps an `Act` and supplies the loop: `Retry` (3 attempts from 100 ms backoff by
default), `Breaker` (open after 5 consecutive failures for 10 s, then one trial), `ActRateLimit`
(a token bucket that paces or drops) and `refractory`. Acts that finish at once run inline on
the reflex arc; the rest run on the efferent runtime. Implement `Act` for your own device: an
act returns `Ok(value)`, what it measured or was answered (`Action::value`, or `Action::ok()`
for nothing to say), which goes back to the motor program that sent the command; the webhook
returns its target's JSON answer.

| Act | Does |
|---|---|
| `Respond` | answers the request that started the chain: `{"status", "content_type", "body"}` or the signal's payload; with nobody waiting any more it does nothing, which is not a failure |
| `Webhook` | `POST url` with the command's args as JSON, or the payload as it is; 5xx and 429 are retried; plain `http://`, no redirects; the timeout covers the whole call, response body included |
| `Process` | runs a local command with fixed arguments plus `{"args": [...], "stdin": "..."}`; writing stdin counts against the timeout; the args reach argv as they are, with no shell |
| `FileAppend` | appends `args.line`, or a JSON line with the time, the signal id and the args, on a thread of its own, in order |
| `ChannelOut` | hands `Delivery { command, signal }` to the code holding the receiver |
| `SseOut` | pushes `{"signal", "effector", "args"}` to subscribers of `GET /nerve/effectors/{name}/events` |

### Brains (`furnace::nerve::brain`)

`Llm::new(name, engine)` takes any `LlmEngine` (feature `llm`): the actions declared with
`b.action` go into the system prompt, one a line with their description and arguments
(`{actions}` in your own prompt), the signal's payload is the user message (a question from a
motor neuron, or an action's outcome, are signals too), and the answer is parsed as commands
wherever JSON appears in it: `{"name": ..., "args": ...}`, `{"effector": "...", "args": ...}`,
`{"name": ..., "arguments": ...}`, `{"tool_calls": [...]}`, or arrays of those. The
signals of one summation window are generated at once, so an engine with a scheduler batches
them. `ModelNeuron::brain(name, model, input, output)` does the same for any `Model`, with
your functions from signal to input tensors and from outputs to commands, one forward pass per
window; `ModelNeuron::reflex` is the same model inline on a reflex arc (a small classifier).

### Metrics

In `/metrics` (`nerve` in the JSON, `furnace_nerve_*` in Prometheus) and `GET /nerve`:

- per neuron: `received`, `fired`, `quiet_by_threshold`, `quiet_by_habituation`,
  `quiet_by_refractory`, `quiet_not_interested` (signals the neuron itself was not interested
  in; `reason="not_interested"` in Prometheus), `computed` (firings from blocking work, every
  firing of a brain); effectors add `acted`, `failed`, `retries`,
  `dropped_open`, `dropped_rate`, `quiet_refractory`, commands in flight, the breaker's state
  and the latency to the world; brains add generations, commands and the generation latency;
  motor neurons add `started`, `done`, `failed`, `timed_out`, `cancelled` and `running`
- per pathway: signals, the latency from signal-in to signal-out (`latency_ms`), how many
  signals it was too slow for, and the current summation window
- `brain_fraction`: signals that reached a brain over signals received
- per reflex arc (receptor → effector, never queued): stimulus to effector call (`latency_ms`)

Every latency is a histogram of every measurement, as the server's own request latencies are,
with buckets from 1 µs to 10 s. In the JSON a latency is `{"count", "p50", "p95", "p99",
"mean", "max"}` in milliseconds, the percentiles estimated from the buckets (`latency_ms` on
pathways, arcs and effectors, `generation_latency_ms` on an LLM brain, `forward_latency_ms` on
a model). Pathway selection does not use these: it reads the p99 of each pathway's most recent
128 measurements, so it follows a change within a few signals.

The JSON has all of it. Prometheus has the fabric's counters and `brain_fraction`, each
neuron's counters (`kind` and `reason` labels), each pathway's signals (`kind="reflex"` or
`"deliberative"`), the pathway and arc latencies as histograms in seconds
(`furnace_nerve_pathway_latency_seconds_bucket`, `_sum`, `_count`, and the same for
`arc`), and each effector's counters, `inflight` and `circuit_open`. What a
brain or a motor neuron adds, and an effector's latency, are in the JSON only. `GET /nerve`
also shows each action's description and arguments, as a brain is told them.

## The demo

The short one, [`furnace-demo`](../furnace-demo/src/main.rs), needs no model file and no port:

```bash
cargo install --locked --git https://github.com/Gilfeather/furnace furnace-demo && furnace-demo
```

Two receptors (`logs` with no deadline, `smoke` with 5 ms) feed one junction with two
pathways: a reflex that acts in place, and a mock LLM brain that takes about 150 ms. A signal
with no deadline takes the heavier pathway, the brain; once the brain's p99 has been measured,
a 5 ms signal no longer fits there and takes the reflex. Each line shows where a signal went,
why, and how long after the stimulus the actuator acted; repeats show up as quiet at the
receptor. It ends with what the fabric measured, the same numbers `GET /nerve` serves.
Releases after 0.13 also publish it as a container image:
`docker run --rm -t ghcr.io/gilfeather/furnace-demo`.

The full one runs a server, webhooks and a Burn classifier:

```bash
cargo run --release -p furnace-burn --example nerve
```

[`furnace-burn/examples/nerve`](../furnace-burn/examples/nerve/main.rs) runs, in one process, a
tiny webhook receiver
that prints the commands it gets, the server with the fabric above (plus an in-process channel
of 28×28 frames through a small Burn classifier whose verdict is the frame's brightness), and
a demo stream: 200 text events and 200 frames, half of them repeats, 5% urgent. Repeated
events die at the receptor; urgent ones reach the alarm webhook over the reflex arc in a few
microseconds; novel ones go to a mock brain that takes 100 ms to answer with three commands
(to the `webhook`, `command` and `log` effectors).

What it prints, in order (the counts depend on timing, so none are given here):

- `furnace nerve demo: server on :3000, webhook receiver on :3001`
- a `[webhook] ...` line for every call the receiver gets: the alarms, and the brain's
  `{"action": "investigate"}`
- `Text events by status (204 quiet or reflex, 202 to the brain): {...}`, the status codes of
  the 200 `POST /events` requests
- `Frames: N quiet at the receptor, N to the brain (... ms for all)`
- once the brain's queue has drained, a summary from `/metrics`:
  `Signals received N, reached the brain N: brain fraction 0.xx`, then one line per neuron
  (receptors, reflexes, the brain, the effectors) with its kind, how many times it fired and
  how many signals died of habituation there
- `Pathway and reflex-arc latencies: GET /nerve`, and the line count of the journal file the
  `log` effector wrote (in the system's temporary directory)

About half of the signals reach the brain: the repeats are quiet at the receptors and the
urgent ones go to the alarm instead. The brain answers 16 at a time every 100 ms, so its queue
is what takes the time: the brain is the bottleneck, and the fabric keeps everything else away
from it. `GET /nerve` has the measured latencies of each pathway and reflex arc. `--serve`
keeps the server up to poke at with `curl` (`POST /events`, `GET /nerve`).

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

- Not a workflow engine or a DAG DSL: pathways and neurons are code, and a motor neuron's
  program is an async closure; nothing is persisted, and a program that the process loses is
  lost.
- Not an agent framework: the brain names actions, and motor neurons carry them out with the
  loops a body has (retry, adjust, ask a small question, give up and report); planning belongs
  to the brain.
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
