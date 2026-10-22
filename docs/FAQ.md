# FAQ

- [Isn't the nervous system just a cache?](#isnt-the-nervous-system-just-a-cache)
- [How is it different from Ray Serve, Triton or BentoML?](#how-is-it-different-from-ray-serve-triton-or-bentoml)
- [How is it different from Erlang, actors or a message queue?](#how-is-it-different-from-erlang-actors-or-a-message-queue)
- [Do I need an LLM?](#do-i-need-an-llm)
- [Why pure Rust, and why so few dependencies?](#why-pure-rust-and-why-so-few-dependencies)
- [Can I use the server without the nervous system?](#can-i-use-the-server-without-the-nervous-system)
- [Is it production ready?](#is-it-production-ready)

## Isn't the nervous system just a cache?

No. The only cache-like part is **habituation** in a gate: a feature seen again shortly after
is attenuated, and once it falls below the gate's threshold the signal stops there. Even that
differs from a cache: a cache answers a repeat with the stored result, habituation answers it
with nothing, and the attenuation fades with time, so the same input gets through again later.

The rest is not about remembering answers at all:

- **Routing at every junction.** Each junction has pathways with measured latencies. A signal
  takes the heaviest pathway whose p99 fits its remaining deadline: the brain when there is
  time, a reflex when there is not. The measurements move as the system runs, so when the
  brain slows down, urgent signals stop going there without anyone reconfiguring anything.
- **Acting on the world.** Effectors (webhooks, commands, files, channels, responses) carry out
  commands with their own retry, circuit breaker, rate limit and refractory period.

A cache in front of a model saves repeated work on the same input. The nervous system decides,
for every input, which work to do at all and who acts on the result.

## How is it different from Ray Serve, Triton or BentoML?

They are mature model servers with large communities; Furnace is young (see the
[limits](../README.md#limits-and-what-is-not-done-yet)). The differences in kind:

| | Furnace | Ray Serve | Triton | BentoML |
|---|---|---|---|---|
| Language you extend it in | Rust | Python | C++ / Python backends | Python |
| Shape | an axum router in your service, or a standalone binary | deployments on a Ray cluster | a standalone server | a service framework and packaging |
| Scale-out | one process; scale with replicas (Kubernetes, KServe) | built in (the Ray cluster) | replicas | replicas, BentoCloud |
| Model engines | Burn, candle, or anything behind `TensorModel` / `LlmEngine` | any Python | TensorRT, ONNX Runtime, PyTorch, Python and more | any Python |
| GPU | Burn's backends build, barely measured | yes | its main strength | yes |
| Composition | the nervous system: junctions choose reflex or brain per signal by deadline and measured latency | deployment graphs, written by you | ensembles, business logic scripting | services calling services |

What Furnace speaks is shared: the Open Inference Protocol (KServe / Triton v2) over HTTP and
gRPC, and the OpenAI API for LLMs, so clients written for those servers work against it.

Pick Triton for heavy GPU serving with many engines, Ray Serve or BentoML for Python models and
Python teams. Furnace is for a Rust service that wants inference inside it, without a Python
runtime or a C++ toolchain, and for systems where sensors, small fast models, a slow brain and
actuators must be wired together with latency in mind.

## How is it different from Erlang, actors or a message queue?

**Actors** (Erlang processes, Akka, actix) give every unit a mailbox: each message is queued
and handled later, which buys isolation, supervision and distribution. In Furnace a reflex
pathway is a direct function call on the firing thread, with no mailbox and no scheduler, which
is why a reflex arc takes microseconds; only deliberative pathways have a queue, and the queue
sums a window of signals into one firing of the brain. Furnace has no supervision trees and no
distribution: a fabric is one process. You could build the same behavior on actors, at the
cost of a mailbox per hop.

**Message queues** (Kafka, NATS, RabbitMQ) are durable, shared between machines and
replayable. Furnace's queues are none of that: in memory, bounded, dropped when full, gone
when the process stops. They are for choosing work under a deadline, not for keeping it. The
two fit together: a queue's consumer can feed a receptor (HTTP, SSE, lines on stdin, a
channel), and an effector can publish to a queue through a webhook or a command.

## Do I need an LLM?

No. The `llm` feature is optional, and nothing in the nervous system needs it:

- A **reflex** is any code: a closure (`FnNeuron`), a small model (`ModelNeuron` over any
  `Model`), or your own `Neuron`.
- A **brain** is whatever is slow and expensive: an LLM (`Llm` over any `LlmEngine`), a large
  vision model (`ModelNeuron::brain`), a closure that calls a remote server, or nothing at all.
- The server without the nervous system serves tensor models and needs no LLM either.

The demo uses a mock LLM only because it needs no download.

## Why pure Rust, and why so few dependencies?

- **Build with `cargo` alone.** No C or C++ compiler, no CMake, no `protoc`, no Python: gRPC
  uses a pure-Rust protobuf compiler, the tokenizer a pure-Rust regex engine, TLS a
  RustCrypto-based provider. Cross-compiling to musl or aarch64 works the same way.
- **One small binary to deploy**, with a distroless container image and no runtime to install.
- **A supply chain you can read.** `furnace-server` with no features depends on 45 crates
  (`cargo tree`); every dependency is checked by `cargo deny` in CI. Fewer crates mean fewer
  things that can break, be compromised or slow the build.
- **Memory safety through the whole stack**, the TLS primitives included.

The costs are real: the TLS provider is Furnace's own and unaudited, and Burn's CPU kernels
are not always as fast as hand-tuned C++ ones ([docs/PERFORMANCE.md](PERFORMANCE.md) has the
comparisons).

## Can I use the server without the nervous system?

Yes. `Server::new().model(..)` serves models over HTTP (KServe / Triton v2 and a simpler
`/predict`) and gRPC, with dynamic batching, and `Server::router()` mounts it in your axum app.
The nervous system is always compiled in, because the server's own routes run on it, but you
never have to build a fabric yourself.

## Is it production ready?

Not proven yet: one author, no long production record, an unaudited TLS provider and an API
that may change before 1.0. The [limits](../README.md#limits-and-what-is-not-done-yet) list
them, and [ROADMAP-1.0.md](ROADMAP-1.0.md) says what has to happen before 1.0. If you try it,
an issue about what broke or a line in [ADOPTERS.md](../ADOPTERS.md) helps more than anything.
