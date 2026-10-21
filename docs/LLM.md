# Serving LLMs

With the `llm` feature, Furnace serves LLMs with the OpenAI API, next to the other models of the
same server: one process can answer `POST /v1/chat/completions` for a chat model and
`POST /v2/models/{model}/infer` for a Burn classifier, behind the same API keys, rate limits,
concurrency limit, TLS and metrics.

Furnace is the server side, not the engine. It owns the OpenAI routes and their errors, SSE
streaming, cancellation when a client disconnects, admission and backpressure, and the metrics
(time to first token, inter-token latency, tokens per second). The model runs behind a trait,
so its speed is the engine's: Furnace adds the server layer around it.

Everything is pure Rust: the tokenizer (`tokenizers` with fancy-regex, without Oniguruma or
esaxx), the chat templates (`minijinja`) and the engine adapters. CI fails if a C or C++ crate
enters the dependency tree.

LLMs are served from the library only: the `furnace` command, the release binaries and the
Docker image do not serve them. Your program builds a `Server` with `.llm(engine)`:

```toml
[dependencies]
furnace-server = { version = "0.13", features = ["llm"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tracing-subscriber = "0.3"
```

For GGUF files through candle, add `furnace-candle`. It is not on crates.io yet (see below),
so take both crates from the repository, which keeps them the same version of
`furnace-server`:

```toml
[dependencies]
furnace-server = { git = "https://github.com/Gilfeather/furnace", features = ["llm"] }
furnace-candle = { git = "https://github.com/Gilfeather/furnace" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tracing-subscriber = "0.3"
```

This page follows `main`, which is released as 0.13; until 0.13 is on crates.io, use the
`git` line for `furnace-server` in either case.

## A GGUF file, with candle

The [`furnace-candle`](../furnace-candle) crate serves quantized GGUF files of the llama family
(Llama, SmolLM, Mistral and others with the `llama` architecture) and Qwen2 / Qwen2.5 on the
CPU, through [candle](https://github.com/huggingface/candle). The tokenizer and the chat
template come from the GGUF file.

```rust,no_run
use furnace::Server;
use furnace_candle::Gguf;

#[tokio::main]
async fn main() -> Result<(), furnace::llm::BoxError> {
    tracing_subscriber::fmt::init(); // Furnace logs through `tracing`
    let engine = Gguf::new("qwen2.5-0.5b-instruct-q4_k_m.gguf").load()?;
    Server::new().port(8080).llm(engine).serve().await?;
    Ok(())
}
```

The same as an example, with a file of your choice:

```bash
cargo run --release -p furnace-candle --example serve_gguf -- qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Nothing is downloaded for you; for this one, `qwen2.5-0.5b-instruct-q4_k_m.gguf` from
[Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
(491 MB). The model's name, what clients put in `"model"`, is the file name without `.gguf`
(`qwen2.5-0.5b-instruct-q4_k_m` here); `Gguf::name` sets another. Other options:
`max_context` (4096 tokens by default, since the KV cache grows with it), `max_sequences` (4),
`tokenizer` (a `tokenizer.json` to use instead of the GGUF file's), and `scheduler` (worker
threads, queue length).

`furnace-candle` is not on crates.io yet: the published candle-core 0.11.0 builds Oniguruma
(C) for its GGUF tokenizer. Until a candle release drops it, `furnace-candle/Cargo.toml` takes
candle from git, pinned to one revision of a fork: candle's main branch (which uses
fancy-regex) plus a pending upstream fix that moves to zip 9 for security advisories.

## Your own model: `StepModel`

A model that can prefill a prompt and decode one token at a time implements `StepModel`:

```rust,no_run
use furnace::Server;
use furnace::llm::{BoxError, Logits, Scheduled, StepModel, TextPipeline};

#[derive(Debug)]
struct MyLlm;                           // the weights, shared by every request

struct MyCache;                         // KV cache and position of one request

impl MyLlm {
    fn load() -> Result<Self, BoxError> { todo!() }
}

impl StepModel for MyLlm {
    type Sequence = MyCache;

    fn name(&self) -> &str { "my-llm" }
    fn max_context(&self) -> usize { 2048 }
    fn max_sequences(&self) -> usize { 8 }     // what the KV memory allows

    fn prefill(&self, prompt: &[u32]) -> Result<(MyCache, Logits), BoxError> { todo!() }
    fn decode(&self, cache: &mut MyCache, token: u32) -> Result<Logits, BoxError> { todo!() }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let pipeline = TextPipeline::from_files("tokenizer.json", Some("tokenizer_config.json".as_ref()))?;
    Server::new().llm(Scheduled::new(MyLlm::load()?, pipeline)).serve().await?;
    Ok(())
}
```

`Logits` is the next-token logits over the vocabulary, on the host (`Logits(Vec<f32>)`);
`BoxError` is `Box<dyn std::error::Error + Send + Sync>`; `TextPipeline` is the tokenizer and
the chat template. `max_sequences` defaults to 1. Their documentation, and the rest of
`furnace::llm`, is on [docs.rs](https://docs.rs/furnace-server/latest/furnace/llm/).

`Scheduled` is Furnace's scheduler. It runs the model on its own worker threads and does
everything else: the chat template, tokenization, sampling (temperature, top-p, top-k, seed,
presence and frequency penalties) on the host logits, incremental detokenization, stop strings,
`max_tokens` and EOS. Concurrent requests are interleaved token by token, so a long answer
never holds the model for the others. Between two decode steps it prefills waiting prompts up
to `SchedulerConfig::max_prefill_tokens` (512 by default; a longer prompt starts alone), so a
burst of arrivals starts together while the running streams wait for a bounded amount of
prefill.
A model whose forward pass can take several sequences of different lengths overrides
`decode_batch`, and that alone gives continuous batching.

The crate documentation of `furnace::llm` has a complete example: a Burn model served this way.

Engines that schedule themselves implement `LlmEngine` instead: a request goes in, a stream of
events (`Started`, `Text`, then `Finished` or `Error`) comes out, and dropping the stream
cancels the request. An engine that can refuse a request only once it has looked at it (say, it
tokenizes on its own threads and finds the prompt too long) sends `Error` in place of `Started`:
a stream's status waits for the first event, so that refusal is still a 400, not an error after
a 200. Furnace's scheduler then stays out of the way.

## The API

| Route | |
|---|---|
| `POST /v1/chat/completions` | Chat, through the model's chat template |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | The LLMs (with `max_model_len`), then the server's other models |

Any OpenAI client works with `base_url = "http://<host>:<port>/v1"`, streaming included; an
API key, if the server has them, goes in as the client's API key.

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen2.5-0.5b-instruct-q4_k_m", "stream": true,
  "stream_options": {"include_usage": true},
  "messages": [{"role": "user", "content": "Name three Rust web frameworks."}]
}'
```

`model` is the name the engine reports: for a GGUF file, its file name without `.gguf`; for
a `StepModel`, what its `name()` returns. `GET /v1/models` lists them. Without `model`, a
request goes to the first LLM added.

**Served:** `messages` (text content, or text parts), `prompt` (one string), `max_tokens` /
`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`,
`frequency_penalty`, `stop` (up to 4 strings), `stream`, `stream_options.include_usage`.
Unknown fields are ignored, as other servers do. **Refused with a 400:** `n` other than 1,
`logprobs`, `tools`, a `response_format` other than text, `echo`, `suffix`, image parts.

**Streaming** follows OpenAI's format: a first chunk with the role, content deltas of whole
characters, a chunk with `finish_reason` (`stop` or `length`), the usage chunk with
`choices: []` when `include_usage` is set (and `"usage": null` on every other chunk), then
`data: [DONE]`. An error after the stream started is sent as an error payload, followed by
`[DONE]`. Idle streams get a keep-alive comment every 15 s.

**Errors** have OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:

| Status | When |
|---|---|
| 400 `invalid_request_error` | A field furnace does not serve, an invalid value, a prompt beyond the context (`code: context_length_exceeded`) |
| 404 `invalid_request_error` | An unknown `model` (`code: model_not_found`) |
| 503 `server_error` | `--max-concurrent-requests` reached, or the model's queue is full (`code: server_busy`) |
| 500 `server_error` | The model failed |

API keys (401) and the rate limit (429) answer in furnace's usual error shape, as on every
route.

## Cancellation and backpressure

When a client disconnects, its stream is dropped, and the engine drops the request: furnace's
scheduler frees the sequence and its KV cache at its next step. A request still waiting in the
queue is skipped. A matched stop string ends the request the same way.

Every LLM request holds one of the server's `max_concurrent_requests` slots until its response
is complete, and gets a 503 when none is free. `Scheduled` also refuses requests beyond its
queue (`SchedulerConfig::max_waiting`, 256 by default), and never runs more sequences than the
model's `max_sequences`.

A model that panics fails its own sequences with an error, and the worker goes on, but only
where panics unwind. furnace's release profile sets `panic = "abort"`, so the `furnace` binary
and the examples stop instead; a library user on the default profile keeps a model that may
have been left half-updated.

`request_timeout` bounds the LLM routes too: a whole answer that takes longer is a 504
(`"code": "timeout"`), and a stream whose engine sends nothing for that long ends with an error
payload and `[DONE]`. Either way the request is cancelled.

## Metrics

`GET /metrics` has an `llm` entry per model in JSON, and these series in the Prometheus format,
labelled `model`. They are measured at the server edge, from the events of each request, so they
mean the same for every engine.

| Series | |
|---|---|
| `furnace_llm_time_to_first_token_seconds` | Request received to the first token sent |
| `furnace_llm_inter_token_latency_seconds` | Between two tokens of one stream |
| `furnace_llm_queue_seconds` | Request received to the engine starting on it |
| `furnace_llm_prompt_tokens_total`, `furnace_llm_generation_tokens_total` | Tokens; tokens/s is their `rate()` |
| `furnace_llm_requests_running`, `furnace_llm_requests_waiting` | Requests now |
| `furnace_llm_requests_finished_total{reason}` | `stop`, `length`, `cancelled`, `error` |
| `furnace_llm_requests_rejected_total` | 503s |
| `furnace_llm_prefill_seconds`, `furnace_llm_decode_step_seconds` | Model time per step (`Scheduled` only) |

With `Scheduled`, the end-to-end time minus the model's step times is furnace's own overhead.

## Server overhead, measured

[`benches/llm`](../benches/llm) streams the same mock answer from every server, so they differ
only in the server layer: 64 tokens of `"x"` per stream, one every 20 ms, as OpenAI chat chunks
over SSE. N clients send streaming requests back to back on keep-alive connections, and for
each token the client records its lateness against the schedule. The time to first token
(TTFT) is the first token's lateness, since the engine sends it at once.

- **furnace**: `Server::llm` with an engine that paces its own tokens (as an engine with its
  own scheduler does). This is furnace's HTTP layer, SSE, metrics and admission.
- **scheduled**: furnace's scheduler over a `StepModel` whose decode step takes 20 ms for every
  running sequence at once (a batched forward pass), through the text pipeline: sampling over
  257 tokens, detokenization and stop strings.
- **hand-written**: axum with `Sse` and the same paced engine behind a channel. One route,
  about as little as a server can do.
- **FastAPI**: FastAPI and uvicorn (one worker), with a `StreamingResponse` paced with
  `asyncio.sleep`, built like vLLM's API server.

AMD Ryzen 9 7940HS (16 threads), Windows 11, furnace 0.10.0, Python 3.14,
FastAPI 0.143, uvicorn 0.54. Every server process asks for 1 ms timers (`timeBeginPeriod`),
otherwise Windows would round every interval to 15.6 ms.

| Clients | | furnace | scheduled | hand-written | FastAPI |
|---|---|---|---|---|---|
| 1 | TTFT p50 / p99 | 0.20 / 0.22 ms | 0.34 / 0.35 ms | 0.21 / 0.25 ms | 0.90 / 1.46 ms |
| 16 | TTFT p50 / p99 | 0.23 / 0.44 ms | 20.5 / 21.2 ms | 0.30 / 0.55 ms | 3.49 / 4.30 ms |
| 64 | TTFT p50 / p99 | 0.35 / 1.02 ms | 20.6 / 82.1 ms | 0.48 / 0.80 ms | 8.90 / 15.3 ms |
| 256 | TTFT p50 / p99 | 0.68 / 3.96 ms | 22.1 / 288 ms | 0.76 / 2.72 ms | 63.1 / 74.1 ms |
| 256 | ITL p50 / p99 | 20.02 / 21.83 ms | 20.65 / 22.45 ms | 20.00 / 22.20 ms | 19.99 / 37.16 ms |
| 256 | Tokens/s | 12,965 | 11,783 | 12,968 | 12,280 |
| 256 | Server CPU per 1,000 tokens | 40 ms | 45 ms | 36 ms | 71 ms |
| | Peak memory (private) | 20.1 MB | 23.0 MB | 14.8 MB | 77.6 MB |

In this process, with no interval and no network (`furnace-llm-bench overhead`), a streamed
token costs 1.13 µs in furnace and 0.83 µs in the hand-written server, from the slope between
16- and 1,024-token streams. Furnace adds 0.3 µs per token, then, and nothing measurable per
request (about 5 µs, against the hand-written server's 10). The scheduler path costs 1.16 µs
per token and about 30 µs per request, chat template and tokenization included.

What the numbers say:

- **Furnace's HTTP layer matches the hand-written server** at every load, within the timers'
  millisecond. FastAPI's time to first token grows with the load, to 63 ms at 256 streams,
  and its inter-token p99 doubles there.
- **The scheduler costs one decode step of TTFT**, because a request waits for the step under
  way before its prefill (20 ms here, the mock's step). Under a burst it also waits for the
  prefill budget: at 256 streams, its p99 is about 15 steps of 17 prompts each. Its tokens
  arrive later and later, since each step starts when the previous one ends and the mock's
  sleep overshoots a little each time (about 0.3 ms per step), but ITL stays at the step time.
- The CPU times come from the process counters, in 15.6 ms quanta, so only the 256-client row
  is shown.

Reproduce:

```bash
cd benches/llm
cargo build --release
FURNACE_BENCH_PYTHON=/path/to/venv/python target/release/furnace-llm-bench
target/release/furnace-llm-bench overhead --tokens 1024 --streams 1000
```

The engine behind each server is a mock, so this measures the server layer only.

## On a real model: furnace + candle, llama.cpp, mistral.rs

`furnace-llm-bench real` serves one GGUF file with each server in turn and sends them the same
streaming chat requests: a story, greedy, 128 tokens. Prompt caches are off and every prompt
starts with its own number, so each request pays its prefill. Each server runs with its own
defaults for threads, and with as many sequences as clients.

- **furnace + candle**: furnace-candle 0.1, candle main (`70f72e9`).
- **llama.cpp**: `llama-server` b11732 (C++, the reference).
- **mistral.rs**: `mistralrs serve` 0.9.5, which also runs on candle.

Qwen2.5-0.5B-Instruct Q4_K_M, same machine as above:

| Server | Clients | Failed | TTFT p50 / p99 | ITL p50 / p99 | Tokens/s per stream | Tokens/s total |
|---|---|---|---|---|---|---|
| furnace + candle | 1 | 0 of 1 | 1.1 / 1.1 s | 39.5 / 50.3 ms | 25.1 | 20.8 |
| furnace + candle | 8 | 0 of 8 | 6.9 / 10.9 s | 335 / 490 ms | 2.7 | 18.9 |
| furnace + candle | 32 | 0 of 32 | 20.4 / 40.0 s | 1.3 / 8.7 s | 0.7 | 19.3 |
| llama.cpp | 1 | 0 of 1 | 0.08 / 0.08 s | 8.8 / 10.1 ms | 112.6 | 105.9 |
| llama.cpp | 8 | 0 of 8 | 0.61 / 0.61 s | 29.6 / 51.9 ms | 32.5 | 226.4 |
| llama.cpp | 32 | 0 of 32 | 2.5 / 2.5 s | 94.9 / 102 ms | 10.5 | 281.8 |
| mistral.rs | 1 | 0 of 1 | 1.1 / 1.1 s | 36.5 / 40.6 ms | 27.4 | 22.3 |
| mistral.rs | 8 | **6 of 8** | 11.2 s | 72.9 / 86.7 ms | 8.9 | 12.5 |
| mistral.rs | 32 | **30 of 32** | 38.4 s | 67.0 / 76.4 ms | 4.7 | 5.5 |

The same model in process with furnace-candle alone, with no HTTP and no scheduler: 44 prompt
tokens prefilled in 1.1 s, then 23.9 tokens/s.

What the numbers say:

- **On one stream, furnace + candle runs at the engine's speed**: 25.1 tokens/s through the
  server against 23.9 in process, and the same as mistral.rs (27.4), the other server built on
  candle. The server layer costs nothing visible next to a 40 ms token.
- **llama.cpp is 4.5 times faster per token and 14 times faster on the prompt.** Its CPU
  kernels are that much faster than candle's, and on many streams it also batches them in one
  forward pass (282 tokens/s at 32 streams). candle's quantized models decode one sequence at a
  time, so furnace's scheduler interleaves them and the total stays at the single-stream speed.
  That is engine work: a `StepModel` with a batched `decode_batch` gets continuous batching from
  furnace unchanged.
- **mistral.rs 0.9.5 fails under concurrency on the CPU.** When it batches several prompts in
  one step, the model fails with `input tensor is not contiguous`, and the streams end with a
  `model_error` payload: 6 of 8 and 30 of 32 failed. Furnace answered every request.
- **Furnace's prefill budget suits fast prefills, not this one.** With candle taking about a
  second per prompt, prefilling up to 512 tokens between two decode steps keeps the running
  streams waiting for several seconds (ITL p99 8.7 s at 32 streams). A budget in time rather
  than tokens, from the measured prefill speed, would bound that.

Reproduce, with the files and binaries of your choice:

```bash
target/release/furnace-llm-bench real --gguf qwen2.5-0.5b-instruct-q4_k_m.gguf     --llama-server path/to/llama-server --mistralrs path/to/mistralrs
```

## Not yet

Tool calling and JSON mode, `logprobs`, `n > 1`, `/v1/responses` and `/v1/messages`,
`/v1/embeddings`, prefix caching, and GPU builds of the candle adapter.
