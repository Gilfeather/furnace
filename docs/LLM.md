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

## A GGUF file, with candle

The [`furnace-candle`](../furnace-candle) crate serves quantized GGUF files of the llama family
(Llama, SmolLM, Mistral and others with the `llama` architecture) and Qwen2 / Qwen2.5 on the
CPU, through [candle](https://github.com/huggingface/candle). The tokenizer and the chat
template come from the GGUF file.

```rust
use furnace::Server;
use furnace_candle::Gguf;

#[tokio::main]
async fn main() -> Result<(), furnace::llm::BoxError> {
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
(491 MB). Options: `Gguf::name`, `max_context` (4096 tokens by default, since the KV cache
grows with it), `max_sequences` (4), `tokenizer` (a `tokenizer.json` to use instead of the
GGUF file's), and `scheduler` (worker threads, queue length).

`furnace-candle` is not on crates.io yet: the published candle-core 0.11.0 builds Oniguruma
(C) for its GGUF tokenizer, so it uses candle's main branch from git until the next release.

## Your own model: `StepModel`

A model that can prefill a prompt and decode one token at a time implements `StepModel`:

```rust
impl StepModel for MyLlm {
    type Sequence = MyCache;            // KV cache and position of one request

    fn name(&self) -> &str { "my-llm" }
    fn max_context(&self) -> usize { 2048 }
    fn max_sequences(&self) -> usize { 8 }     // what the KV memory allows

    fn prefill(&self, prompt: &[u32]) -> Result<(MyCache, Logits), BoxError> { ... }
    fn decode(&self, cache: &mut MyCache, token: u32) -> Result<Logits, BoxError> { ... }
}

let pipeline = TextPipeline::from_files("tokenizer.json", Some("tokenizer_config.json".as_ref()))?;
Server::new().llm(Scheduled::new(MyLlm::load()?, pipeline)).serve().await?;
```

`Scheduled` is Furnace's scheduler. It runs the model on its own worker threads and does
everything else: the chat template, tokenization, sampling (temperature, top-p, top-k, seed,
presence and frequency penalties) on the host logits, incremental detokenization, stop strings,
`max_tokens` and EOS. Concurrent requests are interleaved token by token, so a long answer
never holds the model for the others, and at most one prefill runs between two decode steps.
A model whose forward pass can take several sequences of different lengths overrides
`decode_batch`, and that alone gives continuous batching.

The crate documentation of `furnace::llm` has a complete example: a Burn model served this way.

Engines that schedule themselves implement `LlmEngine` instead: a request goes in, a stream of
events (`Started`, `Text`, then `Finished` or `Error`) comes out, and dropping the stream
cancels the request. Furnace's scheduler then stays out of the way.

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

Without `model`, a request goes to the first LLM added.

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

## Not yet

Tool calling and JSON mode, `logprobs`, `n > 1`, `/v1/responses` and `/v1/messages`,
`/v1/embeddings`, prefix caching, and GPU builds of the candle adapter.
