# Toward 1.0

Furnace is 0.x: any minor release may break the API. This page says what 1.0 will freeze, what
is likely to change before then, and what has to be true first. It is the current plan, not a
promise of dates.

## Versions until 1.0

- A **minor** release (0.13 → 0.14) may break the Rust API, the command's flags or the HTTP
  API. Every break is listed in the [CHANGELOG](../CHANGELOG.md) with old → new, under
  "Migrating from 0.x".
- A **patch** release (0.13.0 → 0.13.1) only fixes bugs and never breaks anything.
- Renamed or removed items are not kept as deprecated aliases before 1.0: the CHANGELOG is the
  migration path, and the API stays small.
- The published crates (furnace-server, furnace-burn, furnace-cli and the parts amate, kuebiko
  and kamui) share one version and are released together.

## What 1.0 freezes

**The wire**, which already follows outside specifications and changes least:

- the Open Inference Protocol (KServe / Triton v2) over HTTP and gRPC, `/predict`, and the
  OpenAI chat and completions API;
- `/healthz`, `/readyz`, and the Prometheus metric names and labels on `/metrics`;
- the error bodies and status codes in [docs/API.md](API.md#errors).

The JSON of `/metrics` and `/nerve` may gain fields in minor releases; clients should ignore
fields they do not know.

**The Rust API of `furnace-server`**:

- `Server` and `ServerConfig`, `Model` and `TensorModel`, `HostTensor`, `Values`,
  `TensorSpec`, `IoSpec`, the error types;
- the nervous system's `prelude`: `Fabric` and its builder, `Signal`, `Payload`, `Feature`,
  `Urgency`, `Mechanisms`, `Habituation`, `Summation`, the `Neuron` trait and `Firing`, the
  receptors and effectors, and the metrics types;
- with `llm`, the `LlmEngine` and `StepModel` traits and the sampling parameters.

**The `furnace` command**: its flags and `furnace.toml` keys.

Not covered: items hidden from the documentation (`#[doc(hidden)]`, such as
`nerve::bench_wire`), and log message text.

## Likely to change before 1.0

- **Junctions across machines.** Today a fabric is one process, and a junction elsewhere is
  wired with a webhook and an HTTP receptor. A pathway that crosses the network, with its
  latency measured like any other, would add to the builder and may change `Link` and the
  pathway metrics.
- **HTTPS for outbound webhooks** (additive).
- **The `Neuron` trait and `Firing`**, as more neurons than closures, models and LLMs are
  written against them.
- **The request path of the server**, which is still slower than a bare axum handler (see the
  README's measurements); thinning it may change `ServerConfig`.
- **The LLM traits**, as engines other than candle are connected.

**furnace-burn** exposes Burn's types, and Burn is itself 0.x. furnace-burn follows Burn's
breaking releases and may stay 0.x after furnace-server reaches 1.0, released with a version of
its own from then on.

## Conditions for 1.0

1. **Outside use.** At least one team other than the author runs Furnace in production, and
   what they ran into has been fixed or written down ([ADOPTERS.md](../ADOPTERS.md)).
2. **A quiet API.** Two minor releases in a row without a breaking change to the frozen list
   above.
3. **Checked from then on.** `cargo semver-checks` runs in CI on every pull request, and every
   public item is documented.
4. **TLS verified.** The crypto provider passes published test vectors (Wycheproof) in CI, and
   the README states plainly whether it has had an outside review.
5. **A minimum Rust version policy.** Which compilers 1.x supports, and that raising the
   minimum is a minor release, not a major one.

After 1.0, breaking changes wait for 2.0, and anything removed is deprecated for at least one
minor release first.
