# furnace-burn

Serve [Burn](https://github.com/tracel-ai/burn) models with
[Furnace](https://github.com/Gilfeather/furnace). `furnace-server` is the server (the KServe /
Triton protocol over HTTP and gRPC, batching, reloads, metrics, TLS, the LLM routes and the
nervous system) and runs no model itself; this crate runs Burn models in it:

- `from_fn`: a `forward` function from one float tensor to one, served on `/predict` and
  `/v2/models/{name}/infer` with dynamic batching.
- `BurnModel` and `custom`: a model of named FP32 / INT64 / INT32 tensors.
- `HostTensorExt`: the server's tensors to Burn's and back.
- `weights::load_burnpack`: Burnpack weights, memory-mapped.
- `BackendKind`: the CPU backend, or `wgpu`, `metal`, `cuda` with those features.
- Feature `onnx`: the ONNX files in `models/` next to this crate's manifest (or in
  `$FURNACE_MODELS_DIR`), compiled to Burn at build time, as the `furnace` command serves them.

```toml
[dependencies]
furnace-server = "0.14"
furnace-burn = "0.14"
burn = "0.22"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,ignore
let model = furnace_burn::from_fn("digits", BackendKind::Cpu, &[784], &[10],
    move |images: Tensor<2>| net.forward(images))?;
furnace::Server::new().port(3000).model(model).serve().await
```

The crate documentation and the [repository's README](https://github.com/Gilfeather/furnace#readme)
have complete examples. `vendor/` holds a patched copy of Burn's CPU backend with faster
convolutions, used by this repository's builds only (see [vendor/README.md](vendor/README.md)).

MIT.
