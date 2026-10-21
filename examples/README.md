# Furnace Examples

From the simplest to the most involved:

| Step | Example | What it shows |
|---|---|---|
| 1 | [`resnet18_sample_data`](resnet18_sample_data.rs) | Request bodies for the `furnace` command serving the built-in ResNet-18: no Rust of your own |
| 2 | [`serve_burn_model`](../furnace-burn/examples/serve_burn_model/main.rs) | Your own Burn models from Rust: one with `furnace_burn::from_fn`, one with integer inputs through `BurnModel` |
| 3 | Axum, in the [main README](../README.md#embed-it-in-your-axum-app) | The same API as a router in your own axum app, next to your routes (`Server::router()`) |
| 4 | [`nerve`](../furnace-burn/examples/nerve/main.rs) | The nervous system end to end: receptors, reflexes, a brain and effectors ([docs/NERVE.md](../docs/NERVE.md)) |
| 5 | [`serve_gguf`](../furnace-candle/examples/serve_gguf.rs) | An LLM from a GGUF file, with the OpenAI API ([docs/LLM.md](../docs/LLM.md)) |

## 1. The `furnace` command with ResNet-18

```bash
# furnace-burn/models/resnet18.onnx must exist at build time (see the Quick start in the main README)
cargo build --release
./target/release/furnace --model-name resnet18

cargo run --example resnet18_sample_data      # writes resnet18_single_sample.json and resnet18_batch_sample.json
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_single_sample.json
```

## 2. Your own Burn models

```bash
cargo run --release --example serve_burn_model
curl localhost:3000/v2/models/sentiment        # its inputs and outputs
```

The comment at the top of
[`serve_burn_model/main.rs`](../furnace-burn/examples/serve_burn_model/main.rs) has the
requests to send, and how to load trained weights (`--digits-weights digits.bpk`). Save
weights from your training code with `burn-store`:

```rust,no_run
use burn::module::Module;
use burn_store::{BurnpackStore, ModuleSnapshot};

fn save(model: &impl Module) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_file("my_model.bpk");
    model.save_into(&mut store)?;
    Ok(())
}
```

## 3. Inside your axum app

No example program: the snippet in the [main README](../README.md#embed-it-in-your-axum-app)
is the whole program, with its `Cargo.toml`.

## 4. The nervous system

```bash
cargo run --release --example nerve
cargo run --release --example nerve -- --serve   # keep serving, for curl
```

[docs/NERVE.md](../docs/NERVE.md#the-demo) explains what it prints.

## 5. An LLM from a GGUF file

```bash
cargo run --release -p furnace-candle --example serve_gguf -- qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The GGUF file is not downloaded for you; [docs/LLM.md](../docs/LLM.md#a-gguf-file-with-candle)
says where to get it and how to call the server.

## Other programs here

The Burn ones (`serve_burn_model`, `nerve`, `basic_mnist_create`) are furnace-burn's, in
[`furnace-burn/examples`](../furnace-burn/examples); the others only call a running server.

| Example | What it does |
|---|---|
| `basic_mnist_create` | Writes the sample MLP format (`examples/basic_mnist/model.{bpk,json}`) that `furnace --model-path` serves |
| `basic_mnist_test` | Calls a running server with MNIST-sized inputs (health, info, single, batch) |
| `custom_data_test` | Sends synthetic digit-like images and random data to a running MNIST-sized model |
| `http_bench` | Measures `POST /predict` latency against a running ResNet-18 server |

```bash
cargo run --example basic_mnist_create
./target/release/furnace --model-path examples/basic_mnist/model
cargo run --example basic_mnist_test
```

Generated model files and request bodies are gitignored.
