# Furnace Examples

| Example | What it does |
|---|---|
| `resnet18_sample_data` | Writes `resnet18_single_sample.json` / `resnet18_batch_sample.json` request bodies for the built-in ResNet-18 |
| `basic_mnist_create` | Creates an MNIST-shaped MLP (784 → 128 → 10) as `examples/basic_mnist/model.{bpk,json}` |
| `basic_mnist_test` | Calls a running server with MNIST-sized inputs (health, info, single, batch) |
| `custom_data_test` | Sends synthetic digit-like images and random data to a running MNIST-sized model |
| `nerve` | The nervous system end to end (`--features nerve,llm`): a habituating receptor, a Burn reflex classifier, a mock LLM brain, webhook / command / journal effectors, with a webhook receiver in the same process. See [docs/NERVE.md](../docs/NERVE.md) |

## ResNet-18 (built-in)

```bash
# models/resnet18.onnx must exist at build time (see the main README)
cargo build --release
./target/release/furnace --model-name resnet18

cargo run --example resnet18_sample_data
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_single_sample.json
```

## File-based MLP

```bash
cargo run --example basic_mnist_create
./target/release/furnace --model-path examples/basic_mnist/model
cargo run --example basic_mnist_test
```

## Bringing your own model

- **ONNX**: put the file in `models/` and register it as a built-in model. See
  "Built-in models" in the main README.
- **Burn models you trained**: Furnace's file loader serves the MLP architecture in
  `src/burn_model.rs`. Save weights with `burn-store`:

  ```rust
  use burn_store::{BurnpackStore, ModuleSnapshot};

  let mut store = BurnpackStore::from_file("my_model.bpk");
  model.save_into(&mut store)?;
  ```

  Then write `my_model.json` with the metadata fields from `BurnModelMetadata`
  (`name`, `version`, `input_size`, `hidden_size`, `output_size`, `model_type`, `created_at`).

Generated model files and request bodies are gitignored.
