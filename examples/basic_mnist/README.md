# Basic MNIST Example

Creates a small, randomly initialized MLP for trying out Furnace's file-based model loading.

```bash
# 1. Create the model
cargo run --example basic_mnist_create

# 2. Serve it
cargo run --release --bin furnace -- --model-path examples/basic_mnist/model

# 3. Send requests
cargo run --example basic_mnist_test
```

## Model

- **Architecture**: 784 → 128 (ReLU) → 10
- **Input**: 784 floats (a flattened 28×28 image)
- **Output**: 10 logits
- **Weights**: random, so predictions are meaningless; the point is exercising the pipeline

## Files

- `model.bpk`: weights in Burn's Burnpack format (written with `burn-store`)
- `model.json`: metadata (name, layer sizes, creation time)

Both are generated and gitignored.
