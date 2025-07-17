# Basic MNIST Example

This example demonstrates how to:
1. Create a simple MLP model for MNIST-like data
2. Save the model in .mpk format
3. Load and run inference with Furnace

## Quick Start

```bash
# 1. Create the sample model
cargo run --example basic_mnist_create

# 2. Run inference server
cargo run --bin furnace -- --model-path examples/basic_mnist/model.mpk --log-level debug

# 3. Test inference (in another terminal)
cargo run --example basic_mnist_test
```

## What this example does

- **Model**: Simple 3-layer MLP (784 → 128 → 10)
- **Input**: 28x28 flattened images (784 features)
- **Output**: 10-class probabilities
- **Format**: Burn's MessagePack (.mpk) format

## Files

- `create_model.rs` - Creates and saves the model
- `test_inference.rs` - Tests inference via HTTP API
- `README.md` - This file

The model and data files are generated at runtime and not tracked in git.