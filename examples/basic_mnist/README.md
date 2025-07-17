# Basic MNIST Example

## 🎯 Purpose

**This example is for testing and learning purposes only.**

If you already have a .mpk model file, you don't need this example - just run:
```bash
./furnace --model-path /path/to/your/model.mpk
```

## 🧪 When to Use This Example

- ✅ **Testing Furnace functionality**
- ✅ **Learning how Burn models work**
- ✅ **Development and debugging**
- ❌ **Production use** (use your trained models instead)

## 🚀 Quick Start

```bash
# 1. Create the sample model (only needed once)
cargo run --example basic_mnist_create

# 2. Run inference server
cargo run --bin furnace -- --model-path examples/basic_mnist/model.mpk --log-level debug

# 3. Test inference (in another terminal)
cargo run --example basic_mnist_test
```

## 📊 What This Example Creates

- **Model**: Simple 3-layer MLP (784 → 128 → 10)
- **Input**: 28x28 flattened images (784 features)
- **Output**: 10-class probabilities
- **Format**: Burn's MessagePack (.mpk) format
- **Size**: ~200KB (very small for testing)

## 📁 Generated Files

After running `basic_mnist_create`, you'll get:
- `examples/basic_mnist/model.mpk` - The actual model weights
- `examples/basic_mnist/model.json` - Model metadata

**Note**: These files are generated at runtime and not tracked in git.

## 🔄 Real-World Usage

For production use, replace the example model with your trained model:
```bash
# Instead of the example model
./furnace --model-path examples/basic_mnist/model.mpk

# Use your trained model
./furnace --model-path /path/to/your/trained_model.mpk
```