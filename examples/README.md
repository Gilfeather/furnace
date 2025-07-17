# Furnace Examples

This directory contains examples demonstrating how to use Furnace with different types of models and scenarios.

## 🚀 Quick Start

### If you already have a .mpk model file:
```bash
# Just run Furnace directly
./furnace --model-path /path/to/your/model.mpk
```

### If you need a sample model for testing:
```bash
# 1. Create a sample model
cargo run --example basic_mnist_create

# 2. Start the server
cargo run --bin furnace -- --model-path examples/basic_mnist/model.mpk

# 3. Test inference
cargo run --example basic_mnist_test
```

## 📦 Getting .mpk Model Files

Furnace uses Burn's MessagePack (.mpk) format. You can obtain .mpk files from:

### 🔥 **From Burn Training Scripts**
```rust
// In your Burn training code
use burn::record::CompactRecorder;

// After training your model
let recorder = CompactRecorder::new();
model.save_file("my_model", &recorder)?; // Creates my_model.mpk
```

### 📥 **From Model Repositories**
- Hugging Face models (if converted to Burn format)
- Burn model zoo repositories
- Community-shared Burn models
- Your own trained models

### 🛠️ **Converting from Other Formats**
- ONNX → Burn (using burn-import)
- PyTorch → Burn (using conversion tools)
- TensorFlow → Burn (via ONNX)

### 🧪 **For Testing/Development**
- Use our example creation scripts (see below)
- Generate synthetic models for testing

## Available Examples

### 🎯 Basic MNIST Example (`basic_mnist/`)

**Purpose**: Demonstrates basic Furnace usage with a simple MLP model

**What it creates**:
- Simple MLP model (784→128→10) for MNIST-like data
- Model saved in .mpk format
- Metadata file with model specifications

**When to use**: 
- ✅ Testing Furnace functionality
- ✅ Learning how Burn models work
- ✅ Development and debugging
- ❌ Production use (use your trained models instead)

## Example Structure

Each example follows this structure:
```
examples/
├── example_name/
│   ├── README.md          # Detailed instructions
│   ├── model.mpk          # Generated model file (gitignored)
│   ├── model.json         # Generated metadata (gitignored)
│   └── data/              # Generated data files (gitignored)
├── example_name_create.rs # Model creation script
├── example_name_test.rs   # Inference testing script
└── README.md              # This file
```

## Key Features Demonstrated

- ✅ **Model Creation**: How to create and save Burn models
- ✅ **Multiple Formats**: Support for .burn and .mpk formats
- ✅ **HTTP API**: Complete inference API testing
- ✅ **Error Handling**: Input validation and error responses
- ✅ **Performance**: Timing and throughput testing
- ✅ **Structured Logging**: Development and production logging
- ✅ **Configuration**: Different server configurations

## Development Notes

- **Model files** (.mpk, .burn) are generated at runtime and not tracked in git
- **Data files** are also generated and gitignored to keep the repository clean
- **Examples are self-contained** - each can run independently
- **Comprehensive testing** - each example includes validation and error testing

## Adding New Examples

To add a new example:

1. Create a new directory: `examples/your_example/`
2. Add a README.md with instructions
3. Create `your_example_create.rs` for model creation
4. Create `your_example_test.rs` for testing
5. Update this README.md

Make sure to:
- Add model files to .gitignore
- Include comprehensive error testing
- Document the model architecture and use case
- Provide clear usage instructions