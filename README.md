# 🔥 Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![Binary Size](https://img.shields.io/badge/binary%20size-2.3MB-blue)](https://github.com/Gilfeather/furnace)
[![Inference Time](https://img.shields.io/badge/inference-~0.5ms*-brightgreen)](https://github.com/Gilfeather/furnace)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Gilfeather/furnace/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Gilfeather/furnace?style=social)](https://github.com/Gilfeather/furnace/stargazers)

**Blazingly fast ML inference server powered by Rust and Burn framework**

A high-performance, lightweight HTTP inference server that serves machine learning models with zero Python dependencies. Built with Rust for maximum performance and the Burn framework for native ML operations.




## ✨ Features

- 🦀 **Pure Rust**: Maximum performance, minimal memory footprint (2.3MB binary)
- 🔥 **Burn Integration**: Native ML framework with optimized tensors
- ⚡ **Fast Inference**: Sub-millisecond inference times
- 🛡️ **Production Ready**: Graceful shutdown, comprehensive error handling
- 🌐 **HTTP API**: RESTful endpoints with CORS support
- 📦 **Single Binary**: Zero external dependencies

## 🚀 Quick Start

### Option A: Using Your Own .mpk Model (Recommended)

```bash
# 1. Download or build Furnace
curl -L https://github.com/Gilfeather/furnace/releases/latest/download/furnace-linux-x86_64 -o furnace
chmod +x furnace

# 2. Run with your Burn model
./furnace --model-path /path/to/your/model.mpk --port 3000
```

### Option B: Testing with Sample Model

```bash
# 1. Clone and build
git clone https://github.com/yourusername/furnace.git
cd furnace
cargo build --release

# 2. Create a sample model for testing
cargo run --example basic_mnist_create

# 3. Start the server
./target/release/furnace --model-path examples/basic_mnist/model.mpk --port 3000
```

### 3. Make Predictions

```bash
# Health check
curl http://localhost:3000/healthz

# Model info
curl http://localhost:3000/model/info

# Prediction (MNIST-like: 784 inputs → 10 outputs)
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": '$(python3 -c "print([0.1] * 784)")'}'
```

## 📦 Getting .mpk Model Files

Furnace uses Burn's MessagePack (.mpk) format. Here's how to obtain .mpk files:

### 🔥 From Burn Training Scripts
```rust
use burn::record::CompactRecorder;

// After training your model
let recorder = CompactRecorder::new();
model.save_file("my_model", &recorder)?; // Creates my_model.mpk
```

### 📥 From Model Sources
- **[Burn Examples](https://github.com/tracel-ai/burn/tree/main/examples)**: Official Burn examples with pre-trained models
- **[Burn Book](https://burn.dev/book/)**: Official documentation with model examples
- **[Hugging Face](https://huggingface.co/models?library=burn)**: Models converted to Burn format
- **[Burn Community](https://github.com/tracel-ai/burn/discussions)**: Community-shared models and discussions
- **Your Training**: Export from your Burn training scripts
- **Converted Models**: ONNX/PyTorch models converted to Burn

### 🛠️ Model Conversion
```bash
# Convert ONNX to Burn (example)
burn-import --input model.onnx --output model.mpk

# Convert PyTorch to Burn (via ONNX)
# 1. Export PyTorch to ONNX
# 2. Convert ONNX to Burn
```

### 🎯 Popular Burn Model Examples
- **[MNIST CNN](https://github.com/tracel-ai/burn/tree/main/examples/mnist)**: Convolutional neural network for digit recognition
- **[Text Classification](https://github.com/tracel-ai/burn/tree/main/examples/text-classification)**: BERT-like models for NLP tasks
- **[Image Classification](https://github.com/tracel-ai/burn/tree/main/examples/image-classification)**: ResNet and other vision models
- **[Custom Training](https://github.com/tracel-ai/burn/tree/main/examples/custom-training-loop)**: How to train and save your own models

### 🧪 For Testing/Development
Use our examples to create sample models:
```bash
cargo run --example basic_mnist_create  # Creates MNIST-like MLP
```

## 📊 Performance

⚠️ **Important Note**: Current benchmarks are based on a simple MLP model (784→128→10, ~0.5MB). 
Real-world model performance will vary significantly based on model size and complexity.

### Current Benchmarks (Simple MLP Model)
| Metric | Value |
|--------|-------|
| Binary Size | **2.3MB** |
| Model Size | **~0.5MB** |
| Inference Time | **~0.5ms** |
| Memory Usage | **<50MB** |
| Startup Time | **<100ms** |

### 🚧 Planned Benchmarks (Coming Soon)
| Model Type | Size | Inference Time | Status |
|------------|------|----------------|---------|
| ResNet-18 | ~45MB | TBD | 🔄 In Progress |
| BERT-base | ~110MB | TBD | 📋 Planned |
| YOLO v8n | ~6MB | TBD | 📋 Planned |

*Current tests: MNIST-like MLP on standard hardware. Production model benchmarks coming soon.*

## � API Enpdpoints

### `GET /healthz`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### `GET /model/info`
Model metadata and statistics
```json
{
  "model_info": {
    "name": "sample_mnist_model",
    "input_spec": {"shape": [784], "dtype": "float32"},
    "output_spec": {"shape": [10], "dtype": "float32"},
    "model_type": "burn",
    "backend": "ndarray"
  },
  "stats": {
    "inference_count": 42,
    "total_inference_time_ms": 126.5
  }
}
```

### `POST /predict`
Run inference on input data
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0.1, 0.2, ...]}'
```

Response:
```json
{
  "output": [-0.045, 0.066, 0.068, ...],
  "status": "success",
  "inference_time_ms": 3.0,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## �️ iDevelopment

### Prerequisites
- Rust 1.70+ 
- Cargo

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

### Create Custom Models
Implement the `BurnModel` trait in `src/burn_model.rs` to add support for your own model architectures.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │───▶│  Model Layer    │───▶│   API Layer     │
│                 │    │                 │    │                 │
│ - Argument      │    │ - Model Loading │    │ - HTTP Routes   │
│   Parsing       │    │ - Inference     │    │ - Request       │
│ - Validation    │    │ - Metadata      │    │   Handling      │
│ - Logging Setup │    │ - Error Handling│    │ - CORS          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Burn](https://github.com/tracel-ai/burn) - The native Rust ML framework
- [Axum](https://github.com/tokio-rs/axum) - Web framework for Rust
- [Tokio](https://github.com/tokio-rs/tokio) - Async runtime for Rust