# 🔥 Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![Binary Size](https://img.shields.io/badge/binary%20size-2.3MB-blue)](https://github.com/Gilfeather/furnace)
[![Inference Time](https://img.shields.io/badge/inference-~0.5ms-brightgreen)](https://github.com/Gilfeather/furnace)
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

### 1. Clone and Build

```bash
git clone https://github.com/yourusername/furnace.git
cd furnace
cargo build --release
```

### 2. Create a Sample Model

```bash
cargo run --bin create_sample_model
```

This creates `sample_model.mpk` and `sample_model.json` files.

### 3. Start the Server

```bash
./target/release/furnace --model-path ./sample_model --port 3000
```

### 4. Make Predictions

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

## 📊 Performance

| Metric | Value |
|--------|-------|
| Binary Size | **2.3MB** |
| Inference Time | **~0.5ms** |
| Memory Usage | **<50MB** |
| Startup Time | **<100ms** |

*Tested with MNIST-like model (784→128→10) on standard hardware*

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

## 📋 Roadmap

### 🚀 Upcoming Features

#### High Priority
- 📦 **Batch Inference Endpoint** (`/batch`) - Process multiple inputs in a single request
- 🎯 **GPU Support** - Burn-wgpu backend for accelerated inference
- 🔒 **HTTPS Support** - TLS encryption for production deployments

#### Medium Priority  
- 🔄 **Hot Model Reload** - Update models without server restart
- 📊 **Enhanced Metrics** - Prometheus/OpenTelemetry integration
- 🐳 **Docker Optimization** - Multi-stage builds and smaller images
- 🔧 **Configuration Management** - YAML/TOML config file support

#### Future Enhancements
- 🌐 **Model Registry Integration** - HuggingFace Hub, MLflow support
- 🔀 **Load Balancing** - Multiple model instances
- 📈 **Auto-scaling** - Dynamic resource allocation
- 🧪 **A/B Testing** - Model version comparison

### 🤝 Get Involved

We welcome contributions! Check out our [good first issues](https://github.com/Gilfeather/furnace/labels/good%20first%20issue) to get started.

- 💡 **Feature Requests**: Open an issue with the `enhancement` label
- 🐛 **Bug Reports**: Use the `bug` label and provide reproduction steps  
- 📚 **Documentation**: Help improve our docs and examples
- 🧪 **Testing**: Add test cases and benchmarks

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