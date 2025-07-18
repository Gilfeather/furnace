# 🔥 Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![Binary Size](https://img.shields.io/badge/binary%20size-2.3MB-blue)](https://github.com/Gilfeather/furnace)
[![Inference Time](https://img.shields.io/badge/inference-~4ms*-brightgreen)](https://github.com/Gilfeather/furnace)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Gilfeather/furnace/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Gilfeather/furnace?style=social)](https://github.com/Gilfeather/furnace/stargazers)

**Blazingly fast ML inference server powered by Rust and Burn framework**

A high-performance, lightweight HTTP inference server that serves machine learning models with zero Python dependencies. Built with Rust for maximum performance and supports ONNX models including ResNet-18 for image classification.




## ✨ Features

- 🦀 **Pure Rust**: Maximum performance, minimal memory footprint (2.3MB binary)
- 🔥 **ONNX Support**: Direct ONNX model loading with automatic shape detection
- ⚡ **Fast Inference**: ~4ms inference times for ResNet-18
- 🛡️ **Production Ready**: Graceful shutdown, comprehensive error handling
- 🌐 **HTTP API**: RESTful endpoints with CORS support
- 📦 **Single Binary**: Zero external dependencies
- 🖼️ **Image Classification**: Optimized for computer vision models

## 🚀 Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/yourusername/furnace.git
cd furnace
cargo build --release
```

### 2. Download ResNet-18 Model

```bash
# Download ResNet-18 ONNX model (45MB)
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o resnet18.onnx
```

### 3. Start the Server

```bash
./target/release/furnace --model-path resnet18.onnx --host 127.0.0.1 --port 3000
```

### 4. Generate Test Data

```bash
# Generate ResNet-18 test samples (creates JSON files locally)
cargo run --example resnet18_sample_data
```

This creates the following test files:
- `resnet18_single_sample.json` - Single image test data
- `resnet18_batch_sample.json` - Batch of 3 images test data  
- `resnet18_full_test.json` - Full-size single image (150,528 values)

### 5. Test the API

```bash
# Health check
curl http://localhost:3000/healthz

# Model info
curl http://localhost:3000/model/info

# Single image prediction
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_full_test.json

# Batch prediction
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_batch_sample.json
```

## 🖼️ Supported Models

Furnace supports ONNX models with automatic shape detection. Currently optimized for image classification models.

### 🎯 Tested Models

| Model | Input Shape | Output Shape | Size | Status |
|-------|-------------|--------------|------|---------|
| **ResNet-18** | `[3, 224, 224]` | `[1000]` | 45MB | ✅ **Supported** |
| **MobileNet v2** | `[3, 224, 224]` | `[1000]` | 14MB | 🧪 **Testing** |
| **SqueezeNet** | `[3, 224, 224]` | `[1000]` | 5MB | 🧪 **Testing** |

### 📥 Download Pre-trained Models

```bash
# ResNet-18 (ImageNet classification) - Recommended
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o resnet18.onnx

# MobileNet v2 (lightweight, mobile-friendly)
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" -o mobilenetv2.onnx

# SqueezeNet (very lightweight)
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx" -o squeezenet.onnx
```

### 🔧 Custom Models

To use your own ONNX models:

1. **Export your model to ONNX format**
2. **Ensure input shape compatibility** (currently optimized for image classification)
3. **Test with Furnace** using the same API endpoints

```python
# Example: Export PyTorch model to ONNX
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "my_model.onnx")
```

## 📊 Performance

### ResNet-18 Benchmarks
| Metric | Value |
|--------|-------|
| Binary Size | **2.3MB** |
| Model Size | **45MB** |
| Inference Time | **~4ms** |
| Memory Usage | **<200MB** |
| Startup Time | **<2s** |
| Input Size | **150,528 values** |
| Output Size | **1,000 classes** |

### 🚀 Benchmark Results

**Prerequisites:**
```bash
# 1. Download ResNet-18 model (if not already done)
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o resnet18.onnx

# 2. Generate test data (benchmarks use dynamic model detection)
cargo run --example resnet18_sample_data
```

**Run benchmarks:**
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench single_inference
cargo bench batch_inference
cargo bench latency_measurement
```

### 📈 Performance Characteristics

- **Single Inference**: ~4ms per image (ResNet-18)
- **Batch Processing**: Optimized for batches of 1-8 images
- **Concurrent Requests**: Handles multiple simultaneous requests
- **Memory Efficiency**: Minimal memory allocation per request
- **Throughput**: Scales with available CPU cores

## 🌐 API Endpoints

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
    "name": "resnet18",
    "input_spec": {"shape": [3, 224, 224], "dtype": "float32"},
    "output_spec": {"shape": [1000], "dtype": "float32"},
    "model_type": "burn",
    "backend": "onnx"
  },
  "stats": {
    "inference_count": 42,
    "total_inference_time_ms": 168.0,
    "average_inference_time_ms": 4.0
  }
}
```

### `POST /predict`
Run inference on input data

**Single Image:**
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_full_test.json
```

**Batch Images:**
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_batch_sample.json
```

**Response:**
```json
{
  "output": [0.1, 0.05, 0.02, ...], // 1000 ImageNet class probabilities
  "status": "success",
  "inference_time_ms": 4.0,
  "timestamp": "2024-01-01T12:00:00Z",
  "batch_size": 1
}
```

### 📝 Input Format

ResNet-18 expects normalized RGB image data:
- **Shape**: `[3, 224, 224]` (150,528 values)
- **Format**: Flattened array of float32 values
- **Range**: Typically 0.0 to 1.0 (normalized pixel values)
- **Order**: Channel-first (RGB channels, then height, then width)

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