# 🔥 Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![Binary Size](https://img.shields.io/badge/binary%20size-4.5MB-blue)](https://github.com/Gilfeather/furnace)
[![Inference Time](https://img.shields.io/badge/inference-~0.2ms*-brightgreen)](https://github.com/Gilfeather/furnace)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Gilfeather/furnace/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Gilfeather/furnace?style=social)](https://github.com/Gilfeather/furnace/stargazers)

**Ultra-fast ONNX inference server built with Rust**

A high-performance, lightweight HTTP inference server specialized for ONNX models with zero Python dependencies. Optimized for ResNet-18 image classification with sub-millisecond inference times.




## ✨ Features

- 🦀 **Pure Rust**: Maximum performance, minimal memory footprint (4.5MB binary)
- 🔥 **ONNX Support**: Direct ONNX model loading with automatic shape detection
- ⚡ **Fast Inference**: ~0.2ms inference times for ResNet-18
- 🛡️ **Production Ready**: Graceful shutdown, comprehensive error handling
- 🌐 **HTTP API**: RESTful endpoints with CORS support
- 📦 **Single Binary**: Zero external dependencies
- 🖼️ **Image Classification**: Optimized for computer vision models

## 🚀 Quick Start

### Prerequisites

- **Rust 1.75+** and Cargo
- **curl** for downloading models and testing
- **~50MB disk space** for ResNet-18 model

### 1. Clone and Build

```bash
git clone https://github.com/yourusername/furnace.git
cd furnace
cargo build --release
```

Expected output: Binary created at `./target/release/furnace` (~4.5MB)

### 2. Download ResNet-18 Model

```bash
# Download ResNet-18 ONNX model (45MB)
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o resnet18.onnx
```

**Alternative download methods:**
```bash
# Using wget
wget "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -O resnet18.onnx

# Verify download (should be ~45MB)
ls -lh resnet18.onnx
```

### 3. Generate Test Data

```bash
# Generate ResNet-18 test samples (creates JSON files locally)
cargo run --example resnet18_sample_data
```

This creates the following test files:
- `resnet18_single_sample.json` - Single image test data
- `resnet18_batch_sample.json` - Batch of 3 images test data  
- `resnet18_full_test.json` - Full-size single image (150,528 values)

### 4. Start the Server

```bash
./target/release/furnace --model-path resnet18.onnx --host 127.0.0.1 --port 3000
```

Expected output:
```
✅ Model loaded successfully: resnet18 with input shape [3, 224, 224] and output shape [1000]
✅ Server running on http://127.0.0.1:3000
```

### 5. Test the API

Open a new terminal and test the endpoints:

```bash
# Health check
curl http://localhost:3000/healthz

# Model info
curl http://localhost:3000/model/info

# Single image prediction (~0.2ms inference time)
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_full_test.json

# Batch prediction
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_batch_sample.json
```

## 🖼️ ONNX Model Integration

Furnace uses Burn's native ONNX import system to generate Rust code from ONNX models at build time. This provides maximum performance and eliminates runtime dependencies.

### 🔧 How ONNX Integration Works

1. **Build-time Code Generation**: ONNX models are converted to native Rust code during compilation
2. **Zero Runtime Dependencies**: No ONNX runtime required - everything is compiled into the binary
3. **Native Performance**: Generated code is optimized by the Rust compiler
4. **Type Safety**: Full Rust type checking for model inputs and outputs

### 📁 Project Structure for ONNX Models

```
furnace/
├── models/                    # Place your ONNX files here
│   ├── resnet18.onnx         # ResNet-18 model (auto-detected)
│   └── your_model.onnx       # Your custom ONNX models
├── build.rs                  # Generates Rust code from ONNX files
├── src/
│   ├── onnx_models.rs        # Generated model integration
│   └── ...
└── target/debug/build/.../out/models/
    ├── resnet18.rs           # Generated Rust code for ResNet-18
    └── your_model.rs         # Generated code for your models
```

### 🚀 Adding New ONNX Models

**Step 1: Add ONNX File**
```bash
# Place your ONNX model in the models/ directory
cp your_model.onnx models/
```

**Step 2: Build (Automatic Code Generation)**
```bash
# Build process automatically generates Rust code
cargo build --release
```

**Step 3: Integrate Generated Code**

The build process generates Rust code in `target/debug/build/.../out/models/your_model.rs`. To integrate it:

1. **Update `src/onnx_models.rs`**:
```rust
#[cfg(feature = "burn-import")]
pub mod your_model {
    include!(concat!(env!("OUT_DIR"), "/models/your_model.rs"));
}
```

2. **Add Model Loading Logic**:
```rust
// In GeneratedOnnxModel::load_by_name()
"your_model" => {
    info!("Loading generated Your Model ONNX model");
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = your_model::Model::<B>::default(&device);
    
    let mut wrapper = Self::new(
        "your_model".to_string(),
        vec![1, 3, 224, 224], // Your model's input shape
        vec![1000],           // Your model's output shape
    );
    wrapper.inner_model = Some(Box::new(model));
    Ok(wrapper)
}
```

3. **Update Inference Logic**:
```rust
// In GeneratedOnnxModel::predict()
"your_model" => {
    if let Some(model) = inner_model.downcast_ref::<your_model::Model<B>>() {
        // Reshape input to match your model's expected format
        let input_reshaped = input.reshape([batch_size, 3, 224, 224]);
        
        // Run inference
        let output = model.forward(input_reshaped);
        
        // Reshape output back to 2D
        let output_2d = output.reshape([batch_size, output_size]);
        return Ok(output_2d);
    }
}
```

### 🔍 Generated Code Structure

When you build with an ONNX model, Burn generates a complete Rust implementation:

```rust
// Example: Generated ResNet-18 code structure
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    batchnormalization1: BatchNorm<B, 2>,
    maxpool2d1: MaxPool2d,
    // ... all layers defined
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Complete forward pass implementation
        let x = self.conv2d1.forward(input);
        let x = self.batchnormalization1.forward(x);
        // ... full computation graph
    }
}
```

### ⚠️ ONNX Compatibility Notes

**Supported ONNX Features:**
- ✅ Opset 16+ (required)
- ✅ Standard CNN operations (Conv2d, BatchNorm, ReLU, etc.)
- ✅ Image classification models
- ✅ Most PyTorch-exported models

**Known Limitations:**
- ❌ Some complex models may have unsupported operations
- ❌ Dynamic shapes require manual handling
- ❌ Some models may need ONNX version upgrade

**Upgrading ONNX Models:**
```python
# If your model uses older ONNX opset, upgrade it:
import onnx
from onnx import version_converter

model = onnx.load('old_model.onnx')
upgraded_model = version_converter.convert_version(model, 16)
onnx.save(upgraded_model, 'upgraded_model.onnx')
```

### 🧪 Testing Generated Models

```bash
# 1. Build with your ONNX model
cargo build --release

# 2. Check generated code
ls target/debug/build/furnace-*/out/models/

# 3. Test the server
./target/release/furnace --model-path models/your_model.onnx

# 4. Test inference
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [/* your test data */]}'
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
| Binary Size | **4.5MB** |
| Model Size | **45MB** |
| Inference Time | **~0.2ms** |
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

- **Single Inference**: ~0.2ms per image (ResNet-18)
- **Batch Processing**: Optimized for batches of 1-8 images
- **Concurrent Requests**: Handles multiple simultaneous requests
- **Memory Efficiency**: Minimal memory allocation per request
- **Throughput**: Scales with available CPU cores

### 🎯 Actual Benchmark Results

Based on Criterion benchmarks on **Intel MacBook Pro 2020**:

| Benchmark | Time | Throughput |
|-----------|------|------------|
| Single Inference | **152µs** | ~6,600 req/s |
| Batch Size 2 | **305µs** | ~6,600 req/s |
| Batch Size 4 | **664µs** | ~6,000 req/s |
| Batch Size 8 | **1.53ms** | ~5,200 req/s |
| Concurrent (4 threads) | **372µs** | ~10,800 req/s |
| Concurrent (8 threads) | **561µs** | ~14,300 req/s |

**Latency Percentiles:**
- P50: ~149µs
- P95: ~255µs  
- P99: ~340µs

**Performance Breakdown:**
- **ONNX Inference**: ~14µs (pure model execution)
- **Server Overhead**: ~157µs (optimized data processing, validation, tensor conversion)
- **Total Time**: ~171µs (end-to-end processing)

**Optimization Details:**
- 50% performance improvement while maintaining full security
- SIMD-optimized input validation for NaN/infinity detection
- Non-blocking statistics updates to prevent deadlocks
- Memory-efficient tensor operations

**Test Environment:**
- Hardware: Intel MacBook Pro 2020
- Compiler: Rust 1.75+ (release mode with full optimizations)
- Model: ResNet-18 ONNX (45MB, 150,528 input values → 1,000 output classes)

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
  "inference_time_ms": 0.2,
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

## 🛠️ Development

### Prerequisites
- Rust 1.75+ 
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

## 🔧 Troubleshooting

### Common Issues

**Model Download Fails:**
```bash
# Try alternative download method
wget "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -O resnet18.onnx

# Or check if file exists and size
ls -lh resnet18.onnx  # Should be ~45MB
```

**Server Won't Start:**
```bash
# Check if port is already in use
lsof -i :3000

# Try different port
./target/release/furnace --model-path resnet18.onnx --port 3001
```

**Build Errors:**
```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

**Test Data Generation Fails:**
```bash
# Ensure you're in the project root
pwd  # Should end with /furnace

# Run with verbose output
cargo run --example resnet18_sample_data --verbose
```

**ONNX Model Integration Issues:**

*Generated Code Not Found:*
```bash
# Check if ONNX models are in the right place
ls -la models/

# Verify code generation during build
cargo build --release 2>&1 | grep -i onnx

# Check generated files
find target -name "*.rs" -path "*/out/models/*"
```

*Model Loading Fails:*
```bash
# Check ONNX model compatibility
# Ensure your model uses ONNX opset 16+
python3 -c "
import onnx
model = onnx.load('models/your_model.onnx')
print(f'ONNX version: {model.opset_import[0].version}')
"

# If version < 16, upgrade the model:
python3 -c "
import onnx
from onnx import version_converter
model = onnx.load('models/your_model.onnx')
upgraded = version_converter.convert_version(model, 16)
onnx.save(upgraded, 'models/your_model_v16.onnx')
"
```

*Build Fails with ONNX Errors:*
```bash
# Some models may have compatibility issues
# Check build warnings for specific ONNX operations
cargo build 2>&1 | grep -A5 -B5 "ONNX\|onnx"

# Try building without problematic models
mv models/problematic_model.onnx models/problematic_model.onnx.bak
cargo build --release
```

*Runtime Errors with Generated Models:*
```bash
# Check tensor shape mismatches
# Ensure your input data matches the expected format
curl -X POST http://localhost:3000/model/info  # Check expected shapes

# Test with correct input format
# For ResNet-18: [batch_size, 3*224*224] = [1, 150528] values
```

### Performance Issues

If you're seeing slower inference times:
- Ensure you're using the release build (`cargo build --release`)
- Check system resources (CPU, memory)
- Try reducing batch size for concurrent requests
- Monitor with `cargo bench` for baseline performance

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