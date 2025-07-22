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

### 2. Generate Test Data

```bash
# Generate ResNet-18 test samples (creates JSON files locally)
cargo run --example resnet18_sample_data
```

This creates the following test files:
- `resnet18_single_sample.json` - Single image test data
- `resnet18_batch_sample.json` - Batch of 3 images test data  
- `resnet18_full_test.json` - Full-size single image (150,528 values)

### 3. Build with ONNX Support

```bash
# Build with burn-import feature for ONNX model generation
cargo build --features burn-import --release
```

Expected build output:
```
Generating ONNX models following Burn documentation
Generating model: resnet18
✅ Model 'resnet18' generated successfully
   Compiling furnace v0.3.0
    Finished release [optimized] target(s)
```

### 4. Start the Server

```bash
# Start server with built-in ResNet-18 model
./target/release/furnace --model-name resnet18 --host 127.0.0.1 --port 3000
```

Expected output:
```
🔧 Logging initialized log_level=INFO is_production=false
🔥 Starting furnace inference server session_id=...
📋 Server configuration model_name=resnet18 server_host=127.0.0.1 server_port=3000
📦 Loading model model_name=resnet18
Loading built-in model: resnet18
Successfully loaded built-in model: resnet18 with backend: burn-resnet18
✅ Model loaded successfully input_shape=[1, 3, 224, 224] output_shape=[1000]
🚀 Starting HTTP server
🚦 Concurrency limit set to 100 requests  
✅ Server running on http://127.0.0.1:3000
```

### 5. Test the API

Open a new terminal and test the endpoints:

```bash
# Health check
curl http://localhost:3000/health
# Expected: {"status":"healthy","model_loaded":true,...}

# Model info  
curl http://localhost:3000/model/info
# Expected: {"model_info":{"name":"resnet18","input_spec":{"shape":[1,3,224,224]},...}}

# Single image prediction (~0.2ms inference time)
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_full_test.json
# Expected: {"output":[0.1,0.05,0.02,...],...}

# Batch prediction
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_batch_sample.json
# Expected: {"output":[[...],[...],[...]],"batch_size":3,...}
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

### 🚀 Adding Custom ONNX Models

**Furnace automatically detects and integrates ONNX models!** Just place them in the `models/` directory and rebuild.

#### Step 1: Add Your ONNX File
```bash
# Place your ONNX model in the models/ directory
cp your_model.onnx models/

# Verify file placement
ls -la models/
# Should show: resnet18.onnx, your_model.onnx, etc.
```

#### Step 2: Automatic Build Process
```bash
# Build with burn-import feature for ONNX processing
cargo build --features burn-import
```

**What happens during build:**
- 🔍 Auto-detects all `.onnx` files in `models/` directory
- 🦀 Converts each ONNX model to native Rust code
- ✅ Successfully generated models become available
- ❌ Failed models are skipped (with helpful error messages)

**Build Output Example:**
```
Generating ONNX models following Burn documentation
Generating model: resnet18
✅ Model 'resnet18' generated successfully
Generating model: your_model
❌ Failed to generate model 'your_model' - incompatible ONNX format
   This model will be skipped. Consider simplifying the ONNX file.
```

#### Step 3: Add Model to Code (Manual Integration)

For successfully generated models, add them to `src/models/mod.rs`:

**3a. Add Module Declaration:**
```rust
// Add your model module
#[cfg(all(feature = "burn-import", model_your_model))]
pub mod your_model {
    include!(concat!(env!("OUT_DIR"), "/models/your_model.rs"));
}

// Re-export the model
#[cfg(all(feature = "burn-import", model_your_model))]
pub use your_model::Model as YourModel;
```

**3b. Add to BuiltInModel enum:**
```rust
pub enum BuiltInModel {
    ResNet18,
    #[cfg(model_your_model)]
    YourModel,  // Add your model here
}
```

**3c. Add Model Loading Logic:**
```rust
impl BuiltInModel {
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "resnet18" => Ok(Self::ResNet18),
            #[cfg(model_your_model)]
            "yourmodel" => Ok(Self::YourModel),  // Add your model
            // ...
        }
    }

    pub fn create_model(&self) -> Result<Box<dyn BurnModel>> {
        match self {
            // ... existing models ...
            #[cfg(model_your_model)]
            Self::YourModel => {
                let model = YourModel::<Backend>::default();
                Ok(Box::new(SimpleYourModelWrapper {
                    model: Arc::new(Mutex::new(model)),
                    name: "yourmodel".to_string(),
                    input_shape: vec![1, 3, 224, 224],  // Adjust for your model
                    output_shape: vec![1000],            // Adjust for your model
                }))
            }
        }
    }
}
```

**3d. Create Model Wrapper:**
```rust
// Add wrapper struct for your model
#[cfg(all(feature = "burn-import", model_your_model))]
#[derive(Debug)]
pub struct SimpleYourModelWrapper {
    model: Arc<Mutex<YourModel<Backend>>>,
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

// Implement BurnModel trait
#[cfg(all(feature = "burn-import", model_your_model))]
impl BurnModel for SimpleYourModelWrapper {
    fn predict(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        // Validate input and reshape as needed for your model
        let model = self.model.lock().unwrap();
        let output = model.forward(input);  // Adjust based on your model's requirements
        Ok(output)
    }
    
    // Implement other required methods...
    fn get_name(&self) -> &str { &self.name }
    fn get_input_shape(&self) -> &[usize] { &self.input_shape }
    fn get_output_shape(&self) -> &[usize] { &self.output_shape }
    fn get_backend_info(&self) -> String { "burn-yourmodel".to_string() }
    // ...
}
```

#### Step 4: Build and Test
```bash
# Rebuild with your new model integration
cargo build --features burn-import

# List available models
cargo run --bin furnace --features burn-import -- --help

# Start server with your model
cargo run --bin furnace --features burn-import -- --model-name yourmodel --port 3000

# Test inference
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [/* your test data */]}'
```

### 🎯 Complete Example: Adding MobileNet

**Step-by-step example of adding MobileNet v2:**

```bash
# 1. Download MobileNet ONNX model
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" -o models/mobilenetv2.onnx

# 2. Build (automatic generation)
cargo build --features burn-import
# Expected output: ✅ Model 'mobilenetv2' generated successfully

# 3. Add cfg declaration to build.rs (if not already present)
# Add to build.rs: println!("cargo:rustc-check-cfg=cfg(model_mobilenetv2)");
```

**Add to src/models/mod.rs:**
```rust
// Module declaration
#[cfg(all(feature = "burn-import", model_mobilenetv2))]
pub mod mobilenetv2 {
    include!(concat!(env!("OUT_DIR"), "/models/mobilenetv2.rs"));
}

// Re-export
#[cfg(all(feature = "burn-import", model_mobilenetv2))]
pub use mobilenetv2::Model as MobileNetV2Model;

// Add to BuiltInModel enum
pub enum BuiltInModel {
    ResNet18,
    #[cfg(model_mobilenetv2)]
    MobileNetV2,
}

// Wrapper struct
#[cfg(all(feature = "burn-import", model_mobilenetv2))]
#[derive(Debug)]
pub struct SimpleMobileNetV2ModelWrapper {
    model: Arc<Mutex<MobileNetV2Model<Backend>>>,
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

// Add to from_name, create_model, etc...
```

**Test the new model:**
```bash
# Build and test
cargo build --features burn-import
cargo run --bin furnace --features burn-import -- --model-name mobilenetv2
```

### 📋 Best Practices for Custom Models

**✅ Recommended Workflow:**
1. **Test with smaller models first** (SqueezeNet, MobileNet)
2. **Verify ONNX compatibility** before integration
3. **Use descriptive model names** (lowercase, no spaces)
4. **Add comprehensive error handling** in your wrapper
5. **Test thoroughly** with your specific input data

**⚠️ Common Pitfalls:**
- Don't forget the `#[cfg(...)]` attributes
- Match model names exactly (case-sensitive in file paths)
- Ensure input/output shapes match your actual model
- Add cfg check declarations to build.rs for new models
- Test with both single and batch predictions

**📋 Model Integration Checklist:**
- [ ] ONNX file placed in `models/` directory
- [ ] Build succeeds with "✅ Model generated successfully"
- [ ] Added module declaration with proper cfg attributes
- [ ] Added to BuiltInModel enum
- [ ] Added to from_name() method
- [ ] Added to create_model() method
- [ ] Added wrapper struct and BurnModel implementation
- [ ] Added cfg check to build.rs
- [ ] Tested server startup
- [ ] Tested inference API

### 🔍 Debugging Model Integration

**Check if model was generated:**
```bash
# Look for generated Rust files
find target -name "*.rs" -path "*/out/models/*"

# Check build output for your model
cargo build --features burn-import 2>&1 | grep -i "your_model"
```

**Verify conditional compilation:**
```bash
# Check which models are enabled
cargo build --features burn-import -v 2>&1 | grep "model_"
```

**Test model loading:**
```bash
# Try to start server (will show available models if yours isn't found)
cargo run --bin furnace --features burn-import -- --model-name nonexistent
# Error message will list available models
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
- ❌ Models with broadcasting dimension conflicts
- ❌ Extremely large models (>2GB) may cause memory issues

**Troubleshooting Failed Model Generation:**

*1. ONNX Version Issues:*
```python
# Check ONNX opset version
import onnx
model = onnx.load('models/your_model.onnx')
print(f'ONNX opset version: {model.opset_import[0].version}')

# Upgrade to supported version (16+)
from onnx import version_converter
upgraded = version_converter.convert_version(model, 16)
onnx.save(upgraded, 'models/your_model_v16.onnx')
```

*2. Model Simplification:*
```bash
# Install ONNX simplifier
pip install onnx-simplifier

# Simplify complex models
python -c "
import onnx
from onnxsim import simplify
model = onnx.load('models/complex_model.onnx')
simplified, check = simplify(model)
onnx.save(simplified, 'models/simple_model.onnx')
"
```

*3. Shape Broadcasting Issues:*
```bash
# If you see "Invalid shape for broadcasting" errors:
# - Try model simplification first
# - Check if model has dynamic shapes
# - Consider using a different model architecture
# - Report issue with model details for potential fix
```

*4. Memory Issues:*
```bash
# For very large models:
export RUST_MIN_STACK=8388608  # Increase stack size
cargo build --features burn-import
```

### 🧪 Testing Generated Models

**Quick validation workflow:**
```bash
# 1. Build with burn-import feature
cargo build --features burn-import

# 2. Check generated code
find target -name "*.rs" -path "*/out/models/*"
# Should show: resnet18.rs, your_model.rs, etc.

# 3. Test available models
cargo run --bin furnace --features burn-import -- --help
# Will show available model names in help text

# 4. Test server startup
cargo run --bin furnace --features burn-import -- --model-name yourmodel --port 3000

# 5. Test model info endpoint
curl http://localhost:3000/model/info

# 6. Test inference
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [/* your test data matching model input shape */]}'
```

**Performance testing:**
```bash
# Generate test data for your model
cargo run --example create_test_data -- --model yourmodel

# Run benchmarks
cargo bench --features burn-import
```

## 🖼️ Supported Models

Furnace supports ONNX models with automatic shape detection. Currently optimized for image classification models.

### 🎯 Tested Models

| Model | Input Shape | Output Shape | Size | Status |
|-------|-------------|--------------|------|---------|
| **ResNet-18** | `[3, 224, 224]` | `[1000]` | 45MB | ✅ **Supported** |
| **MobileNet v2** | `[3, 224, 224]` | `[1000]` | 14MB | 🧪 **Compatible** |
| **SqueezeNet** | `[3, 224, 224]` | `[1000]` | 5MB | 🧪 **Compatible** |
| **GPT-NeoX** | `[1, 512]` | `[50257]` | 1.7MB | ❌ **Incompatible** |
| **Your Custom Model** | `[?, ?, ?]` | `[?]` | ?MB | 🔄 **Add with guide above** |

### 📥 Built-in Pre-trained Models

Furnace includes built-in models that are compiled during build time:

| Model | Status |
|-------|--------|
| **ResNet-18** | ✅ **Available** (45MB, built-in) |
| **MobileNet v2** | 🔄 **Add to models/** |
| **SqueezeNet** | 🔄 **Add to models/** |

To add additional models, place ONNX files in the `models/` directory and rebuild with `--features burn-import`.

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
# Generate test data (benchmarks use built-in model)
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

**Model Loading Fails:**
```bash
# Check available built-in models
./target/release/furnace --help

# Verify model was built
find target -name "resnet18.rs" -path "*/out/models/*"
```

**Server Won't Start:**
```bash
# Check if port is already in use
lsof -i :3000

# Try different port
./target/release/furnace --model-name resnet18 --port 3001
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