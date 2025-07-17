# ONNX to Burn Conversion Guide

## 🎯 Purpose

Convert existing ONNX models to Burn's .mpk format for use with Furnace. This enables you to use production-ready models from ONNX Model Zoo, Hugging Face, or your own exported ONNX models.

## 🔄 Conversion Flow

```
ONNX Model → burn-import → Burn (.mpk) → Furnace
```

## 📋 Prerequisites

```bash
# Install burn-import tool
cargo install burn-import

# Optional: Install onnxruntime for testing
pip install onnxruntime
```

## 🚀 Quick Start

### Step 1: Get an ONNX Model

**From ONNX Model Zoo (Recommended):**
```bash
# MNIST CNN (good for testing)
curl -L https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx -o mnist-8.onnx

# ResNet-18 (ImageNet classification)
curl -L https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx -o resnet18.onnx

# MobileNet v2 (lightweight)
curl -L https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx -o mobilenetv2.onnx

# SqueezeNet (very lightweight)
curl -L https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx -o squeezenet.onnx
```

### Step 2: Convert ONNX to Burn

```bash
# Convert ONNX to Burn format
burn-import --input mnist-8.onnx --output mnist_burn

# This creates:
# - mnist_burn.mpk (model weights in MessagePack format)
# - mnist_burn.json (model metadata)
```

### Step 3: Use with Furnace

```bash
# Start Furnace with converted model
./furnace --model-path mnist_burn.mpk --port 3000

# Test the model
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [/* 784 values for 28x28 MNIST image */]}'
```

## 🧪 Complete Test Example

Use our automated test script:

```bash
# Run comprehensive conversion test
python examples/onnx_conversion/test_onnx_conversion.py
```

This script will:
1. Download ONNX models from Model Zoo
2. Test them with onnxruntime
3. Convert to Burn format using burn-import
4. Test with Furnace server
5. Provide detailed results and troubleshooting

## 📊 Supported Models

### ✅ **Working Models**
| Model | Size | Input Shape | Use Case |
|-------|------|-------------|----------|
| MNIST CNN | ~26KB | [1, 1, 28, 28] | Digit recognition |
| SqueezeNet | ~5MB | [1, 3, 224, 224] | Lightweight image classification |
| MobileNet v2 | ~14MB | [1, 3, 224, 224] | Mobile-friendly classification |

### ⚠️ **May Work (Test First)**
| Model | Size | Notes |
|-------|------|-------|
| ResNet-18 | ~45MB | Some operators may not be supported |
| ResNet-50 | ~98MB | Complex architecture, test carefully |
| DenseNet | ~32MB | Dense connections may cause issues |

### ❌ **Known Limitations**
- **Transformer models** (BERT, GPT) - Complex attention mechanisms
- **Object detection** (YOLO, SSD) - Post-processing operations
- **Models with custom operators** - burn-import has limited operator support
- **Dynamic shapes** - Static shapes work better

## 🔧 Troubleshooting

### Common Issues

**1. "Unsupported operator" error:**
```bash
# Check which operators are used
python -c "
import onnx
model = onnx.load('your_model.onnx')
ops = set(node.op_type for node in model.graph.node)
print('Operators used:', sorted(ops))
"
```

**2. Shape mismatch errors:**
- Ensure input shapes match exactly
- Use static shapes instead of dynamic
- Check batch dimension (usually 1 for inference)

**3. Conversion fails silently:**
- Check burn-import version: `burn-import --version`
- Try with simpler models first
- Check file permissions and disk space

### Testing Your Model

Before converting, test with onnxruntime:

```python
import onnxruntime as ort
import numpy as np

# Load and test ONNX model
session = ort.InferenceSession("your_model.onnx")
input_info = session.get_inputs()[0]
print(f"Input: {input_info.name} {input_info.shape}")

# Create test input
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_info.name: test_input})
print(f"Output shape: {output[0].shape}")
```

## 📚 Model Sources

### 🏭 **Production Models**
- **[ONNX Model Zoo](https://github.com/onnx/models)**: Official pre-trained models
- **[Hugging Face ONNX](https://huggingface.co/models?library=onnx)**: Community models
- **[ONNX Runtime Models](https://onnxruntime.ai/docs/get-started/with-python.html#model-zoo)**: Optimized models

### 🔄 **Model Conversion Tools**
- **PyTorch**: `torch.onnx.export()`
- **TensorFlow**: `tf2onnx` converter
- **Scikit-learn**: `skl2onnx` converter
- **Keras**: `keras2onnx` converter

## 🎯 Best Practices

### 1. **Start Simple**
- Begin with MNIST or SqueezeNet
- Test the conversion pipeline
- Gradually move to complex models

### 2. **Validate Outputs**
- Compare ONNX vs Burn outputs
- Use same test inputs
- Check numerical accuracy

### 3. **Optimize for Furnace**
- Use static input shapes
- Prefer smaller models for faster inference
- Test with realistic input data

### 4. **Production Deployment**
- Validate model accuracy thoroughly
- Test under load conditions
- Monitor inference performance
- Have fallback plans for unsupported models

## 🚀 Next Steps

1. **Try the automated test**: `python examples/onnx_conversion/test_onnx_conversion.py`
2. **Convert your own models**: Follow the 3-step process above
3. **Report issues**: Help improve burn-import by reporting conversion failures
4. **Contribute**: Share working model conversions with the community

## 📖 Additional Resources

- **[burn-import Documentation](https://github.com/tracel-ai/burn/tree/main/crates/burn-import)**
- **[Burn Framework Guide](https://burn.dev/book/)**
- **[ONNX Specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md)**
- **[Furnace Documentation](../../README.md)**