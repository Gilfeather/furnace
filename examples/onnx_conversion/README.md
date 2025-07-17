# ONNX Model Conversion Example

## 🎯 Purpose

This example demonstrates how to convert existing PyTorch/TensorFlow models to Burn's .mpk format via ONNX, enabling you to use production-ready models with Furnace.

## 🔄 Conversion Flow

```
PyTorch/TensorFlow → ONNX → Burn (.mpk) → Furnace
```

## 📋 Prerequisites

```bash
# Python dependencies
pip install torch torchvision onnx onnxruntime

# Rust dependencies (burn-import)
cargo install burn-import
```

## 🚀 Step-by-Step Guide

### Step 1: Export PyTorch Model to ONNX

Create a PyTorch model and export it:

```python
# pytorch_to_onnx.py
import torch
import torch.nn as nn
import torchvision.models as models

# Example 1: Simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and export model
model = SimpleCNN()
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 32, 32)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "simple_cnn.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ Model exported to simple_cnn.onnx")
```

### Step 2: Convert ONNX to Burn

```bash
# Convert ONNX to Burn format
burn-import --input simple_cnn.onnx --output simple_cnn_burn

# This creates:
# - simple_cnn_burn.mpk (model weights)
# - simple_cnn_burn.json (metadata)
```

### Step 3: Test with Furnace

```bash
# Start Furnace with converted model
./furnace --model-path simple_cnn_burn.mpk --port 3000

# Test inference
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [/* 3072 values for 32x32x3 image */]}'
```

## 🏭 Production Examples

### ResNet-18 from torchvision

```python
# resnet_to_onnx.py
import torch
import torchvision.models as models

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

```bash
# Convert to Burn
burn-import --input resnet18.onnx --output resnet18_burn

# Use with Furnace
./furnace --model-path resnet18_burn.mpk
```

### BERT from Hugging Face

```python
# bert_to_onnx.py
from transformers import BertModel, BertTokenizer
import torch

model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Prepare dummy input
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length")

# Export to ONNX
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    "bert_base.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state']
)
```

## ⚠️ Known Limitations

### ONNX → Burn Conversion
- **Not all ONNX operators supported** by burn-import
- **Complex models may fail** during conversion
- **Custom layers** might not be supported
- **Dynamic shapes** can be problematic

### Workarounds
1. **Simplify model architecture** before export
2. **Use static shapes** instead of dynamic
3. **Test conversion** with smaller models first
4. **Check burn-import documentation** for supported operators

## 🧪 Testing Your Converted Model

Create a test script to validate the conversion:

```python
# test_conversion.py
import onnxruntime as ort
import numpy as np

# Test ONNX model
onnx_session = ort.InferenceSession("simple_cnn.onnx")
test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
onnx_output = onnx_session.run(None, {"input": test_input})

print("ONNX output shape:", onnx_output[0].shape)
print("ONNX output sample:", onnx_output[0][0][:5])

# Compare with Furnace output
# (Use curl or HTTP client to test Furnace)
```

## 📚 Resources

- **[burn-import Documentation](https://github.com/tracel-ai/burn/tree/main/crates/burn-import)**
- **[ONNX Model Zoo](https://github.com/onnx/models)**: Pre-trained ONNX models
- **[PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)**
- **[TensorFlow to ONNX](https://github.com/onnx/tensorflow-onnx)**

## 🎯 Next Steps

1. **Start with simple models** (MLP, basic CNN)
2. **Test conversion pipeline** thoroughly
3. **Validate outputs** match original model
4. **Scale to production models** gradually
5. **Report issues** to burn-import if conversion fails