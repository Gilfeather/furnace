#!/usr/bin/env python3
"""
PyTorch to ONNX conversion examples for Furnace
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

def create_simple_cnn():
    """Create a simple CNN for CIFAR-10 classification"""
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
    
    return SimpleCNN()

def create_simple_mlp():
    """Create a simple MLP for demonstration"""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = x.view(-1, 784)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return SimpleMLP()

def export_simple_mlp():
    """Export simple MLP to ONNX"""
    print("🔥 Creating Simple MLP...")
    
    model = create_simple_mlp()
    model.eval()
    
    # Dummy input (28x28 image flattened)
    dummy_input = torch.randn(1, 784)
    
    # Export to ONNX
    output_path = "simple_mlp.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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
    
    print(f"✅ Simple MLP exported to {output_path}")
    print(f"   Input shape: [batch_size, 784]")
    print(f"   Output shape: [batch_size, 10]")
    return output_path

def export_simple_cnn():
    """Export simple CNN to ONNX"""
    print("🔥 Creating Simple CNN...")
    
    model = create_simple_cnn()
    model.eval()
    
    # Dummy input (CIFAR-10: 32x32x3)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Export to ONNX
    output_path = "simple_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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
    
    print(f"✅ Simple CNN exported to {output_path}")
    print(f"   Input shape: [batch_size, 3, 32, 32]")
    print(f"   Output shape: [batch_size, 10]")
    return output_path

def export_resnet18():
    """Export pre-trained ResNet-18 to ONNX"""
    print("🔥 Loading pre-trained ResNet-18...")
    
    try:
        model = models.resnet18(pretrained=True)
        model.eval()
        
        # Dummy input (ImageNet: 224x224x3)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        output_path = "resnet18.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
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
        
        print(f"✅ ResNet-18 exported to {output_path}")
        print(f"   Input shape: [batch_size, 3, 224, 224]")
        print(f"   Output shape: [batch_size, 1000] (ImageNet classes)")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to export ResNet-18: {e}")
        print("   This might be due to network issues downloading pre-trained weights")
        return None

def test_onnx_model(onnx_path):
    """Test the exported ONNX model"""
    try:
        import onnxruntime as ort
        
        print(f"🧪 Testing {onnx_path}...")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_shape = session.get_inputs()[0].shape
        print(f"   Input shape: {input_shape}")
        
        # Create test input
        if len(input_shape) == 2:  # MLP
            test_input = np.random.randn(1, input_shape[1]).astype(np.float32)
        elif len(input_shape) == 4:  # CNN
            test_input = np.random.randn(1, input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
        else:
            print(f"   Unsupported input shape: {input_shape}")
            return
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})
        
        print(f"   Output shape: {output[0].shape}")
        print(f"   Output sample: {output[0][0][:5]}")
        print("✅ ONNX model test successful!")
        
    except ImportError:
        print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")

def main():
    """Main function to export models"""
    print("🚀 PyTorch to ONNX Conversion Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("examples/onnx_conversion", exist_ok=True)
    os.chdir("examples/onnx_conversion")
    
    models_to_export = [
        ("Simple MLP (recommended for testing)", export_simple_mlp),
        ("Simple CNN", export_simple_cnn),
        ("ResNet-18 (requires internet)", export_resnet18),
    ]
    
    exported_models = []
    
    for name, export_func in models_to_export:
        print(f"\n📦 {name}")
        try:
            onnx_path = export_func()
            if onnx_path:
                exported_models.append(onnx_path)
                test_onnx_model(onnx_path)
        except Exception as e:
            print(f"❌ Failed to export {name}: {e}")
    
    print("\n🎉 Export Summary:")
    print("=" * 30)
    for model_path in exported_models:
        print(f"✅ {model_path}")
    
    if exported_models:
        print("\n🔄 Next Steps:")
        print("1. Install burn-import: cargo install burn-import")
        print("2. Convert to Burn format:")
        for model_path in exported_models:
            model_name = model_path.replace('.onnx', '')
            print(f"   burn-import --input {model_path} --output {model_name}_burn")
        print("3. Use with Furnace:")
        print("   ./furnace --model-path <model_name>_burn.mpk")
    else:
        print("\n❌ No models were successfully exported")

if __name__ == "__main__":
    main()