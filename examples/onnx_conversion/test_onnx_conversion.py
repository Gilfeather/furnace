#!/usr/bin/env python3
"""
Test script to verify ONNX models work before converting to Burn
"""

import numpy as np
import requests
import subprocess
import os
import sys

def download_onnx_model(url, filename):
    """Download ONNX model from URL"""
    print(f"📥 Downloading {filename}...")
    try:
        result = subprocess.run(['curl', '-L', url, '-o', filename], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Downloaded {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def test_onnx_model(onnx_path):
    """Test ONNX model with onnxruntime"""
    try:
        import onnxruntime as ort
        
        print(f"🧪 Testing {onnx_path}...")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"   Input: {input_info.name} {input_info.shape} ({input_info.type})")
        print(f"   Output: {output_info.name} {output_info.shape} ({output_info.type})")
        
        # Create test input
        input_shape = input_info.shape
        # Handle dynamic batch size
        if input_shape[0] == 'batch_size' or input_shape[0] is None:
            input_shape = [1] + list(input_shape[1:])
        
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        output = session.run([output_info.name], {input_info.name: test_input})
        
        print(f"   ✅ ONNX inference successful!")
        print(f"   Output shape: {output[0].shape}")
        print(f"   Output sample: {output[0].flatten()[:5]}")
        
        return True
        
    except ImportError:
        print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
        return False

def convert_to_burn(onnx_path, output_name):
    """Convert ONNX model to Burn format using burn-import"""
    print(f"🔄 Converting {onnx_path} to Burn format...")
    
    try:
        # Check if burn-import is installed
        result = subprocess.run(['burn-import', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ burn-import not found. Install with: cargo install burn-import")
            return False
        
        # Convert ONNX to Burn
        result = subprocess.run(['burn-import', '--input', onnx_path, '--output', output_name], 
                              capture_output=True, text=True, check=True)
        
        # Check if .mpk file was created
        mpk_file = f"{output_name}.mpk"
        if os.path.exists(mpk_file):
            print(f"✅ Conversion successful! Created {mpk_file}")
            file_size = os.path.getsize(mpk_file) / (1024 * 1024)  # MB
            print(f"   Model size: {file_size:.2f} MB")
            return True
        else:
            print(f"❌ Conversion failed: {mpk_file} not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ burn-import failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def test_with_furnace(mpk_path, test_input_shape):
    """Test the converted model with Furnace"""
    print(f"🔥 Testing {mpk_path} with Furnace...")
    
    # Check if furnace binary exists
    furnace_paths = ['./furnace', './target/debug/furnace', './target/release/furnace']
    furnace_bin = None
    
    for path in furnace_paths:
        if os.path.exists(path):
            furnace_bin = path
            break
    
    if not furnace_bin:
        print("⚠️  Furnace binary not found. Build with: cargo build --release")
        return False
    
    try:
        # Start Furnace server in background
        print(f"   Starting Furnace server with {mpk_path}...")
        furnace_process = subprocess.Popen([
            furnace_bin, '--model-path', mpk_path, '--port', '3001'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        import time
        time.sleep(2)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:3001/healthz', timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"   ✅ Health check: {health.get('status', 'unknown')}")
                print(f"   Model loaded: {health.get('model_loaded', False)}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"   ❌ Cannot connect to Furnace: {e}")
            return False
        
        # Test model info endpoint
        try:
            response = requests.get('http://localhost:3001/model/info', timeout=5)
            if response.status_code == 200:
                info = response.json()
                model_info = info.get('model_info', {})
                print(f"   Model name: {model_info.get('name', 'unknown')}")
                print(f"   Input shape: {model_info.get('input_spec', {}).get('shape', 'unknown')}")
                print(f"   Output shape: {model_info.get('output_spec', {}).get('shape', 'unknown')}")
            else:
                print(f"   ⚠️  Model info failed: {response.status_code}")
        except requests.RequestException as e:
            print(f"   ⚠️  Model info error: {e}")
        
        # Test inference (if we know the input shape)
        if test_input_shape:
            try:
                # Create test input
                total_size = np.prod(test_input_shape[1:])  # Skip batch dimension
                test_input = np.random.randn(total_size).tolist()
                
                response = requests.post('http://localhost:3001/predict', 
                                       json={'input': test_input}, 
                                       timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Inference successful!")
                    print(f"   Inference time: {result.get('inference_time_ms', 'unknown')}ms")
                    output = result.get('output', [])
                    if output:
                        print(f"   Output sample: {output[:5]}")
                else:
                    print(f"   ❌ Inference failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except requests.RequestException as e:
                print(f"   ❌ Inference error: {e}")
        
        return True
        
    finally:
        # Clean up: terminate Furnace process
        if 'furnace_process' in locals():
            furnace_process.terminate()
            furnace_process.wait()

def main():
    """Main function to test ONNX conversion pipeline"""
    print("🚀 ONNX to Burn Conversion Test")
    print("=" * 50)
    
    # Test models with their download URLs and expected input shapes
    test_models = [
        {
            'name': 'MNIST CNN',
            'url': 'https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx',
            'filename': 'mnist-8.onnx',
            'output_name': 'mnist_burn',
            'input_shape': [1, 1, 28, 28],  # MNIST: 1 channel, 28x28
            'description': 'Simple CNN for digit recognition (good for testing)'
        },
        {
            'name': 'SqueezeNet',
            'url': 'https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx',
            'filename': 'squeezenet.onnx',
            'output_name': 'squeezenet_burn',
            'input_shape': [1, 3, 224, 224],  # ImageNet: 3 channels, 224x224
            'description': 'Lightweight CNN for image classification'
        }
    ]
    
    successful_conversions = []
    
    for model in test_models:
        print(f"\n📦 Testing {model['name']}")
        print(f"   {model['description']}")
        print("-" * 40)
        
        # Step 1: Download ONNX model
        if not download_onnx_model(model['url'], model['filename']):
            continue
        
        # Step 2: Test ONNX model
        if not test_onnx_model(model['filename']):
            continue
        
        # Step 3: Convert to Burn
        if not convert_to_burn(model['filename'], model['output_name']):
            continue
        
        # Step 4: Test with Furnace
        mpk_file = f"{model['output_name']}.mpk"
        if test_with_furnace(mpk_file, model['input_shape']):
            successful_conversions.append(model['name'])
    
    # Summary
    print("\n🎉 Conversion Test Summary")
    print("=" * 30)
    
    if successful_conversions:
        print("✅ Successfully converted models:")
        for model_name in successful_conversions:
            print(f"   - {model_name}")
        
        print("\n🚀 Next steps:")
        print("   1. Use any of the converted .mpk files with Furnace")
        print("   2. Try with your own ONNX models using the same process")
        print("   3. Check burn-import documentation for supported operators")
    else:
        print("❌ No models were successfully converted")
        print("\n🔧 Troubleshooting:")
        print("   1. Install burn-import: cargo install burn-import")
        print("   2. Install onnxruntime: pip install onnxruntime")
        print("   3. Build Furnace: cargo build --release")

if __name__ == "__main__":
    main()