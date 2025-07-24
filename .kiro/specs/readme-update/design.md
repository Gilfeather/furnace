# README.md Update Design

## Overview

Update README.md to accurately reflect the current Furnace implementation that uses ONNX models with burn-import for code generation, supporting ResNet18 with actual inference capabilities.

## Architecture

### Document Structure
1. **Header Section**: Updated project description and current features
2. **Quick Start Section**: Correct build and setup instructions
3. **ONNX Integration Section**: Current ONNX-to-Rust code generation approach
4. **API Documentation**: Updated examples with actual ResNet18 responses
5. **Performance Section**: Current ResNet18 performance metrics
6. **Development Section**: Correct build and test instructions with burn-import
7. **Troubleshooting Section**: Updated for current CLI and functionality

## Components and Interfaces

### Header Component
- **Purpose**: Accurate project introduction
- **Content**: 
  - Updated description focusing on ONNX support with burn-import
  - Current feature list reflecting actual capabilities
  - Performance badges reflecting actual ResNet18 performance
  - Clear positioning as ONNX inference server with code generation

### Quick Start Component
- **Purpose**: Get users running with current functionality
- **Content**:
  - Prerequisites (Rust, system dependencies for ONNX)
  - Correct build command: `cargo build --features burn-import --release`
  - Server startup: `./target/release/furnace --model-name resnet18 --port 3000`
  - API testing with actual ResNet18 responses
  - ONNX model download handled by CI/build process

### ONNX Integration Component
- **Purpose**: Explain current ONNX support architecture
- **Content**:
  - How burn-import generates Rust code from ONNX models
  - ResNet18 as primary supported model
  - Build-time code generation process
  - Model availability and limitations

### API Documentation Component
- **Purpose**: Accurate API reference with current behavior
- **Content**:
  - Health check endpoint examples
  - Model info endpoint with burn-resnet18 backend response
  - Inference endpoint with actual ResNet18 output format
  - Correct input validation behavior (150,528 input elements)
  - Current error response formats

## Data Models

### Current Performance Metrics
```
ResNet18 Model Performance:
- Binary Size: ~4.5MB
- Inference Time: ~25s (actual ResNet18 inference)
- Memory Usage: <200MB
- Input Size: 150,528 values (3×224×224 ResNet-18 format)
- Output Size: 1,000 classes (ImageNet)
- Backend: burn-resnet18
```

### API Response Examples
```
Model Info Response:
{
  "model_info": {
    "name": "resnet18",
    "backend": "burn-resnet18",
    "input_spec": {"shape": [1, 3, 224, 224]},
    "output_spec": {"shape": [1000]}
  }
}

Inference Response:
{
  "output": [-1.596, -0.173, ...], // 1000 actual ResNet18 values
  "status": "success",
  "inference_time_ms": 25477.0
}
```

## Error Handling

### Updated Common Issues
- Build compilation errors with burn-import
- ONNX model download failures in CI
- Server startup problems with --model-name
- Input validation errors for ResNet18 format

### Solutions
- Provide correct build commands with burn-import feature
- Include troubleshooting for --model-name vs --model-path confusion
- Add validation steps for current CLI and ONNX requirements
- Include ONNX-specific error solutions

## Testing Strategy

### Validation Steps
1. Test build process with burn-import feature (cargo build --features burn-import --release)
2. Verify server startup with --model-name resnet18
3. Test all API endpoints return actual ResNet18 responses
4. Validate all command examples work with current CLI
5. Ensure ONNX integration documentation is accurate
6. Test inference with correct ResNet18 input format (150,528 elements)