# README.md Update Design

## Overview

Update README.md to accurately reflect the current simplified Furnace implementation that uses built-in models with dummy implementations, removing all outdated ONNX file loading documentation.

## Architecture

### Document Structure
1. **Header Section**: Updated project description and simplified features
2. **Quick Start Section**: Simplified setup without ONNX downloads
3. **Built-in Models Section**: Current model support (resnet18 with dummy implementation)
4. **API Documentation**: Updated examples with dummy model responses
5. **Performance Section**: Current dummy model performance metrics
6. **Development Section**: Simplified build and test instructions
7. **Troubleshooting Section**: Updated for current functionality

## Components and Interfaces

### Header Component
- **Purpose**: Accurate project introduction
- **Content**: 
  - Updated description focusing on built-in models
  - Simplified feature list (remove ONNX complexity)
  - Current performance badges reflecting dummy model behavior
  - Clear positioning as inference server with built-in models

### Quick Start Component
- **Purpose**: Get users running with current functionality
- **Content**:
  - Prerequisites (Rust only, no model downloads)
  - Simple build command: `cargo build --release`
  - Server startup: `./target/release/furnace --model-name resnet18 --port 3001`
  - API testing with dummy responses
  - Remove all ONNX download steps

### Built-in Models Component
- **Purpose**: Explain current model support
- **Content**:
  - Available models: resnet18 (dummy implementation)
  - Input/output specifications
  - Backend information (dummy)
  - Remove custom model addition instructions

### API Documentation Component
- **Purpose**: Accurate API reference with current behavior
- **Content**:
  - Health check endpoint examples
  - Model info endpoint with dummy backend response
  - Inference endpoint with dummy output (0.1 values)
  - Correct input validation behavior
  - Remove ONNX-specific examples

## Data Models

### Current Performance Metrics
```
Built-in Model Performance:
- Binary Size: ~4.5MB
- Inference Time: ~0.2ms (dummy model)
- Memory Usage: <200MB
- Input Size: 150,528 values (ResNet-18 format)
- Output Size: 1,000 classes (dummy values)
- Backend: dummy
```

### API Response Examples
```
Model Info Response:
{
  "name": "resnet18",
  "backend": "dummy",
  "input_spec": {"shape": [1, 3, 224, 224]},
  "output_spec": {"shape": [1000]}
}

Inference Response:
{
  "outputs": [[0.1, 0.1, 0.1, ...]], // 1000 dummy values
  "status": "success"
}
```

## Error Handling

### Updated Common Issues
- Build compilation errors
- Server startup problems with --model-name
- Port binding issues
- Input validation errors

### Solutions
- Provide correct build commands
- Include troubleshooting for --model-name vs --model-path confusion
- Add validation steps for current CLI
- Remove ONNX-related error solutions

## Testing Strategy

### Validation Steps
1. Test simplified build process (cargo build --release)
2. Verify server startup with --model-name resnet18
3. Test all API endpoints return dummy responses
4. Validate all command examples work with current CLI
5. Ensure no broken links or outdated references
6. Check that removed ONNX sections don't leave orphaned content