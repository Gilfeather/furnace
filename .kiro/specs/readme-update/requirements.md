# README.md Update Requirements

## Introduction

Update the README.md to accurately reflect the current simplified state of the Furnace inference server, which now uses built-in models with dummy implementations instead of ONNX file loading.

## Requirements

### Requirement 1: Remove Outdated ONNX File Download Instructions

**User Story:** As a developer following the setup guide, I want accurate instructions that work with the current implementation, so that I don't waste time trying to download unnecessary files.

#### Acceptance Criteria

1. WHEN a user reads the Quick Start section THEN they SHALL NOT see instructions to download ResNet-18 ONNX files
2. WHEN a user follows setup steps THEN they SHALL NOT see curl commands for downloading .onnx files
3. WHEN a user reads prerequisites THEN they SHALL NOT see model download requirements
4. WHEN a user views the setup process THEN they SHALL understand models are built-in

### Requirement 2: Update Command Examples to Use --model-name

**User Story:** As a developer wanting to run Furnace, I want correct command examples that work with the current CLI, so that I can successfully start the server.

#### Acceptance Criteria

1. WHEN a user sees server startup examples THEN they SHALL use --model-name resnet18 instead of --model-path
2. WHEN a user views CLI help examples THEN they SHALL see updated command syntax
3. WHEN a user reads troubleshooting commands THEN they SHALL use --model-name parameter
4. WHEN a user follows any command example THEN it SHALL work with the current CLI

### Requirement 3: Remove Complex ONNX Integration Documentation

**User Story:** As a developer reading about Furnace capabilities, I want to understand the current simplified architecture, so that I don't expect features that don't exist.

#### Acceptance Criteria

1. WHEN a user reads the ONNX Model Integration section THEN they SHALL understand it's been simplified to built-in models
2. WHEN a user views model addition instructions THEN they SHALL understand custom ONNX models are not supported
3. WHEN a user reads about build-time code generation THEN they SHALL understand it's been removed
4. WHEN a user sees architecture descriptions THEN they SHALL reflect the current dummy model approach

### Requirement 4: Update Build Instructions

**User Story:** As a developer building Furnace, I want correct build commands that work with the current codebase, so that I can successfully compile the project.

#### Acceptance Criteria

1. WHEN a user follows build instructions THEN they SHALL use cargo build --release without burn-import features
2. WHEN a user sees feature-specific builds THEN they SHALL understand burn-import is no longer needed for basic usage
3. WHEN a user reads about dependencies THEN they SHALL see the simplified dependency list
4. WHEN a user builds the project THEN they SHALL get a working binary with built-in models

### Requirement 5: Update API Documentation and Examples

**User Story:** As a developer integrating with Furnace API, I want accurate examples that work with dummy models, so that I can test the integration successfully.

#### Acceptance Criteria

1. WHEN a user tests API endpoints THEN they SHALL see dummy model responses (backend: "dummy")
2. WHEN a user reads model info examples THEN they SHALL see built-in model metadata
3. WHEN a user follows inference examples THEN they SHALL get consistent dummy outputs (0.1 values)
4. WHEN a user checks performance metrics THEN they SHALL understand they reflect dummy model behavior

### Requirement 6: Update Performance and Benchmark Information

**User Story:** As a developer evaluating Furnace performance, I want accurate information about current capabilities, so that I can set appropriate expectations.

#### Acceptance Criteria

1. WHEN a user reads performance metrics THEN they SHALL understand they reflect dummy model performance
2. WHEN a user views benchmark results THEN they SHALL see current actual performance with built-in models
3. WHEN a user reads about optimization features THEN they SHALL understand their current effectiveness
4. WHEN a user compares performance claims THEN they SHALL match actual dummy model behavior