# README.md Update Requirements

## Introduction

Update the README.md to accurately reflect the current state of the Furnace inference server, focusing on ResNet-18 support and removing outdated MNIST references.

## Requirements

### Requirement 1: Accurate Project Description

**User Story:** As a developer discovering Furnace, I want to understand what the project does and its current capabilities, so that I can evaluate if it meets my needs.

#### Acceptance Criteria

1. WHEN a user reads the project description THEN they SHALL understand that Furnace is a ResNet-18 focused inference server
2. WHEN a user views the badges THEN they SHALL see accurate performance metrics for ResNet-18 (~4ms inference time)
3. WHEN a user reads the features list THEN they SHALL see ONNX support and ResNet-18 optimization highlighted

### Requirement 2: Complete Setup Instructions

**User Story:** As a developer wanting to use Furnace, I want clear step-by-step setup instructions, so that I can get the server running quickly.

#### Acceptance Criteria

1. WHEN a user follows the setup instructions THEN they SHALL be able to download ResNet-18 model successfully
2. WHEN a user runs the build commands THEN they SHALL get a working binary
3. WHEN a user generates test data THEN they SHALL have the necessary JSON files for testing
4. WHEN a user starts the server THEN it SHALL load ResNet-18 and be ready for inference

### Requirement 3: Comprehensive API Documentation

**User Story:** As a developer integrating with Furnace, I want detailed API documentation with ResNet-18 examples, so that I can make successful API calls.

#### Acceptance Criteria

1. WHEN a user reads the API section THEN they SHALL see ResNet-18 specific input/output formats
2. WHEN a user views the examples THEN they SHALL see actual curl commands that work with ResNet-18
3. WHEN a user checks the response format THEN they SHALL understand the 1000-class ImageNet output

### Requirement 4: Accurate Performance Information

**User Story:** As a developer evaluating Furnace, I want accurate performance benchmarks for ResNet-18, so that I can assess if it meets my performance requirements.

#### Acceptance Criteria

1. WHEN a user reads the performance section THEN they SHALL see ResNet-18 specific metrics
2. WHEN a user views the benchmark instructions THEN they SHALL be able to reproduce the results
3. WHEN a user checks the system requirements THEN they SHALL understand the prerequisites

### Requirement 5: Clear Development Instructions

**User Story:** As a contributor to Furnace, I want clear development setup and contribution guidelines, so that I can contribute effectively.

#### Acceptance Criteria

1. WHEN a developer wants to contribute THEN they SHALL find clear build and test instructions
2. WHEN a developer runs benchmarks THEN they SHALL get consistent results
3. WHEN a developer wants to extend the code THEN they SHALL understand the architecture