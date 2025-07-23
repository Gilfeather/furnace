# README.md Update Requirements

## Introduction

Update the README.md to accurately reflect the current state of the Furnace inference server, which uses ONNX models with burn-import for code generation and supports built-in ResNet18 model.

## Requirements

### Requirement 1: Fix CLI Help Examples

**User Story:** As a developer reading the CLI help, I want to see examples that match the current CLI options, so that I understand how to use the tool correctly.

#### Acceptance Criteria

1. WHEN a user runs --help THEN they SHALL see examples using --model-name instead of --model-path
2. WHEN a user views CLI examples THEN they SHALL see current available options
3. WHEN a user reads long_about text THEN it SHALL reflect the current CLI structure

### Requirement 2: Update README Command Examples

**User Story:** As a developer following the README, I want command examples that work with the current implementation, so that I can successfully run the server.

#### Acceptance Criteria

1. WHEN a user sees server startup examples THEN they SHALL use --model-name resnet18
2. WHEN a user follows build instructions THEN they SHALL use cargo build --features burn-import
3. WHEN a user reads troubleshooting commands THEN they SHALL use current CLI parameters
4. WHEN a user tests API examples THEN they SHALL work with the actual backend (burn-resnet18)

### Requirement 3: Update API Documentation Examples

**User Story:** As a developer integrating with the API, I want accurate examples that reflect the actual responses, so that I can integrate correctly.

#### Acceptance Criteria

1. WHEN a user reads model info examples THEN they SHALL see backend: "burn-resnet18"
2. WHEN a user views inference examples THEN they SHALL see actual ResNet18 output format
3. WHEN a user checks input specifications THEN they SHALL see correct ResNet18 input shape [1,3,224,224]
4. WHEN a user reads error examples THEN they SHALL match current validation behavior

### Requirement 4: Clarify ONNX Integration Status

**User Story:** As a developer understanding Furnace architecture, I want clear information about ONNX support, so that I know what's currently implemented.

#### Acceptance Criteria

1. WHEN a user reads about ONNX support THEN they SHALL understand it uses burn-import for code generation
2. WHEN a user views model support THEN they SHALL see ResNet18 as the primary supported model
3. WHEN a user reads build instructions THEN they SHALL understand burn-import feature is required
4. WHEN a user sees architecture descriptions THEN they SHALL reflect the current ONNX-to-Rust code generation approach