# README.md Update Implementation Plan

- [ ] 1. Remove outdated ONNX file download instructions
  - Remove all curl commands for downloading .onnx files from Quick Start section
  - Remove ResNet-18 model download step from prerequisites
  - Remove model size and download verification steps
  - Update prerequisites to only include Rust and basic system requirements
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Update all command examples to use --model-name
  - Replace all --model-path examples with --model-name resnet18
  - Update server startup command to ./target/release/furnace --model-name resnet18 --port 3001
  - Fix CLI help examples in long description
  - Update troubleshooting commands to use --model-name parameter
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 3. Remove complex ONNX integration documentation
  - Remove entire "ONNX Model Integration" section
  - Remove "How ONNX Integration Works" subsection
  - Remove "Adding Custom ONNX Models" instructions
  - Remove build-time code generation explanations
  - Remove ONNX compatibility and troubleshooting sections
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Update build instructions to remove burn-import references
  - Change build command from cargo build --features burn-import to cargo build --release
  - Remove burn-import feature explanations
  - Update build output examples to reflect current behavior
  - Remove ONNX code generation build steps
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. Update API documentation with dummy model examples
  - Update model info endpoint examples to show backend: "dummy"
  - Update inference examples to show dummy output values (0.1)
  - Fix input validation error examples
  - Update all curl commands to work with current dummy model responses
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 6. Update performance metrics for dummy model behavior
  - Update performance badges to reflect dummy model inference times (~0.2ms)
  - Update benchmark results to show current dummy model performance
  - Update memory usage and binary size information
  - Remove ONNX-specific performance claims
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 7. Simplify supported models section
  - Update to show only resnet18 as built-in model with dummy implementation
  - Remove model download links and instructions
  - Update model table to show current dummy backend status
  - Remove custom model addition instructions
  - _Requirements: 1.4, 3.3_

- [ ] 8. Update troubleshooting section
  - Remove ONNX-related error solutions
  - Add --model-name vs --model-path confusion troubleshooting
  - Update port binding and server startup troubleshooting
  - Remove model download failure solutions
  - Update build error troubleshooting for current dependencies
  - _Requirements: 2.4, 4.4_

- [ ] 9. Test and validate all updated instructions
  - Test build command: cargo build --release
  - Test server startup: ./target/release/furnace --model-name resnet18 --port 3001
  - Test all API curl commands return expected dummy responses
  - Verify no broken links or references to removed sections
  - Ensure all command examples work with current CLI
  - _Requirements: 2.4, 4.4, 5.4, 6.2_