# Implementation Plan

- [x] 1. Fix build.rs to follow Burn documentation
  - Update build.rs to use correct ModelGen configuration
  - Generate Rust code from ONNX files instead of .mpk files
  - Remove problematic model filtering and error handling
  - _Requirements: 1.1, 1.2_

- [ ] 2. Create proper module structure for generated models
  - [x] 2.1 Create src/models/mod.rs with include! macros
    - Create models module directory structure
    - Add include! macro for resnet18 generated code
    - Export generated models properly
    - _Requirements: 1.2, 1.3_

  - [x] 2.2 Create GeneratedBurnModel wrapper
    - Implement wrapper for generated Burn models
    - Add proper tensor shape conversion (2D ↔ 4D)
    - Implement BurnModel trait for generated models
    - _Requirements: 1.3, 1.4_

- [ ] 3. Implement built-in model management
  - [x] 3.1 Create BuiltInModel enum
    - Define enum for available built-in models
    - Implement from_name method for model selection
    - Add create_model method for model instantiation
    - _Requirements: 2.1, 2.2_

  - [x] 3.2 Update model loading logic
    - Modify load_model_with_config to support built-in models
    - Add proper error handling for unknown models
    - Maintain fallback to dummy model
    - _Requirements: 2.2, 2.3_

- [x] 4. Update main.rs to use built-in models correctly
  - Remove file path creation for built-in models
  - Use BuiltInModel enum for model selection
  - Update error messages and logging
  - _Requirements: 2.1, 2.4_

- [ ] 5. Implement proper tensor shape handling
  - [ ] 5.1 Add tensor conversion utilities
    - Create utilities for 2D to 4D tensor conversion
    - Handle batch dimension properly
    - Add validation for tensor shapes
    - _Requirements: 1.4, 3.3_

  - [ ] 5.2 Update prediction pipeline
    - Modify predict methods to handle shape conversions
    - Ensure compatibility with existing API
    - Add proper error handling for shape mismatches
    - _Requirements: 3.1, 3.3_

- [ ] 6. Update API responses with correct model information
  - [ ] 6.1 Fix model info endpoint
    - Return correct input/output shapes for generated models
    - Include proper model type and backend information
    - Maintain compatibility with existing response format
    - _Requirements: 3.1, 3.2_

  - [ ] 6.2 Enhance prediction error handling
    - Provide clear error messages for shape mismatches
    - Include expected vs actual shapes in error responses
    - Test error scenarios thoroughly
    - _Requirements: 3.4_

- [ ] 7. Ensure backward compatibility
  - [ ] 7.1 Maintain existing Burn model support
    - Verify .mpk/.json model loading still works
    - Test existing model functionality
    - Ensure no regression in existing features
    - _Requirements: 4.1, 4.3_

  - [ ] 7.2 Preserve dummy model fallback
    - Keep dummy model as fallback option
    - Test fallback scenarios
    - Maintain existing dummy model behavior
    - _Requirements: 4.2, 4.3_

- [ ] 8. Create comprehensive tests
  - [ ] 8.1 Unit tests for generated models
    - Test model initialization and inference
    - Test tensor shape conversions
    - Test error handling scenarios
    - _Requirements: 1.3, 1.4_

  - [ ] 8.2 Integration tests for API endpoints
    - Test /model/info with generated models
    - Test /predict with various input shapes
    - Test error responses and edge cases
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 8.3 End-to-end testing
    - Test complete workflow from CLI to API response
    - Test model switching between built-in models
    - Verify backward compatibility with existing tests
    - _Requirements: 4.4_