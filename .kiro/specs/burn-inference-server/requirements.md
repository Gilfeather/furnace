# Requirements Document

## Introduction

furnaceは、Rust製のBurnベース推論サーバです。Pythonに依存せず、単一バイナリとして動作し、HTTP APIを通じて機械学習モデルの推論サービスを提供します。このプロジェクトのSprint 1では、CLIによるモデルロード機能と基本的なHTTP APIエンドポイントの実装を行います。

## Requirements

### Requirement 1

**User Story:** As a developer, I want to specify a .burn model file path via CLI arguments, so that the server can load and serve the specified model for inference.

#### Acceptance Criteria

1. WHEN the user runs the furnace command with --model-path argument THEN the system SHALL accept the path to a .burn model file
2. WHEN a valid .burn model file path is provided THEN the system SHALL successfully load the model into memory
3. WHEN an invalid or non-existent model file path is provided THEN the system SHALL display an appropriate error message and exit gracefully
4. WHEN the --model-path argument is missing THEN the system SHALL display usage information and exit with an error code

### Requirement 2

**User Story:** As a client application, I want to send inference requests via HTTP POST to /predict, so that I can get predictions from the loaded model.

#### Acceptance Criteria

1. WHEN a POST request is sent to /predict with valid JSON input THEN the system SHALL process the input through the loaded model
2. WHEN the inference is successful THEN the system SHALL return a JSON response with the prediction results and success status
3. WHEN the input format is invalid THEN the system SHALL return a 400 Bad Request status with error details
4. WHEN the model inference fails THEN the system SHALL return a 500 Internal Server Error status
5. WHEN the input data shape doesn't match the model's expected input shape THEN the system SHALL return a 400 Bad Request with validation error

### Requirement 3

**User Story:** As a monitoring system, I want to check the server health via GET /healthz, so that I can verify the server is running and the model is loaded.

#### Acceptance Criteria

1. WHEN a GET request is sent to /healthz THEN the system SHALL return a JSON response with health status
2. WHEN the server is running and model is loaded THEN the system SHALL return status "healthy" with model_loaded: true
3. WHEN the server is running but model failed to load THEN the system SHALL return status "unhealthy" with model_loaded: false
4. WHEN the health check is requested THEN the system SHALL respond within 100ms

### Requirement 4

**User Story:** As a client application, I want to retrieve model metadata via GET /model/info, so that I can understand the model's input/output specifications.

#### Acceptance Criteria

1. WHEN a GET request is sent to /model/info THEN the system SHALL return JSON with model metadata
2. WHEN model info is requested THEN the system SHALL include model name, input shape, output shape, and model type
3. WHEN the model is not loaded THEN the system SHALL return a 503 Service Unavailable status
4. WHEN model info is requested THEN the system SHALL return the information without performing inference

### Requirement 5

**User Story:** As a developer, I want to configure server host and port via CLI arguments, so that I can deploy the server in different environments.

#### Acceptance Criteria

1. WHEN the --port argument is provided THEN the system SHALL bind the server to the specified port
2. WHEN the --host argument is provided THEN the system SHALL bind the server to the specified host address
3. WHEN port or host arguments are not provided THEN the system SHALL use default values (127.0.0.1:3000)
4. WHEN an invalid port number is provided THEN the system SHALL display an error and exit gracefully
5. WHEN the specified port is already in use THEN the system SHALL display an appropriate error message

### Requirement 6

**User Story:** As a developer, I want comprehensive logging throughout the application, so that I can debug issues and monitor server behavior.

#### Acceptance Criteria

1. WHEN the server starts THEN the system SHALL log the model loading process with INFO level
2. WHEN inference requests are received THEN the system SHALL log request details with INFO level
3. WHEN errors occur THEN the system SHALL log error details with ERROR level
4. WHEN the server binds to a port THEN the system SHALL log the server address with INFO level
5. WHEN model loading fails THEN the system SHALL log detailed error information with ERROR level

### Requirement 7

**User Story:** As a developer, I want the server to handle CORS properly, so that web applications can make requests to the inference API.

#### Acceptance Criteria

1. WHEN a preflight OPTIONS request is received THEN the system SHALL respond with appropriate CORS headers
2. WHEN cross-origin requests are made THEN the system SHALL include Access-Control-Allow-Origin headers
3. WHEN CORS-enabled requests are made to any endpoint THEN the system SHALL allow the requests to proceed
4. WHEN invalid CORS requests are made THEN the system SHALL handle them gracefully without crashing

### Requirement 8

**User Story:** As a developer, I want the system to properly validate input data format, so that invalid requests are rejected before reaching the model.

#### Acceptance Criteria

1. WHEN JSON input is malformed THEN the system SHALL return a 400 Bad Request with parsing error details
2. WHEN required fields are missing from the request THEN the system SHALL return a 400 Bad Request with validation error
3. WHEN input array dimensions don't match model expectations THEN the system SHALL return a 400 Bad Request with shape mismatch error
4. WHEN input data types are incorrect THEN the system SHALL return a 400 Bad Request with type validation error