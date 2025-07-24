# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2025-01-19

### Fixed
- Fixed ONNX model generation using proper OUT_DIR environment variable
- Resolved Docker build issues by removing Docker support
- Updated CLI examples in README to use --model-name instead of --model-path
- Corrected API response examples to match actual ResNet18 output format
- Fixed performance metrics to reflect actual ResNet18 inference times (~25s)

### Changed
- Simplified CI workflow by removing Docker build job
- Updated README.md with accurate CLI usage and API documentation
- Improved build process reliability for ONNX code generation

### Removed
- Docker support and Dockerfile (focusing on native binary distribution)
- create_sample_model dependency from benchmarks

## [0.3.0] - 2025-01-21

### Added
- ResNet-18 ONNX model support with automatic shape detection
- Comprehensive benchmark suite with Criterion
- Server overhead analysis to separate ONNX inference from processing time
- Secure performance optimizations with SIMD-optimized validation
- Detailed performance breakdown in documentation
- Troubleshooting section in README.md
- ResNet-18 sample data generation examples

### Changed
- **BREAKING**: Replaced MNIST focus with ResNet-18 specialization
- Updated inference time from ~4ms to ~0.17ms (171µs) with optimizations
- Binary size updated to 4.5MB (from 2.3MB) due to enhanced ONNX support
- Complete README.md overhaul with accurate performance metrics
- Improved API documentation with ResNet-18 specific examples

### Performance Improvements
- 50% server processing optimization while maintaining full security
- ONNX inference: ~14µs (pure model execution)
- Server overhead: ~157µs (optimized data processing, validation, tensor conversion)
- Total end-to-end time: ~171µs
- Maximum throughput: 14,300 req/s (8 threads on Intel MacBook Pro 2020)
- Latency percentiles: P50: 149µs, P95: 255µs, P99: 340µs

### Security
- SIMD-optimized input validation for NaN/infinity detection
- Complete input validation maintained while achieving performance gains
- Non-blocking statistics updates to prevent deadlocks
- Memory-efficient tensor operations

### Removed
- Custom benchmark system in favor of Criterion
- MNIST model references and examples
- Outdated performance claims and documentation

### Technical Details
- Test Environment: Intel MacBook Pro 2020
- Compiler: Rust 1.75+ (release mode with full optimizations)
- Model: ResNet-18 ONNX (45MB, 150,528 input values → 1,000 output classes)

## [0.2.0] - Previous Release
- Initial implementation with basic ML inference capabilities
- MNIST model support
- Basic HTTP API endpoints

## [0.1.0] - Initial Release
- Basic project structure
- Core inference server functionality