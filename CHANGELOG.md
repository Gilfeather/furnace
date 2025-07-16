# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-16

### 🎉 Initial Release

#### ✨ Features
- **High-Performance ML Inference Server**: Pure Rust implementation with Burn framework
- **HTTP API**: RESTful endpoints for model inference
  - `GET /healthz` - Health check endpoint
  - `GET /model/info` - Model metadata and statistics
  - `POST /predict` - Run inference on input data
- **Single Binary Deployment**: Zero external dependencies (2.3MB binary)
- **Sub-millisecond Inference**: Average inference time ~0.5ms
- **Production Ready**: Graceful shutdown, comprehensive error handling
- **CORS Support**: Cross-origin resource sharing enabled
- **Memory Efficient**: <50MB memory usage, <100ms startup time

#### 🛠️ Technical Features
- **Burn Framework Integration**: Native ML operations with optimized tensors
- **Async HTTP Server**: Built with Axum and Tokio for high concurrency
- **Comprehensive Testing**: 17 unit tests + integration tests
- **CI/CD Pipeline**: Automated testing, linting, and security audits
- **Docker Support**: Containerized deployment ready

#### 📊 Performance Metrics
- **Binary Size**: 2.3MB
- **Inference Time**: ~0.5ms average (measured: 0.47ms)
- **Memory Usage**: <50MB
- **Startup Time**: <100ms
- **Concurrent Requests**: Tested and validated

#### 📚 Documentation
- **Comprehensive README**: Quick start guide, API documentation, architecture overview
- **Development Guide**: Local setup, testing, and contribution guidelines
- **Issue Templates**: Bug reports, feature requests, and good first issues
- **Roadmap**: Future features and enhancement plans

#### 🤝 Community
- **10 Good First Issues**: Beginner-friendly contribution opportunities
- **GitHub Templates**: Structured issue and PR templates
- **MIT License**: Open source and permissive licensing

### 🔮 Roadmap
- Batch inference endpoint (`/batch`)
- GPU support (Burn-wgpu backend)
- HTTPS/TLS support
- Hot model reload
- Enhanced metrics and monitoring
- Configuration file support

---

**Full Changelog**: https://github.com/Gilfeather/furnace/commits/v0.1.0