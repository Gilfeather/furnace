# Contributing to Furnace 🔥

Thank you for your interest in contributing to Furnace! We welcome contributions from everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Scope: server, not engine](#scope-server-not-engine)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Scope: server, not engine

Furnace is an inference *server*. Its job is everything around the model: loading any ONNX
model, HTTP, input validation, batching, backpressure, timeouts, metrics, configuration,
logging and packaging. It should need to know almost nothing about what a model computes.

How fast the model itself runs (convolution and matrix kernels, SIMD, GPU backends) is
[Burn](https://github.com/tracel-ai/burn)'s job. Speed-ups there belong upstream in Burn, where
every Burn user gets them and Furnace picks them up by updating. `furnace-burn/vendor/burn-flex`
holds CPU kernel changes only until a Burn release includes them (see
[furnace-burn/vendor/README.md](furnace-burn/vendor/README.md)); please don't add new compute
kernels to Furnace.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Git
- A C linker (usually comes with a C compiler)

### Setting up the development environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/furnace.git
cd furnace

# Download the ResNet-18 ONNX model (compiled in at build time)
mkdir -p furnace-burn/models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o furnace-burn/models/resnet18.onnx
# Second built-in model used by the end-to-end tests
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" -o furnace-burn/models/mnist.onnx

# Build the project
cargo build --release

# Run tests (dependencies are optimized even in debug builds, so no --release needed)
cargo test

# Run the development server
cargo run --release -- --model-name resnet18 --port 3000
```

### Development Tools

We recommend installing these tools for development:

```bash
# Code formatting
rustup component add rustfmt

# Linting
rustup component add clippy

# Security auditing
cargo install cargo-audit

# Code coverage
cargo install cargo-tarpaulin
```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check the existing issues to avoid duplicates
2. Use the latest version of Furnace
3. Include as much detail as possible

When submitting a bug report, include:

- Operating system and version
- Rust version
- Furnace version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)

### Suggesting Features

We welcome feature suggestions! Please:

1. Check existing issues and discussions
2. Open an issue with the "enhancement" label
3. Describe the feature and its use case
4. Consider implementation details

### Contributing Code

1. **Pick an issue**: Look for issues labeled "good first issue" or "help wanted"
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes**: Follow our coding standards
4. **Test**: Ensure all tests pass
5. **Commit**: Use conventional commit messages
6. **Push**: Push to your fork
7. **Pull Request**: Open a PR with a clear description

## Coding Standards

### Rust Code Style

- Use `cargo fmt` to format code
- Follow Rust naming conventions
- Use `cargo clippy` and fix all warnings
- Write documentation for public APIs
- Add tests for new functionality

### Code Organization

```text
src/                     # furnace-server: the server, no ML framework
├── lib.rs               # Library exports
├── api.rs               # HTTP API handlers, limits, backpressure
├── model.rs             # TensorModel, and the Model wrapper: validation, batching, stats
├── tensor.rs            # Host tensors and their specs
├── nerve/               # The nervous system
├── llm/                 # The LLM layer (feature `llm`)
└── error.rs             # Error types
furnace-burn/src/        # Burn models
├── model.rs             # BurnModel, from_fn, custom, the load functions
├── tensor.rs            # Host tensors to Burn tensors and back
├── backend.rs           # Runtime backend selection (Burn Device)
├── weights.rs           # Burnpack weights, memory-mapped
├── mlp.rs               # File-based MLP
└── models/mod.rs        # Built-in models generated from ONNX (feature `onnx`)
furnace-cli/src/main.rs  # The `furnace` command
furnace-batch/           # The batching engine (no dependencies)
furnace-metrics/         # Latency histograms and the Prometheus text format (no dependencies)
furnace-alloc/           # The `furnace` command's global allocator (no dependencies)
```

### Error Handling

- Write error types as enums with their own `Display` and `std::error::Error` impls (no
  `thiserror`: every dependency of the server reaches every user)
- Provide meaningful error messages
- Include context when propagating errors
- Handle errors gracefully in the API layer

### Logging

- Use `tracing` for structured logging
- Use appropriate log levels:
  - `error!`: For errors that need immediate attention
  - `warn!`: For warnings that should be noted
  - `info!`: For general information
  - `debug!`: For debugging information
  - `trace!`: For very detailed debugging

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test module
cargo test model::tests

# Run integration tests
cargo test --test integration_tests

# Run with verbose output
cargo test -- --nocapture
```

### Writing Tests

- Write unit tests for all new functions
- Add integration tests for API endpoints
- Test error cases, not just happy paths
- Use descriptive test names
- Mock external dependencies when possible

### Test Structure

```text
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_function_name_with_valid_input() {
        // Arrange
        let input = create_test_input();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_output);
    }
}
```

## Documentation

### Code Documentation

- Document all public APIs with `///` comments
- Include examples in documentation when helpful
- Use `cargo doc` to generate documentation
- Keep documentation up to date with code changes

### README Updates

When making significant changes:

- Update the README.md if behavior changes
- Update code examples if APIs change
- Add new features to the feature list

## Pull Request Process

### Before Submitting

1. **Rebase your branch**: `git rebase main`
2. **Run tests**: `cargo test`
3. **Check formatting**: `cargo fmt --all -- --check`
4. **Run clippy**: `cargo clippy --all-targets --all-features -- -D warnings`
5. **Update documentation**: If you changed public APIs

### PR Description

Include in your PR description:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Breaking Changes**: Any breaking changes?
- **Related Issues**: Link to related issues

### PR Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of changes made
- Another change

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Breaking Changes
List any breaking changes

Closes #issue_number
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, your PR will be merged
4. CI will automatically run tests and checks

## Release Process

Releases are managed by maintainers:

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions will automatically create a release

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```text
feat(api): add batch prediction endpoint
fix(model): handle invalid input shapes correctly
docs: update API documentation
test(api): add integration tests for error handling
```

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing documentation
- Ask in pull request comments

## Recognition

Contributors are recognized in:
- The README.md contributors section
- Release notes for significant contributions
- GitHub's contributor graph

Thank you for contributing to Furnace! 🔥