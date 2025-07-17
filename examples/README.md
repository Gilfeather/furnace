# Furnace Examples

This directory contains examples demonstrating how to use Furnace with different types of models and scenarios.

## Available Examples

### 🎯 Basic MNIST Example (`basic_mnist/`)

A complete example showing how to:
- Create a simple MLP model for MNIST-like data (784→128→10)
- Save the model in Burn's MessagePack (.mpk) format
- Load and serve the model with Furnace
- Test inference via HTTP API

**Quick Start:**
```bash
# 1. Create the sample model
cargo run --example basic_mnist_create

# 2. Start the server (in one terminal)
cargo run --bin furnace -- --model-path examples/basic_mnist/model.mpk --log-level debug

# 3. Test inference (in another terminal)
cargo run --example basic_mnist_test
```

## Example Structure

Each example follows this structure:
```
examples/
├── example_name/
│   ├── README.md          # Detailed instructions
│   ├── model.mpk          # Generated model file (gitignored)
│   ├── model.json         # Generated metadata (gitignored)
│   └── data/              # Generated data files (gitignored)
├── example_name_create.rs # Model creation script
├── example_name_test.rs   # Inference testing script
└── README.md              # This file
```

## Key Features Demonstrated

- ✅ **Model Creation**: How to create and save Burn models
- ✅ **Multiple Formats**: Support for .burn and .mpk formats
- ✅ **HTTP API**: Complete inference API testing
- ✅ **Error Handling**: Input validation and error responses
- ✅ **Performance**: Timing and throughput testing
- ✅ **Structured Logging**: Development and production logging
- ✅ **Configuration**: Different server configurations

## Development Notes

- **Model files** (.mpk, .burn) are generated at runtime and not tracked in git
- **Data files** are also generated and gitignored to keep the repository clean
- **Examples are self-contained** - each can run independently
- **Comprehensive testing** - each example includes validation and error testing

## Adding New Examples

To add a new example:

1. Create a new directory: `examples/your_example/`
2. Add a README.md with instructions
3. Create `your_example_create.rs` for model creation
4. Create `your_example_test.rs` for testing
5. Update this README.md

Make sure to:
- Add model files to .gitignore
- Include comprehensive error testing
- Document the model architecture and use case
- Provide clear usage instructions