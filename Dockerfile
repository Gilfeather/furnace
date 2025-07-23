# Build stage
FROM rust:1.88-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    g++ \
    cmake \
    libfontconfig1-dev \
    libfreetype6-dev \
    libexpat1-dev \
    libbz2-dev \
    libpng-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml ./

# Create dummy files to build dependencies
RUN mkdir -p src/bin benches tests && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/benchmark.rs && \
    echo "fn main() {}" > benches/inference_benchmark.rs && \
    echo "fn main() {}" > tests/integration_tests.rs

# Generate lock file and build dependencies (this will be cached if Cargo.toml doesn't change)
RUN cargo build --release && rm -rf src benches tests

# Copy source code
COPY src/ src/
COPY benches/ benches/
COPY tests/ tests/

# Download ONNX models and create sample model
RUN mkdir -p models && \
    curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o models/resnet18.onnx

# Build the actual application
RUN touch src/main.rs && cargo build --release

# Create sample model for runtime
RUN ./target/release/create_sample_model

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r furnace && useradd -r -g furnace furnace

# Create directories
RUN mkdir -p /app/models && chown -R furnace:furnace /app

# Copy binaries
COPY --from=builder /app/target/release/furnace /usr/local/bin/furnace
COPY --from=builder /app/target/release/benchmark /usr/local/bin/benchmark
RUN chmod +x /usr/local/bin/furnace /usr/local/bin/benchmark

# Copy models
COPY --from=builder /app/models/ /app/models/
COPY --from=builder /app/sample_model.mpk /app/models/model.mpk

# Switch to non-root user
USER furnace

WORKDIR /app

# Expose port
EXPOSE 3000

# Health check (using wget since curl might not be available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/healthz || exit 1

# Default command
ENTRYPOINT ["furnace"]
CMD ["--model-path", "/app/models/model.mpk", "--host", "0.0.0.0", "--port", "3000"]