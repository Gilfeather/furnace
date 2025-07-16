# Build stage
FROM rust:1.88-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml ./

# Create dummy files to build dependencies
RUN mkdir -p src benches tests && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > benches/inference_benchmark.rs && \
    echo "fn main() {}" > tests/integration_tests.rs

# Generate lock file and build dependencies (this will be cached if Cargo.toml doesn't change)
RUN cargo build --release && rm -rf src benches tests

# Copy source code
COPY src/ src/
COPY benches/ benches/
COPY tests/ tests/

# Build the actual application
RUN touch src/main.rs && cargo build --release

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

# Copy binary
COPY --from=builder /app/target/release/furnace /usr/local/bin/furnace
RUN chmod +x /usr/local/bin/furnace

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
CMD ["--model-path", "/app/models/model.burn", "--host", "0.0.0.0", "--port", "3000"]