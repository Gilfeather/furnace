# syntax=docker/dockerfile:1
# Furnace in a small container: the server binary and its weights on a distroless base image.
#
#   docker build -t furnace .                        # compiles every furnace-burn/models/*.onnx in
#   docker run --rm -p 3000:3000 furnace             # serves resnet18
#   docker run --rm -p 3000:3000 furnace --model-name mnist
#
#   docker run --rm -p 3000:3000 -p 8001:8001 furnace   # HTTP on 3000, gRPC on 8001
#
# Build arguments: FEATURES (Cargo features, default "grpc tls"; e.g. "grpc tls image"; without "grpc",
# drop `--grpc-port` from the ENTRYPOINT), and MODELS_DIR (default "furnace-burn/models").
#
# The binary links against glibc rather than musl: musl's allocator makes multithreaded
# inference markedly slower (ResNet-18 took 37% longer per image with the musl build).

FROM rust:1.99-slim-bookworm AS build
WORKDIR /src
COPY . .
ARG FEATURES="grpc tls"
ARG MODELS_DIR=furnace-burn/models
# The registry and target/ live in BuildKit cache mounts, so a rebuild after a source change
# recompiles furnace only, not Burn, rustls and tonic. The weights are copied from the out
# directory of this build's build script, which Cargo reports: target/ may hold others from
# earlier builds with other features.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    set -eu; \
    FURNACE_MODELS_DIR="/src/${MODELS_DIR}" \
        cargo build --release --locked -p furnace-cli ${FEATURES:+--features "$FEATURES"} \
        --message-format=json-render-diagnostics > /tmp/build.json; \
    out=$(grep '"reason":"build-script-executed"' /tmp/build.json | grep '#furnace-burn@' \
        | sed 's/.*"out_dir":"\([^"]*\)".*/\1/'); \
    mkdir -p /out; \
    cp target/release/furnace /out/; \
    for f in "$out"/models/*.bpk; do \
        if [ -e "$f" ]; then cp "$f" /out/; fi; \
    done; \
    ls -l /out

# glibc and the C runtime libraries Rust binaries need, nothing else; runs as a non-root user
FROM gcr.io/distroless/cc-debian12:nonroot
# The weights (`<model>.bpk`) sit next to the binary, where furnace looks for them first
COPY --from=build /out/ /app/
EXPOSE 3000 8001
ENTRYPOINT ["/app/furnace", "--host", "0.0.0.0", "--port", "3000", "--grpc-port", "8001"]
CMD ["--model-name", "resnet18"]
