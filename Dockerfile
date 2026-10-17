# Furnace in a small container: the server binary and its weights on a distroless base image.
#
#   docker build -t furnace .                        # compiles every models/*.onnx in
#   docker run --rm -p 3000:3000 furnace             # serves resnet18
#   docker run --rm -p 3000:3000 furnace --model-name mnist
#
# Build arguments: FEATURES (Cargo features, e.g. "image"), and MODELS_DIR (default "models").
#
# The binary links against glibc rather than musl: musl's allocator makes multithreaded
# inference markedly slower (ResNet-18 took 37% longer per image with the musl build).

FROM rust:1.99-slim-bookworm AS build
WORKDIR /src
COPY . .
ARG FEATURES=""
ARG MODELS_DIR=models
RUN set -eu; \
    FURNACE_MODELS_DIR="/src/${MODELS_DIR}" \
        cargo build --release --locked --bin furnace ${FEATURES:+--features "$FEATURES"}; \
    mkdir -p /out; \
    cp target/release/furnace /out/; \
    for f in target/release/build/furnace-server-*/out/models/*.bpk; do \
        if [ -e "$f" ]; then cp "$f" /out/; fi; \
    done; \
    ls -l /out

# glibc and the C runtime libraries Rust binaries need, nothing else; runs as a non-root user
FROM gcr.io/distroless/cc-debian12:nonroot
# The weights (`<model>.bpk`) sit next to the binary, where furnace looks for them first
COPY --from=build /out/ /app/
EXPOSE 3000
ENTRYPOINT ["/app/furnace", "--host", "0.0.0.0", "--port", "3000"]
CMD ["--model-name", "resnet18"]
