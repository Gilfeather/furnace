# syntax=docker/dockerfile:1
# The furnace demo in a container: a nervous system in one process, printed to the terminal.
# No ports, no model files.
#
#   docker build -f furnace-demo/Dockerfile -t furnace-demo .   # from the repository root
#   docker run --rm -t furnace-demo                              # -t for colors

FROM rust:1.99-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p furnace-demo \
    && mkdir -p /out && cp target/release/furnace-demo /out/

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /out/furnace-demo /app/furnace-demo
ENTRYPOINT ["/app/furnace-demo"]
