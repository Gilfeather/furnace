# Vendored crates

## burn-flex

`burn-flex` 0.22.0 (Burn's CPU backend) with two patched files:

- `src/ops/conv.rs`: 2D convolutions build im2col directly from the NCHW input in the weight's
  own `(c_in, kh, kw)` order, instead of re-laying out the weight and the input on every call.
  The work is split into small tasks (a tile of output positions by a block of output channels),
  each filling a cache-sized im2col tile in a reused per-thread buffer and running a
  single-threaded GEMM.
- `src/ops/pool.rs`: max pooling without indices no longer computes and discards the argmax
  index tensor.

ResNet-18 on CPU goes from ~57 ms with stock burn-flex 0.22.0 to ~14 ms per image.

The changes are proposed upstream in tracel-ai/burn. Once a Burn release includes them, delete this
directory and the `[patch.crates-io]` entries in `Cargo.toml` and `benches/compare/Cargo.toml`.

To check the vendored copy against upstream:

```bash
diff -r ~/.cargo/registry/src/*/burn-flex-0.22.0 vendor/burn-flex
```

Only `src/ops/conv.rs` and `src/ops/pool.rs` should differ.
