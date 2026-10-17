# Vendored crates

## burn-flex

`burn-flex` 0.22.0 (Burn's CPU backend) with faster 2D convolution and max pooling:

- `src/ops/conv.rs`: 2D convolutions build im2col directly from the NCHW input in the weight's
  own `(c_in, kh, kw)` order, instead of re-laying out the weight and the input on every call.
  The work is split into small tasks (a tile of output positions by a block of output channels),
  each filling a cache-sized im2col tile in a reused per-thread buffer and running a
  single-threaded GEMM. On x86_64 with AVX2 or AVX-512, and on aarch64 with NEON, `f32`
  convolutions without groups or dilation skip im2col altogether and go to the two paths below.
- `src/ops/conv_direct.rs` (new): direct convolution as an implicit GEMM. The input is copied
  once into a zero-padded buffer, split into phase planes for strides above 1, so every kernel
  tap is a plain vector load; `std::arch` microkernels broadcast the OIHW weights as they are.
- `src/ops/winograd.rs` (new): Winograd F(2x2, 3x3) for 3x3 stride-1 convolutions of up to
  128 x 128 channels, on the same microkernels. The transformed filters are cached by
  `src/ops/weight_cache.rs` (new), keyed by the weight storage's identity and holding only a
  `Weak` reference, so modified weights are never served stale.
- `src/ops/scratch.rs` (new): a small pool of reused work buffers for those kernels.
- `src/ops/pool.rs`: max pooling without indices no longer computes and discards the argmax
  index tensor, and 2D max pooling without dilation runs as two separable passes.

ResNet-18 on CPU goes from ~57 ms with stock burn-flex 0.22.0 to ~8 ms per image.

The im2col and max pooling changes are proposed upstream in tracel-ai/burn. Once a Burn release
includes everything here, delete this directory and the `[patch.crates-io]` entries in
`Cargo.toml` and `benches/compare/Cargo.toml`.

To check the vendored copy against upstream:

```bash
diff -r ~/.cargo/registry/src/*/burn-flex-0.22.0 vendor/burn-flex
```

Only the files listed above should differ, plus `src/ops/mod.rs`, which declares the new
modules.
