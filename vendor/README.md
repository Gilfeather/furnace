# Vendored crates

## burn-flex

`burn-flex` 0.22.0 (Burn's CPU backend) with one patch to `src/ops/conv.rs`: 2D convolutions
build im2col directly from the NCHW input in the weight's own `(c_in, kh, kw)` order, instead of
re-laying out the weight and the input on every call. ResNet-18 on CPU goes from ~57 ms to
~27 ms per image.

The patch is proposed upstream in tracel-ai/burn. Once a Burn release includes it, delete this
directory and the `[patch.crates-io]` entries in `Cargo.toml` and `benches/compare/Cargo.toml`.

To check the vendored copy against upstream:

```bash
diff -r ~/.cargo/registry/src/*/burn-flex-0.22.0 vendor/burn-flex
```

Only `src/ops/conv.rs` should differ.
