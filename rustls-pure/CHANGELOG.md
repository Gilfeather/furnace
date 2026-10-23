# Changelog

rustls-pure has a version of its own, apart from Furnace's, and is released only when it
changes. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0,
a breaking change, a new minimum Rust version or a new rustls minor version is a new minor
version.

## [Unreleased]

### Added
- CI builds it for RISC-V, WASI and the browser (`wasm32-unknown-unknown`); the README's
  "Targets" says what an application in the browser turns on.

### Fixed
- The package includes the license text (LICENSE), as the MIT license asks.

### Changed
- The timing tests read both classes' inputs and keep one by a mask, so the class decides no
  memory access. On Neoverse N2, reading only the class's input had made the X25519 all-zero
  check differ between classes (|t| up to 58); with this change it passes on every runner, and
  CONSTANT_TIME.md records why.

## [0.15.2] - 2026-10-23

Hardening: ML-KEM's encapsulation randomness is wiped, X25519's all-zero check is branch-free
code of rustls-pure's own, and the peer's bytes and constant time are checked in CI.

### Added
- Kani proofs (`cargo kani -p rustls-pure`, run in CI on changes to rustls-pure) of the code
  that takes the peer's bytes before the RustCrypto crates do: the size and shape of every
  group's key shares, how the X25519MLKEM768 shares are split and put together in both
  directions, the hybrid secret's order, record lengths and Ed25519 key and signature sizes.
- Constant time, checked three ways (CONSTANT_TIME.md): a review of rustls-pure's own code for
  secret-dependent branches, comparisons and wiping, with a survey of what each primitive crate
  does and documents; a check of the machine code of rustls-pure's own secret handling on
  x86_64 and aarch64 (`ct-asm.sh`, on pull requests); and timing tests in the style of dudect
  of every operation that handles a secret, gated on a control, weekly on x86_64 Linux,
  aarch64 Linux and Apple silicon.

### Changed
- P-256 and P-384 key shares are checked for the uncompressed point's exact size by
  rustls-pure before the curve crates parse them, as well as for its first byte; the same
  shares are refused as before.
- X25519's all-zero check is rustls-pure's own branch-free code instead of x25519-dalek's
  `was_contributory` (which is also constant time), so that its machine code can be checked;
  the same results are refused.

### Fixed
- ML-KEM's encapsulation randomness, from which the server's ML-KEM secret follows, is made in
  place and wiped; before, a copy of it was left unwiped on the stack.

### Security
- RUSTSEC-2026-0385: `ctutils` below 0.4.4, under the AEADs, ML-KEM and the curve crates, has
  masks that could compile to a branch on a secret. The repository's lockfile has 0.4.4; an
  application's own lockfile needs `cargo update -p ctutils`.

## [0.15.1] - 2026-10-22

Less work and fewer dependencies for the same behaviour; nothing measured, all found by reading
the code.

### Changed
- Dependencies without the default features rustls-pure does not use: `aes-gcm` and
  `chacha20poly1305` without OS randomness for nonces and the allocating API, `p256` and
  `p384` without PEM, `std` and their precomputed base-point tables (memory traded for time),
  `sha2` without allocation and OIDs. Every RustCrypto crate still picks AES-NI / ARMv8 AES,
  CLMUL / PMULL, AVX2 and SHA-NI / ARMv8 SHA at run time by itself.
- The README recommends X25519 (the default) and Ed25519 certificates for speed.
- Nonces are built without a length check at run time: the 12-byte nonce and 16-byte tag
  are now type bounds on the AEADs, checked at compile time.
- ECDSA signature verification hashes the message on the stack instead of into a `Vec`, and
  only after the key and the signature have parsed.
- The server's private key is shared with each handshake's signer instead of copied into it,
  so the secret exists once in memory.

## [0.15.0] - 2026-10-22

### Added
- Post-quantum key exchange: X25519MLKEM768 (ML-KEM-768 from FIPS 203, with X25519), as
  draft-ietf-tls-ecdhe-mlkem and rustls's own providers have it, on RustCrypto's `ml-kem`. It
  is offered first, as in rustls's default provider; a client offers its X25519 part as a share
  of its own too, so a server without ML-KEM takes plain X25519 without a HelloRetryRequest.
  Checked against Wycheproof's ML-KEM-768 vectors (key generation, encapsulation, decapsulation),
  with tests of how the hybrid puts its shares and secret together, and against OpenSSL 3.5.
- Full handshakes in memory between rustls-pure clients and servers (`tests/handshake.rs`):
  every cipher suite, key exchange group and key type, with client certificates,
  HelloRetryRequest, resumption, key updates and refused certificates.
- Random and broken input with proptest, in place of fuzzing (`src/tests/properties.rs`):
  key shares of every group, ML-KEM keys and ciphertexts, signatures and records.
- The interoperability tests also run on pull requests that change rustls-pure; on Linux they
  run on Debian 13 for its OpenSSL 3.5, and X25519MLKEM768 must be checked, not skipped.

### Changed
- X25519MLKEM768 is the first key exchange group, before X25519, P-256 and P-384. Peers without
  it keep working (see above); its key shares are larger (1216 bytes from the client, 1120 from
  the server).

## [0.14.1] - 2026-10-22

### Changed
- The minimum supported Rust version is 1.89 (it was 1.95, Furnace's), the lowest its
  dependencies allow; a weekly CI job tests it, and the newest rustls 0.23 release.
- The README says plainly that RSA signatures are not verified, so rustls-pure does not suit a
  client for arbitrary public websites, and which rustls and Rust versions are supported.

### Added
- Interoperability tests against OpenSSL, GnuTLS and curl (`interop/run.sh`), weekly on Linux
  and macOS.
- A security policy (SECURITY.md), including how advisories reach RustSec.

## [0.14.0] - 2026-10-22

Released with Furnace 0.14.0; no change to the provider.

## [0.13.0] - 2026-10-22

The provider behind Furnace's `tls` feature, as a crate of its own: TLS 1.3 with AES-GCM and
ChaCha20-Poly1305, X25519, P-256 and P-384 key exchange, and ECDSA P-256 / P-384 and Ed25519
keys and certificates, on RustCrypto primitives. Its tests run Google's Wycheproof vectors for
every algorithm offered and check nonce construction, record integrity, the source of
randomness and the wiping of keys.
