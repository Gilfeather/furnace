# rustls-pure

A crypto provider for [rustls](https://github.com/rustls/rustls) built on
[RustCrypto](https://github.com/RustCrypto) primitives.

- **Pure Rust.** No C, no assembly, no `ring` or `aws-lc-rs`, not even in its tests: it builds
  wherever Rust does, with no C toolchain, and `#![forbid(unsafe_code)]` holds in the provider
  itself.
- **Post-quantum key exchange.** X25519MLKEM768 (ML-KEM-768, FIPS 203, together with X25519),
  offered first by default as rustls's own default provider does, so a connection is safe from
  a future quantum computer as long as either half holds.
- **Checked against Wycheproof and other TLS implementations.** The tests run Google's
  [Wycheproof](https://github.com/C2SP/wycheproof) test vectors for every algorithm offered,
  about 4,800 cases: valid inputs must be accepted, crafted bad ones (forged signatures, points
  off the curve, altered tags, low-order keys, unreduced ML-KEM keys, malformed DER) refused.
  Full handshakes run in memory for every suite, group and key type, with HelloRetryRequest,
  resumption and key updates; random and broken input goes to every parser; and OpenSSL,
  GnuTLS and curl connect to it and serve it, X25519MLKEM768 included. Further tests pin what
  none of these can see: each record's nonce is unique, randomness comes from the operating
  system, and every key is wiped when dropped.
- **Strict about bad input.** Where the standards leave a choice, it refuses: an X25519
  exchange that ends in an all-zero secret (RFC 8446 section 7.4.2), in the hybrid too,
  compressed or malformed key shares, malleable Ed25519 signatures and small-order Ed25519 keys
  (`verify_strict`).
- **TLS 1.3 only.** No TLS 1.2, no RSA, no legacy suites: less code to get wrong and to review.

It has **not had an outside security audit**, and neither have the RustCrypto crates under it.
The checks above are mechanical; they are not a substitute for one. If you need audited TLS,
use rustls with its default provider, or terminate TLS in a proxy.

| | Supported |
|---|---|
| Protocol | TLS 1.3 |
| Cipher suites | `TLS13_AES_128_GCM_SHA256`, `TLS13_AES_256_GCM_SHA384`, `TLS13_CHACHA20_POLY1305_SHA256` |
| Key exchange | X25519MLKEM768 (preferred), X25519, P-256, P-384 |
| Keys and certificates | ECDSA P-256, ECDSA P-384, Ed25519 (PKCS#8, or SEC1 for ECDSA) |

X25519MLKEM768 follows draft-ietf-tls-ecdhe-mlkem as rustls's own providers do: the ML-KEM part
comes first in the key shares and the secret, and a client offers its X25519 part as a separate
share too, so a server without ML-KEM can take plain X25519 without a HelloRetryRequest. Keys
are wiped when dropped, and ML-KEM keys the peer sends must pass FIPS 203's modulus check.
Post-quantum signatures (ML-DSA) are not offered: certificate authorities do not issue them
yet.

## What it is for, and what not

**rustls-pure does not verify RSA signatures.** It accepts certificates signed with ECDSA
P-256 / P-384 or Ed25519 only, and refuses any chain with an RSA signature in it. Much of the
public web still serves RSA certificates, or ECDSA ones under RSA-signed intermediates and
roots, so **it is not suited to a client that connects to arbitrary public websites**. It is
meant for connections where you choose the certificates: your own servers and services,
service-to-service and mutual TLS, an internal CA, and public servers whose whole chain is
ECDSA.

RSA is left out on purpose. Its only pure-Rust implementation, the `rsa` crate, has an unfixed
timing side channel ([RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)),
which would come with it into every build; and ECDSA and Ed25519 are the modern choice anyway.
Every CA, cert-manager and Let's Encrypt can issue ECDSA certificates.

## Usage

```toml
[dependencies]
rustls = { version = "0.23", default-features = false, features = ["std"] }
rustls-pure = "0.15"
```

Pass the provider to rustls's config builders, with TLS 1.3 as the protocol version. A server:

```rust
use std::sync::Arc;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

let certs = CertificateDer::pem_file_iter("cert.pem")?.collect::<Result<Vec<_>, _>>()?;
let key = PrivateKeyDer::from_pem_file("key.pem")?;
let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls_pure::provider()))
    .with_protocol_versions(&[&rustls::version::TLS13])?
    .with_no_client_auth()
    .with_single_cert(certs, key)?;
```

A client:

```rust
use std::sync::Arc;

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;

let mut roots = rustls::RootCertStore::empty();
roots.add(CertificateDer::from_pem_file("ca.pem")?)?;
let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls_pure::provider()))
    .with_protocol_versions(&[&rustls::version::TLS13])?
    .with_root_certificates(roots)
    .with_no_client_auth();
```

Both work with tokio-rustls and anything else that takes a rustls config.

## Speed

For speed, prefer X25519 (the default, inside X25519MLKEM768) for key exchange and Ed25519
certificates where your clients take them. RustCrypto's P-256 and P-384 are written for
correctness in portable Rust, and are slower than the assembly in ring, aws-lc-rs or OpenSSL;
their precomputed base-point tables, which speed up key generation and signing, are on (built
once on first use, about 25 KB for P-256 and 56 KB for P-384). The ciphers and hashes use
AES-NI, CLMUL, AVX2, SHA-NI and their ARM counterparts when the CPU has them.

## Supported versions

- **rustls** 0.23.46 or any later 0.23 release. A weekly CI job builds and tests against the
  newest one.
- **Rust** 1.89 or later (the MSRV), tested weekly too.

rustls-pure has a version of its own, apart from Furnace's, and a release only when it changes
([CHANGELOG](CHANGELOG.md)). Before 1.0, raising either minimum, or moving to rustls 0.24, is
a minor version.

## Compared with the other pure-Rust providers

Others have the same aim, most with a wider reach: from the RustCrypto organization,
[rustls-rustcrypto](https://github.com/RustCrypto/rustls-rustcrypto);
[rustls-provider-rustcrypto-pq](https://github.com/ctf16/rustls-provider-rustcrypto-pq), with
more post-quantum algorithms; and [OxiTLS](https://github.com/cool-japan/oxitls), a TLS facade
over rustls with its own fork of rustls-rustcrypto's provider. From their crates and
repositories, as of October 2026:

| | rustls-pure | rustls-provider-rustcrypto-pq | rustls-rustcrypto | OxiTLS |
|---|---|---|---|---|
| Release | 0.15 | 0.1.1 (October 2026) | 0.0.2-alpha (April 2024); its README says not to use it in production | 0.3.0 (August 2026) |
| Outside audit | none | none | none | none stated |
| TLS versions | 1.3 | 1.2 and 1.3 | 1.2 and 1.3 | 1.2 and 1.3 |
| Post-quantum key exchange | X25519MLKEM768, first by default | X25519MLKEM768, SecP256r1MLKEM768, SecP384r1MLKEM1024, ML-KEM alone | none | X25519MLKEM768 (a feature) |
| Post-quantum signatures | none | ML-DSA | none | none stated |
| RSA | none, so no RUSTSEC-2023-0071 | yes (the `rsa` release candidate) | yes | yes |
| Wycheproof vectors in its tests | yes, every algorithm offered | none found | none found | none found |
| Checked against other implementations | OpenSSL, GnuTLS and curl, on Linux and macOS | rustls's aws-lc-rs provider | | |
| C or assembly in its tests | none | aws-lc-rs (dev-dependency) | | |
| X25519 all-zero secret | refused | | not checked | |
| Ed25519 verification | `verify_strict` | | `verify` | |

Pick one of the others if you need TLS 1.2, RSA, ML-DSA or QUIC (rustls-rustcrypto). Pick
rustls-pure if TLS 1.3 with ECDSA or Ed25519 keys is enough and you want the narrowest,
strictest provider, checked against test vectors and other implementations without any C.

## Tests

`cargo test -p rustls-pure` from a checkout of the repository runs:

- the Wycheproof vectors, in [`wycheproof/`](wycheproof/README.md) (Apache-2.0, copied unchanged
  from a pinned commit), and the checks of nonces, records, randomness, key wiping and how the
  hybrid key exchange puts its parts together (`src/tests.rs`)
- full handshakes in memory between rustls-pure clients and servers (`tests/handshake.rs`):
  every cipher suite, key exchange group and key type, with client certificates,
  HelloRetryRequest, resumption, key updates, and certificates from the wrong CA refused. The
  certificates are in [`testdata/`](testdata/generate.sh). rustls's own providers would make
  a natural peer, but they bring C and assembly, so the other side is rustls-pure here and other
  implementations come in through the interoperability tests
- random and broken input, with [proptest](https://crates.io/crates/proptest)
  (`src/tests/properties.rs`): key shares of every group, ML-KEM keys and ciphertexts,
  signatures and records. This takes the place of fuzzing: cargo-fuzz runs on libFuzzer, which
  is C++

`bash rustls-pure/interop/run.sh` checks it against other TLS implementations on the same
machine, with certificates it makes for the run: OpenSSL, and GnuTLS and curl when installed,
each as a client of rustls-pure's server and as a server for its client, with every cipher
suite, key exchange group (X25519MLKEM768 with OpenSSL 3.5 or later) and key type, with client
certificates, and refusing a server with an RSA certificate or an unknown CA. It runs weekly,
and on pull requests that change rustls-pure, on Linux and macOS.

The vectors, the certificates and the scripts are left out of the published crate.

## Security

Report vulnerabilities privately, as [SECURITY.md](SECURITY.md) describes.

MIT licensed. Forged in kamiforge: built as part of [Furnace](https://github.com/Gilfeather/furnace),
the pure-Rust inference server, which serves HTTPS and gRPC with it.
