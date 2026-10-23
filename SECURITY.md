# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[security advisories](https://github.com/Gilfeather/furnace/security/advisories/new), not in a
public issue. Include what is affected, how to reproduce it, and the impact you expect. You
should get a first answer within a week. Once a fix is released, the advisory is published with
credit to you, unless you prefer otherwise.

rustls-pure, the TLS crypto provider (released apart from Furnace), has a policy of its own:
[rustls-pure/SECURITY.md](rustls-pure/SECURITY.md).

## Supported versions

Fixes go into the latest release (0.x: the latest minor version). There are no backports to
older minor versions before 1.0.

## What Furnace does to stay safe

- **Pure Rust**, including TLS: no C or C++ in the server, so no memory-safety bugs from them.
  The TLS crypto is RustCrypto behind rustls; RSA is refused because the pure-Rust RSA crate has
  an unfixed timing side channel (RUSTSEC-2023-0071).
- **Dependencies are audited whenever they change and weekly** with `cargo deny` against the RustSec
  advisory database; no advisory is ignored.
- **Requests are bounded**: body size, batch size, concurrency (`503` beyond it), timeouts, and
  every input is checked against the model before it reaches it.
- **Access control** when exposed: API keys (compared in constant time), per-client rate limits,
  mutual TLS, and an admin API that is off unless a token is set.
- **Releases are verifiable**: every binary and container image carries a build provenance
  attestation, and each release lists SHA-256 checksums and an SPDX software bill of materials.

```bash
gh attestation verify furnace-linux-x86_64 --repo Gilfeather/furnace
gh attestation verify oci://ghcr.io/gilfeather/furnace:0.16.0 --repo Gilfeather/furnace
```
