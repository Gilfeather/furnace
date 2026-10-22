# rustls-pure security policy

## Reporting a vulnerability

Report it privately through GitHub's
[security advisories](https://github.com/Gilfeather/furnace/security/advisories/new) for the
Furnace repository, where rustls-pure lives, with "rustls-pure" in the title. Please do not open
a public issue. Say which version is affected, how to reproduce it (a test vector or a packet
capture helps most), and what an attacker gains.

What counts, for example:

- a signature, certificate chain, key share or record that should be refused and is accepted
  (or the other way round, where that breaks connections)
- a nonce or key used twice, or a key left in memory after it is dropped
- a timing difference in rustls-pure's own code that depends on secret data
- a panic or unbounded work that a peer can trigger
- any disagreement with the Wycheproof vectors

Bugs in rustls itself or in a primitive crate belong to those projects: rustls's
[security policy](https://github.com/rustls/rustls/security/policy), the policy in each
RustCrypto repository (such as [AEADs](https://github.com/RustCrypto/AEADs/security/policy) or
[elliptic-curves](https://github.com/RustCrypto/elliptic-curves/security/policy)), and
[dalek-cryptography](https://github.com/dalek-cryptography/curve25519-dalek) for X25519 and
Ed25519. If you are not
sure where a problem sits, report it here and we will take it to the right place. That
rustls-pure does not verify RSA signatures is a design decision (see the README), not a
vulnerability.

## What happens next

1. You get an answer within a week, and we agree on the issue and its severity.
2. The fix is written in the advisory's private fork, with a test that fails without it (a
   Wycheproof-style vector where possible).
3. A patch release of rustls-pure goes to crates.io (see "Releasing rustls-pure" in
   [DEVELOPMENT.md](../DEVELOPMENT.md)), and a Furnace release follows if Furnace's
   dependency needs raising.
4. The GitHub advisory is published, with a CVE requested through GitHub, crediting you unless
   you prefer otherwise; then it is filed with RustSec (below), so `cargo audit` and
   `cargo deny` flag the affected versions.

We aim to release a fix within 30 days of the report, sooner for anything exploitable.

## Supported versions

Fixes go into the latest release of rustls-pure: before 1.0, the latest minor version
(0.14.x now). There are no backports to older minor versions. rustls-pure has its own version,
apart from Furnace's.

## Filing a RustSec advisory

Whether a vulnerability was reported to us or we found it ourselves, once the fixed version is
on crates.io:

1. Fork [rustsec/advisory-db](https://github.com/rustsec/advisory-db) and add
   `crates/rustls-pure/RUSTSEC-0000-0000.md` (RustSec assigns the number when it merges):

   ````markdown
   ```toml
   [advisory]
   id = "RUSTSEC-0000-0000"
   package = "rustls-pure"
   date = "2026-01-31"          # the day the fix was released
   url = "https://github.com/Gilfeather/furnace/security/advisories/GHSA-xxxx-xxxx-xxxx"
   categories = ["crypto-failure"]
   keywords = ["tls"]
   aliases = ["CVE-2026-xxxxx", "GHSA-xxxx-xxxx-xxxx"]

   [versions]
   patched = [">= 0.14.2"]
   ```

   # One line saying what is wrong

   What an attacker can do, which versions are affected, and what to upgrade to.
   ````

   `categories` come from RustSec's list (`crypto-failure`, `denial-of-service`,
   `memory-exposure`, ...); add `unaffected` for versions that never had the bug.
2. Check it with `cargo install rustsec-admin` and `rustsec-admin lint` in the fork, then open
   the pull request.
3. When it is merged, add the RUSTSEC id to the GitHub advisory's references.

A flaw found in a dependency is filed against that crate in the same way, by its maintainers or
by us, after they have had the chance to fix it.
