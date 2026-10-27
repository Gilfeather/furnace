# 🛠️ Development Guide

## Branch Strategy

We use a feature branch workflow to maintain code quality and enable collaborative development.

### Branch Naming Convention

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes  
- `docs/documentation-update` - Documentation changes
- `perf/performance-improvement` - Performance optimizations
- `refactor/code-cleanup` - Code refactoring
- `test/test-improvements` - Test additions/improvements

### Workflow

1. **Create a branch from main**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   ```bash
   # Make changes
   cargo test  # Ensure tests pass
   cargo clippy  # Check for lints
   cargo fmt  # Format code
   ```

3. **Commit with descriptive messages**
   ```bash
   git add .
   git commit -m "feat: add new inference optimization
   
   - Implement tensor caching for repeated inputs
   - Reduce memory allocations by 30%
   - Add benchmarks for performance validation"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   gh pr create --title "Add inference optimization" --body "Description of changes"
   ```

5. **After PR approval, merge and cleanup**
   ```bash
   git checkout main
   git pull origin main
   git branch -d feature/your-feature-name
   ```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## Code Quality Checks

Before submitting a PR, ensure:

```bash
# All tests pass
cargo test

# No clippy warnings
cargo clippy --all-targets -- -D warnings

# Code is formatted
cargo fmt --check

# No security advisories, and licenses, bans and sources as deny.toml allows (what CI runs;
# `cargo install cargo-deny`)
cargo deny check
```

## When CI Runs

- **Pull requests to main** run every verification workflow: CI, Kani, TLA+, Loom, Miri,
  rustls-pure, Security Audit and KServe end-to-end. Each starts with a `Changed files` job
  (`bash .github/changed-files.sh <workflow>`) that decides which of its jobs the pull request
  needs, from the diff since the pull request's merge base and the rules in
  `.github/relevance`. Each ends in one gate job, `<Workflow> required`, which passes when
  every job passed or was skipped as not needed, and fails when one failed or was cancelled. A
  newer push to the same pull request cancels the older run.
- **Pushes to main** run nothing: what is merged has passed as a pull request. Only the
  release workflows run on tags.
- **Nightly on main** CI runs everything pull requests leave out (Windows and macOS, beta and
  the MSRV, Metal, the vendored kernels, the bench harnesses, furnace-candle, the Kubernetes
  manifests, coverage). TLA+, Loom and `kani-nightly.yml` (the Kani harnesses too long for a
  pull request) run nightly, Miri weekly, each on what changed since its last successful
  scheduled run. None of these, nor the weekly soak test, runs twice on one commit of main:
  when the workflow already ran to the end on it, passed or failed, a scheduled run checks
  nothing, since the lockfile and pinned tools would give the same result and the earlier run
  shows it. Kani's pull request harnesses have no schedule: they were proved on the pull
  request that merged them. Weekly and in full, since what they check changes outside the
  repository: rustls-pure (the newest rustls, its MSRV, interoperability against the system's
  TLS tools), Security Audit (new advisories), KServe end-to-end (downloads the release and
  KServe), the constant-time tests (with the machine-code check; the compiler's code may change).
  Scheduled runs are never cancelled. The README's CI badge shows the last nightly CI run on
  main.
- **By hand only:** the benchmarks (`bench.yml`); nothing measures on a schedule or a release.

### Kani proofs not run in CI

Five Kani proofs are not run in CI until they are decomposed: they never completed on GitHub
runners (cut at the six-hour limit or by a runner shutdown, with no counterexample). Their code
stays, and `cargo kani -p <package> --harness <name> --exact -Z function-contracts -Z stubbing`
runs them locally. Kani's `Every Kani harness runs` check lists them as parked, so no other proof
can drop out of CI unnoticed.

- kamui, at the allocator's 64 slots (their 4-slot versions run on every pull request):
  `verification::take_hands_out_a_block_once_64_slots`,
  `verification::keep_keeps_a_block_or_refuses_it_64_slots`,
  `verification::alloc_and_dealloc_stay_in_bounds_64_slots`,
  `verification::realloc_keeps_the_contents_64_slots`.
- furnace-server: `tensor::verification::concat_batch_never_panics` (`joined_batch`'s contract,
  which it builds on, runs on every pull request).

### Which changes run which jobs

`.github/relevance` holds the rules, one line per workflow and output: path patterns in the
syntax of GitHub's `paths` filters, and `crate:<package>`. The rules follow what each job
reads:

- **Docs, licences and changelogs run nothing**, except in CI: the Markdown files whose Rust
  examples compile as doc-tests (README.md, CONTRIBUTING.md, examples/README.md, docs/*.md)
  and docs/openapi.json, which the server's tests compare with its routes.
- **Each check counts only the files it reads.** Kani a package's proofs: `rustls-pure/src`,
  `kamui/src`, and furnace-server's files with proofs (`src/tensor.rs`, `src/model.rs`,
  `src/api/infer.rs`, `src/json_fast.rs`, `src/nerve/wire.rs`) and `build.rs`. Miri
  `kamui/src` and `kamui/tests`. Loom `amate/src` and `src/nerve`, each for its own models. TLA+
  `specs/`. rustls-pure's machine-code check `rustls-pure/src` and `ct-asm.sh`; its
  interoperability tests `rustls-pure/src`, `rustls-pure/examples` and `rustls-pure/interop`.
  CI everything else in the workspace, but other workflows' files, `specs/`, `deploy/`,
  `benches/` (workspaces of their own, which the nightly run checks), `dev/` and packaging.
  cargo-deny the lockfile, the workspace's manifests and `deny.toml`; CodeQL Rust code.
- **Cargo.toml and Cargo.lock count by what a package is built from.** For `crate:<package>`,
  `.github/dependency-tree.py` lists the package's resolved dependencies from the lockfile and
  the manifests that decide its build (its own and those of the workspace packages it reaches,
  and the workspace's profiles, lints and patches) at both commits, without the workspace's
  version numbers and the manifests' descriptive fields. A release's version bump, or an
  update of a dependency the package does not reach, leaves it unchanged.
- **A dependency update counts for the checks that run its code.** A check that runs only some
  of a package's dependencies names them, `crate:<package>=<crate>,...` (an empty list: only
  the package's own code), read off the code it runs:

  | Check | Dependencies it runs |
  | --- | --- |
  | Kani kamui, Kani furnace-server, rustls-pure's machine-code check | none: their proofs and checked functions call std only |
  | Kani rustls-pure | `rustls` (SharedSecret) and `zeroize` (its wiping) |
  | Loom amate | `loom` |
  | Loom furnace-server | `loom`, `tokio` (the nerve's channels), `bytes`, `tracing` |
  | Miri | all of kamui's: its tests run proptest and its random number generator |
  | rustls-pure's interoperability tests and target builds, CI's tests (Wycheproof among them) | all of rustls-pure's, or the workspace's |
  | rustls-pure's constant-time tests | all of rustls-pure's |

  So a crypto-bigint update runs rustls-pure's interoperability, target builds and timing
  tests and CI, and no proof; a tokio update runs the nerve's Loom models and CI. A Kani
  version change is in the workflow file, which counts for all of its jobs. A crate the list
  names that the package no longer reaches fails the run, so the lists cannot go stale
  quietly; a proof that starts calling a new dependency needs it added.
- **The constant-time tests run on a pull request only when rustls-pure's dependencies
  change** (a crypto crate's security update is timed before it is merged), and weekly. Not on
  code changes: timing on shared runners is slow and noisy, and the machine-code check and the
  proofs cover rustls-pure's own code. They have no `required` check: a noisy runner can fail
  the control, so the result shows on the pull request without blocking it.
- **The deploy manifests count by more than their versions.** KServe end-to-end runs when
  `deploy/kserve` changes other than in its version numbers (`~` in the rules): a release's
  new image tag and download URL name what is published only after the merge.
- A workflow's own file and the three relevance files count for all of its jobs; another
  workflow's file counts for none.

`bash .github/changed-files-test.sh` checks the rules against example pull requests (which
outputs each must turn on); CI runs it when they change. Change the rules and the examples
together.

## GitHub Branch Protection

`main` requires the gate jobs to pass before a pull request is merged (Settings → Branches →
`main` → Require status checks to pass): `CI required`, `Kani required`, `TLA+ required`,
`Loom required`, `Miri required`, `rustls-pure required`, `Security Audit required` and
`KServe end-to-end required`. The branch need not be up to date with main first.

## Local Development

### Prerequisites
- Rust 1.70+
- Git
- GitHub CLI (optional but recommended)

### Setup
```bash
git clone https://github.com/Gilfeather/furnace.git
cd furnace
cargo build
cargo test
```

### Running the Server
```bash
# Create sample model
cargo run --bin create_sample_model

# Start server
cargo run --bin furnace -- --model-path ./sample_model --port 3000
```

### Testing
```bash
# Unit tests
cargo test --release

# Integration tests
cargo test --release --test integration_tests

# Soak run: the server and the nervous system under a steady load (here for 10 minutes);
# .github/workflows/soak.yml runs it for hours, by hand and weekly (once per commit)
FURNACE_SOAK_SECS=600 cargo test --profile ci --test soak -- --ignored --nocapture

# Benchmarks (furnace-burn's criterion bench; `onnx` builds in furnace-burn/models/*.onnx)
cargo bench -p furnace-burn --features onnx
```

## Release Process

1. Create release branch: `git checkout -b release/v1.0.0`
2. Update the version in `Cargo.toml`, `amate/Cargo.toml`, `kuebiko/Cargo.toml`, `kamui/Cargo.toml`, `furnace-burn/Cargo.toml` and `furnace-cli/Cargo.toml` (and their dependencies on each other). Not rustls-pure's: it has its own (below)
3. Update `CHANGELOG.md`
4. Create PR to main
5. After merge, create GitHub release
6. GitHub Actions will build and publish binaries

The release workflow (on a `v*` tag) publishes amate, kuebiko, kamui, rustls-pure,
furnace-server, furnace-burn and furnace-cli to crates.io in that order, through crates.io's
Trusted Publishing (no token stored in the repository): amate, kuebiko, kamui and rustls-pure
only when the repository variable `PUBLISH_PART_CRATES` is `true`, furnace-burn and furnace-cli
only when `PUBLISH_BURN_CRATES` is. A crate whose version is on crates.io already is skipped
(`.github/publish-crate.sh`), so a version published by hand before the tag does not fail the
release. furnace-server depends on amate, kuebiko and rustls-pure (with `tls`), furnace-cli on
kamui: their versions must be on crates.io before the workflow reaches those crates, published
by the workflow itself or by hand.

### Releasing rustls-pure

rustls-pure has a version of its own, apart from the other crates, and is released only when
it changes (a fix, a new minimum Rust or rustls version, a README that crates.io should show):

1. In a PR: raise `version` in `rustls-pure/Cargo.toml` (before 1.0, a patch for fixes and
   docs, a minor for anything breaking, a higher MSRV or a new rustls minor version), and
   move its entries in `rustls-pure/CHANGELOG.md` under that version with the date. If
   Furnace needs the new version, raise furnace-server's `rustls-pure` dependency too.
2. Merge, then tag that `main` commit and push the tag:
   `git tag rustls-pure-v0.14.1 && git push origin rustls-pure-v0.14.1`.
3. `.github/workflows/rustls-pure-release.yml` checks that the tag matches the version, runs
   the tests and publishes it through Trusted Publishing (`.github/publish-crate.sh` skips a
   version already on crates.io).

The furnace release workflow also publishes rustls-pure, but only a version not on crates.io
yet, so a Furnace release does not bump or republish it. Trusted Publishing for rustls-pure
needs two entries on crates.io (below): workflow `release.yml` and workflow
`rustls-pure-release.yml`. A vulnerability fix follows
[rustls-pure/SECURITY.md](rustls-pure/SECURITY.md).

### Publishing a new crate for the first time

Trusted Publishing works only for a crate that is already on crates.io, so a new crate's first
version is published by hand, by an owner, with an API token (crates.io → Account Settings → API
Tokens, scope `publish-new`). Whether that comes before or after the tag depends on where the
new crate sits:

- **A crate that furnace-server or furnace-cli depends on** (amate, kuebiko and kamui in
  0.13.0, rustls-pure in 0.14.0): by hand *before* the tag, or the workflow fails at
  furnace-server. A part that depends on nothing else of the workspace, such as rustls-pure,
  can instead be published at its current version as soon as it is on `main` (steps 2 to 4).
  1. Merge the release PR (versions bumped).
  2. `cargo login` with the token, then, from an up-to-date `main`:
     `cargo publish --locked -p amate`, `cargo publish --locked -p kuebiko`,
     `cargo publish --locked -p kamui` (for rustls-pure: `cargo publish --locked -p rustls-pure`).
  3. Trusted Publishing for each (below), then
     `gh variable set PUBLISH_PART_CRATES --body true` (already set since 0.13.0).
  4. Revoke the API token (`cargo logout`, and delete it on crates.io).
  5. Push the `v<version>` tag on that `main` commit. The workflow skips the versions already
     on crates.io and publishes the rest.
- **A crate that depends on furnace-server** (furnace-burn and furnace-cli in 0.12.0): by hand
  *after* the tag.
  1. Wait until the release workflow has published furnace-server of the new version.
  2. `cargo login` with the token, then, from a checkout of the release tag:
     `cargo publish -p furnace-burn`, and once it is on crates.io, `cargo publish -p furnace-cli`.
  3. Trusted Publishing for each (below), then
     `gh variable set PUBLISH_BURN_CRATES --body true`.
  4. Revoke the API token.

Trusted Publishing on crates.io, for each new crate: Settings → Trusted Publishing → Add →
GitHub, with repository owner `Gilfeather`, repository name `furnace`, workflow filename
`release.yml` and no environment (rustls-pure: a second entry with `rustls-pure-release.yml`).
