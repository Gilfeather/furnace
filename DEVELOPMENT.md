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
  (`.github/changed-files.sh`, the pull request's diff from its merge base against the
  workflow's patterns) and runs its heavy jobs only when the pull request changes a file they
  check. Each ends in one gate job, `<Workflow> required`, which passes when every job passed
  or was skipped as not needed, and fails when one failed or was cancelled. A newer push to
  the same pull request cancels the older run.
- **Pushes to main** run nothing: what is merged has passed as a pull request. Only the
  release workflows run on tags.
- **Nightly on main** CI runs everything pull requests leave out (Windows and macOS, beta and
  the MSRV, Metal, the vendored kernels, the bench harnesses, furnace-candle, the Kubernetes
  manifests, coverage); Kani, TLA+ and Loom run nightly too, and `kani-nightly.yml` runs the
  Kani harnesses too long for a pull request. Miri, rustls-pure (with the newest rustls and its
  MSRV), Security Audit, KServe end-to-end, the constant-time tests, the soak test and the
  benchmarks run weekly. Scheduled runs are never cancelled. The README's CI badge shows the
  last nightly CI run on main.

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
# .github/workflows/soak.yml runs it for hours, by hand and weekly
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
