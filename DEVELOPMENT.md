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
cargo clippy -- -D warnings

# Code is formatted
cargo fmt --check

# No security vulnerabilities
cargo audit
```

## GitHub Branch Protection

To protect the main branch:

1. Go to **Settings** → **Branches** in GitHub
2. Add rule for `main` branch
3. Enable:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

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
cargo test

# Integration tests
cargo test --test integration_tests

# Benchmarks
cargo bench
```

## Release Process

1. Create release branch: `git checkout -b release/v1.0.0`
2. Update version in `Cargo.toml`
3. Update `CHANGELOG.md`
4. Create PR to main
5. After merge, create GitHub release
6. GitHub Actions will build and publish binaries