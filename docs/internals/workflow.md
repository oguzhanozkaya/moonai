---
description: Guide to set up, build, test, and deploy the project.
---

# Workflow

## Use `justfile` Commands

**All development commands MUST go through the `justfile`.** Do not run raw commands or tooling directly if any just recipe exists.

```bash
# Correct
just build
just test
just check

# Incorrect
cargo build --workspace
cargo test --workspace
cargo fmt --all
```

The `justfile` ensures consistent behavior across all developers and CI environments.

---

## Setup

### One-time Setup

```bash
# Install Python dependencies for analysis
just setup-uv
```

### Verify Setup

```bash
just check
```

---

## Development Commands

### Building

| Command            | Purpose                        |
| ------------------ | ------------------------------ |
| `just build-debug` | Debug build (fast iteration)   |
| `just build`       | Release build (optimized)      |
| `just run`         | Run release binary with config |
| `just run-debug`   | Run debug binary with config   |

### Code Quality

| Command      | Purpose                                    |
| ------------ | ------------------------------------------ |
| `just fix`   | Auto-fix formatting and lint issues        |
| `just check` | Verify code quality (format, clippy, ruff) |
| `just gate`  | Full quality gate: `check` + `test`        |

**`just check`** runs:

1. Suppression grep (no `#[allow]` anywhere)
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets --all-features`
4. `ruff format . --check`
5. `ruff check .`

**`just fix`** runs:

1. `cargo fmt --all`
2. `cargo clippy --fix`
3. `ruff format .`
4. `ruff check --fix`

### Testing

| Command                  | Purpose                    |
| ------------------------ | -------------------------- |
| `just test`              | Run all workspace tests    |
| `just test -- <pattern>` | Run tests matching pattern |

### Output Management

| Command              | Purpose                               |
| -------------------- | ------------------------------------- |
| `just clean`         | Remove build artifacts (`target/`)    |
| `just clean-outputs` | Remove simulation outputs (`output/`) |

### Analysis

| Command        | Purpose                                      |
| -------------- | -------------------------------------------- |
| `just analyse` | Generate HTML analysis report from `output/` |

### Documentation

| Command     | Purpose                          |
| ----------- | -------------------------------- |
| `just docs` | Serve documentation site locally |

---

## Quality Gate (CI/CD)

Before any commit or pull request:

```bash
just gate
```

This runs `just check` followed by `just test`. No exceptions.

---

## Rust Crate Development

When creating or modifying crates in `crates/`:

1. **Use workspace dependencies** — declare shared deps in `[workspace.dependencies]` and reference with `{ workspace = true }`
2. **Follow module structure** — `pub mod`, `pub use` pattern
3. **No suppression** — never use `#[allow(...)]` to silence lint errors

---

## Python Analysis Tools

The `analysis/` directory contains Python tools for generating reports from simulation output.

```bash
# Setup
just setup-uv

# Generate report
just analyse
```

---

## Common Development Workflow

```bash
# 1. Create a new branch
git checkout -b feature/my-feature

# 2. Make changes to code

# 3. Fix any formatting/lint issues
just fix

# 4. Run quality gate
just gate

# 5. Commit (if gate passes)
git add .
git commit -m "描述"

# 6. Push and create PR
git push -u origin feature/my-feature
```
