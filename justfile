# MoonAI - Rust Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.


# Set up Python environments for simulation and analysis
[group('build')]
setup-uv:
  uv sync

# Build all crates in the workspace
[group('build')]
build-debug:
  cargo build --workspace

# Build in release mode
[group('build')]
build:
  cargo build --workspace --release


# Run the release build with default config (pass additional args after --)
[default]
[group('run')]
run *args:
  cargo run --release -- {{args}}

# Run the debug build with default config (pass additional args after --)
[group('run')]
run-debug *args:
  cargo run -- {{args}}

# Generate the self-contained HTML analysis report from output/
[group('run')]
analyse:
  uv run analysis


# Fix: format and lint
[group('dev')]
fix:
  cargo fmt --all
  cargo clippy --workspace --all-targets --all-features --fix --allow-dirty

  ruff check . --fix
  ruff format .

# Check code: format, lint checks and manual supression command grep
[group('dev')]
check:
  ! rg -n -F -e '#[allow' -e '#![allow' -g '*.rs' -g '!tests/**'
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features

  ruff format . --check
  ruff check .

# Run tests
[group('dev')]
test *args:
  cargo test --workspace --all-targets --all-features --locked -- --nocapture {{args}}

# Lint code
[group('dev')]
gate: check test

update:
  cargo update


# Remove build artifacts
[group('clean')]
clean:
  cargo clean
  ruff clean

# Remove all output and generated report artifacts
[group('clean')]
clean-outputs:
  rm -rf output/


# Clean and start website at localhost
[group('docs')]
docs:
  rm -rf site/
  zensical serve
