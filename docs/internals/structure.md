---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Structure

```
moonai/
├── .github/                    # GitHub workflows
├── analysis/                   # Python simulation analysis package
├── assets/                     # assets (fonts, logo)
├── docs/                       # Documentation source
├── legacy/                     # Legacy C++ implementation (read-only, for reference)
├── crates/                     # Rust workspace (moonai-*)
├── tests/                      # Google Test unit tests
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── Cargo.toml                  # Rust workspace manifest (workspace config, lints)
├── Cargo.lock                  # Locked dependency versions
├── clippy.toml                # Clippy linter configuration
├── rustfmt.toml               # Rust formatter configuration
├── rust-toolchain.toml        # Rust toolchain specification
├── ruff.toml                  # Ruff linter configuration for Python
├── README.md                   # Project readme
├── config.lua                  # Unified config: default run + experiment matrix
├── justfile                    # Rust project commands
├── pyproject.toml             # Python package config (hatchling build)
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

## Legacy C++ Implementation (`legacy/`)

The `legacy/` directory contains the **original C++ implementation** of MoonAI. This codebase is **frozen and read-only** — it serves as a reference for understanding the original design and can be consulted during the Rust rewrite but is no longer actively maintained.

### Legacy Contents

| File/Directory      | Purpose                                              |
| ------------------- | ---------------------------------------------------- |
| `CMakeLists.txt`    | Root CMake configuration                             |
| `CMakePresets.json` | Build presets for Linux/Windows                      |
| `.clang-format`     | LLVM code style configuration                        |
| `.clang-tidy`       | Static analysis configuration                        |
| `vcpkg.json`        | vcpkg dependency manifest                            |
| `justfile-cpp`      | C++ build commands (`just -f legacy/justfile-cpp`)   |
| `architecture.md`   | System architecture diagrams and design notes        |
| `main.cpp`          | C++ entry point                                      |
| `app/`              | Application orchestration, main loop                 |
| `core/`             | Types, config, Lua runtime, seeded RNG               |
| `evolution/`        | NEAT genome, neural network, speciation              |
| `metrics/`          | CSV/JSON logging, aggregation                        |
| `simulation/`       | ECS-based simulation (agents, physics, spatial grid) |
| `visualization/`    | SFML rendering, UI overlay                           |

## Rust Workspace (`crates/`)

The Rust rewrite lives in `crates/` and implements a GPU-first architecture:

```
crates/
├── moonai-types/               # Core types (Vec2, NodeGene, ConnectionGene, etc.)
├── moonai-config/              # SimulationConfig, CLI args, Lua loading
├── moonai-evolution/           # NEAT algorithm, CUDA kernels
├── moonai-simulation/          # GPU simulation, persistent kernel
├── moonai-metrics/            # CSV/JSON logging
├── moonai-ui/                  # wgpu rendering, egui overlay
└── moonai/                     # Binary crate, signal handling
```

## `analysis/`

| File             | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `__main__.py`    | CLI entry point (`uv run analysis`)      |
| `pipeline.py`    | Orchestrates the analysis run            |
| `io.py`          | Run discovery, CSV/JSON loading          |
| `labels.py`      | Groups runs into experiment conditions   |
| `plots.py`       | Generates embedded matplotlib figures    |
| `genome.py`      | Renders neural network topology diagrams |
| `summary.py`     | Prepares summary statistics              |
| `html_report.py` | Renders self-contained HTML document     |
| `report.html`    | Jinja2 HTML report template              |

## Documentation (`docs/`)

| Path                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `_assets`                | Documentation assets, extra.css, extra.js, and reports |
| `index.md`               | Documentation home                                     |
| `usage.md`               | Usage guide and CLI reference                          |
| `about.md`               | Project overview and motivation                        |
| `installation.md`        | Build and installation instructions                    |
| `reports.md`             | Links to project reports                               |
| `internals/roadmap.md`   | Tasks, bugs, and roadmap                               |
| `internals/structure.md` | This file                                              |
| `internals/workflow.md`  | Development workflow                                   |
| `internals/standarts.md` | Coding standards                                       |
