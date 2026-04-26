---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

```
moonai/
├── .github/                    # GitHub workflows
├── analysis/                   # Python simulation analysis package
├── docs/                       # Documentation source
├── crates/                     # Rust workspace (moonai-*)
├── runtime/                    # Runtime assets (config/, assets/)
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── Cargo.toml                  # Rust workspace manifest (workspace config, lints)
├── Cargo.lock                  # Locked dependency versions
├── clippy.toml                # Clippy linter configuration
├── rustfmt.toml               # Rust formatter configuration
├── rust-toolchain.toml        # Rust toolchain specification
├── ruff.toml                  # Ruff linter configuration for Python
├── README.md                   # Project readme
├── justfile                    # Rust project commands
├── pyproject.toml             # Python package config (hatchling build)
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

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
