---
description: Conventions, rules and policies for MoonAI development.
---

# Standards

## Rust

### Toolchain

| Tool | Version | Config |
|------|---------|--------|
| Rust | 1.95.0 | `rust-toolchain.toml` |
| Clippy | (bundled) | `clippy.toml`, `Cargo.toml` |
| rustfmt | (bundled) | `rustfmt.toml` |

### Edition and MSRV

- **Edition**: 2024
- **MSRV**: 1.95.0
- **Resolver**: 2

### Code Style

- Max line width: 120 characters
- Unix newlines
- Imports and modules auto-sorted
- `use_field_init_shorthand`, `use_try_shorthand` enabled

### Lints

#### Rust Lints

| Lint | Level |
|------|-------|
| `elided_lifetimes_in_paths` | deny |
| `absolute_paths_not_starting_with_crate` | deny |
| `unsafe_code` | warn |
| `unused` | warn |

#### Clippy Lint Groups (All denied)

- `correctness`, `suspicious`, `complexity`, `perf`, `style`

#### Clippy Individual Lints (All Denied)

From restriction/nursery (not covered by groups above):

- `dbg_macro`, `expect_used`, `unwrap_used`, `panic`, `todo`
- `needless_collect`, `redundant_clone`, `large_stack_arrays`
- `missing_const_for_fn`, `option_if_let_else`
- `print_stdout`, `print_stderr`
- `clone_on_ref_ptr`, `rest_pat_in_fully_bound_structs`, `str_to_string`

#### Clippy Thresholds
- `too-many-arguments-threshold`: 12
- `cognitive-complexity-threshold`: 15
- `enum-variant-size-threshold`: 128
- `type-complexity-threshold`: 256

### Rules

#### Suppression Comments — Absolute Ban

- `#[allow(...)]` and `#![allow(...)]` are forbidden in all source files (`src/`)
- Enforced by ripgrep in the quality gate — only `tests/integration.rs` is exempt (crate-wide `expect_used`/`unwrap_used` for integration test infrastructure)
- `expect`/`unwrap` in test functions are permitted via `clippy.toml` settings (`allow-expect-in-tests`, `allow-unwrap-in-tests`), NOT via `#[allow]` attributes

### Toolchain

| Tool | Version | Config |
|------|---------|--------|
| Python | 3.14+ | `pyproject.toml`, `ruff.toml` |

## Rust Standards

### Workspace Lints (`Cargo.toml`)

These lints are **deny** at the workspace level and cannot be overridden locally.

#### Clippy Lints

| Lint | Level |
|------|-------|
| `correctness` | deny |
| `suspicious` | deny |
| `complexity` | deny |
| `perf` | deny |
| `style` | deny |

#### Explicitly Denied Lints

| Lint | Level |
|------|-------|
| `dbg_macro` | deny |
| `expect_used` | deny |
| `needless_collect` | deny |
| `panic` | deny |
| `redundant_clone` | deny |
| `redundant_closure_for_method_calls` | deny |
| `trivially_copy_pass_by_ref` | deny |
| `todo` | deny |
| `uninlined_format_args` | deny |
| `unwrap_used` | deny |
| `implicit_clone` | deny |
| `inefficient_to_string` | deny |
| `large_stack_arrays` | deny |
| `missing_const_for_fn` | deny |
| `needless_pass_by_value` | deny |
| `option_if_let_else` | deny |
| `print_stdout` | deny |
| `print_stderr` | deny |
| `clone_on_ref_ptr` | deny |
| `rest_pat_in_fully_bound_structs` | deny |
| `str_to_string` | deny |

### Clippy Configuration (`clippy.toml`)

| Setting | Value |
|---------|-------|
| MSRV | 1.95 |
| Avoid breaking exported API | true |
| Allow expect in tests | true |
| Allow unwrap in tests | true |
| Disallowed names | `foo`, `bar` |
| Too many arguments threshold | 15 |
| Cognitive complexity threshold | 15 |
| Enum variant size threshold | 128 |
| Type complexity threshold | 256 |
| Large error threshold | 256 |
| Source item ordering | `['enum', 'struct', 'trait']` |

### Rustfmt Configuration (`rustfmt.toml`)

| Setting | Value |
|---------|-------|
| Edition | 2024 |
| Style edition | 2024 |
| Max width | 120 |
| Use small heuristics | Max |
| Newline style | Unix |
| Hard tabs | false |
| Tab spaces | 2 |
| Reorder imports | true |
| Reorder modules | true |
| Remove nested parens | true |
| Use field init shorthand | true |
| Use try shorthand | true |

### Suppression Forbidden

**Suppression commands are forbidden.** This is enforced by the `check` recipe.

```bash
just check
```

Do NOT use:
- `#[allow(...)]`
- `#![allow(...)]`
- `#[expect(...)]` (use in tests only if clippy allows)

### Code Rules

1. **No unsafe code** — `unsafe_code` is warn by default; prefer safe abstractions
2. **No unwrap in production** — use `?`, `Option::ok()`, or `anyhow::Context`
3. **No print to stdout/stderr** — use `tracing::info!`, `tracing::warn!`, etc.
4. **No TODO in code** — resolve before committing
5. **Module item order** — enum → struct → trait
6. **No dead code** — unused code must be removed or marked with `#[allow(dead_code)]` only if truly necessary and documented why

### Release Profile (`Cargo.toml`)

```toml
[profile.release]
strip = true
lto = true
panic = 'abort'
incremental = true
codegen-units = 4
```

## Python Standards

### Python Version

- **Required**: Python 3.14+

### Ruff Configuration (`ruff.toml`)

| Setting | Value |
|---------|-------|
| Line length | 120 |
| Target version | py314 |
| Indent width | 4 |

#### Lint Rules

| Rule Set | Status |
|----------|--------|
| E (pycodestyle errors) | enabled |
| F (pyflakes) | enabled |
| I (isort) | enabled |
| UP (pyupgrade) | enabled |

- Fixable: ALL
- Unfixable: none

#### Format Rules

| Setting | Value |
|---------|-------|
| Quote style | double |
| Indent style | space |

## Cargo Workspace Conventions

### Crate Organization

Each crate MUST:
- Use `version.workspace = true`
- Use `edition.workspace = true`
- Use `authors.workspace = true`
- Use `license.workspace = true`

### Dependencies

- Workspace dependencies MUST use `{ workspace = true }` or inherit from `[workspace.dependencies]`
- Path dependencies for intra-workspace crates only
- External dependencies version-pinned in `[workspace.dependencies]`
- Build dependencies (`[build-dependencies]`) separate from runtime dependencies
