---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Migration Plan: MoonAI C++/SFML → Rust/winit+egui+wgpu

## Overview

Migrate all C++ host code to Rust. Replace SFML visualization with winit + egui + wgpu. Keep CUDA `.cu` files as-is; Rust compiles them via `build.rs`.

## Assumptions

- `config.lua` remains the config format.
- Output schema stays unchanged so Python analysis keeps working.
- Behavioral parity is the goal.

## Target Architecture

```
Cargo.toml
crates/
  moonai_cuda_sys/
    build.rs
    src/lib.rs
    cuda/
      buffers.cu
      batch.cu
      inference_cache.cu
      ffi.cuh
src/
  main.rs
  cli.rs
  app.rs
  config.rs
  lua_config.rs
  random.rs
  types.rs
  state.rs
  simulation/
  evolution/
  metrics/
  ui/
  render/
tests/
```

### CUDA Boundary

- `moonai_cuda_sys` compiles `.cu` files in `build.rs`.
- A narrow `extern "C"` ABI is exposed. All internal CUDA ownership stays inside `.cu`.
- Manual `#[repr(C)]` structs plus opaque handles — **do not use `bindgen`**.

### UI Replacement Stack

- **winit** — event loop, window, input
- **wgpu** — world rendering (predators/prey/food via GPU instancing)
- **egui** — overlay panels, controls, charts, NN topology panel

> **Important:** Current defaults are large (24k predators, 96k prey, 240k food). Render world geometry with wgpu instancing only. Do not attempt to draw world entities through egui shapes.

## Phases

### Phase 1 — Add Cargo Workspace and CUDA Crate

1. Introduce `Cargo.toml` with a workspace containing `moonai_cuda_sys`.
2. Write `moonai_cuda_sys/build.rs` to compile `buffers.cu`, `batch.cu`, `inference_cache.cu`.
3. Keep CMake as a temporary fallback until Rust headless runs successfully.

### Phase 2 — Define Rust/CUDA ABI

1. Create `ffi.cuh` in the CUDA crate plus matching Rust `#[repr(C)]` definitions.
2. Convert CUDA code to export C ABI functions for:
   - Buffer lifecycle and capacity management
   - Sensor build launch
   - Inference launch
   - Post-inference step
   - Synchronization
   - Error/result codes
3. Do **not** expose internal C++ classes (`simulation::Batch`, `evolution::InferenceCache`) to Rust.
4. Expose only opaque handles (`MoonaiCudaBatch`, `MoonaiCudaInferenceCache`) and plain ABI structs (`StepParams`, compiled network views).

### Phase 3 — Port Core/Runtime Infrastructure

Port in this order (no internal dependencies until both sides exist):

- `types` — Vec2, INVALID_ENTITY, SENSOR_COUNT, OUTPUT_COUNT
- `config` — SimulationConfig defaults, validation, JSON snapshot
- `lua_config` — Lua loading with `moonai_defaults` injection, `table_to_config`
- `cli` — argument parsing (all current CLI flags preserved)
- `logger` — file setup, stats/genomes/species CSV+JSON output
- `random` — mt19937_64, next_int/float/gaussian/bool, weighted_select

### Phase 4 — Port Simulation State Model

- `Food` — initialize, respawn_step (deterministic respawn preserved)
- `AgentRegistry` — create, valid, size, compact, find_by_agent_id, swap_entities, pop_back, resize
- `MetricsSnapshot` — same fields, populated by `metrics::refresh`
- `AppState` — UiState, predator/prey registries, food, runtime state, step buffers
- Keep SoA layout (`Vec<T>` for each field) to maintain CUDA handoff compatibility

### Phase 5 — Port Evolution CPU Logic

Port and **test first**:

- `Genome` — nodes, connections, add_node, add_connection, has_connection, has_node, max_node_id, complexity, compatibility_distance, to_json
- `InnovationTracker` — get_innovation, get_split_node_id, init_from_population
- `Mutation` — mutate_weights, add_connection, add_node
- `Crossover` — crossover
- `Species` — compatibility check, add_member, refresh_summary
- `NeuralNetwork` — activate, forward pass
- `NetworkCache` — assign, get, move_entity, remove, clear
- `EvolutionManager` — initialize, seed_initial_population, create_initial_genome, create_child_genome, reproduce_population, DenseReproductionGrid, refresh_species

Port `tests/test_evolution.cpp` tests to Rust first. Make them pass before wiring full runtime.

### Phase 6 — Port Headless Step Pipeline

Recreate `App::step` from `src/app/app.cpp` and `src/simulation/simulation.cpp`:

```
prepare_step
  → pack_state
  → batch.upload_async
  → batch.launch_build_sensors_async

run_inference
  → predator inference launch
  → prey inference launch

resolve_step
  → batch.launch_post_inference_async
  → batch.download_async
  → batch.synchronize
  → apply_results
  → collect_step_events

post_step
  → predator.compact
  → prey.compact
  → food.respawn_step
```

Keep logging/output behavior identical.

### Phase 7 — First Milestone: Headless Rust

Before touching GUI, Rust must successfully:

1. `cargo test` — evolution and config tests pass.
2. `cargo build` — Rust + CUDA compile cleanly without CMake.
3. `cargo run -- --validate config.lua` — matches current behavior.
4. Run a small seeded experiment headless and produce:
   - `config.json`
   - `stats.csv`
   - `species.csv`
   - `genomes.json`

### Phase 8 — Replace SFML with winit + wgpu + egui

Port input semantics from `src/visualization/visualization_manager.cpp`:

| Key/Action | Behavior |
|------------|----------|
| Space | Pause / resume |
| ↑ / ↓ or + / - | Increase / decrease simulation speed |
| . | Step one frame (while paused) |
| S | Save screenshot |
| Esc | Quit |
| Left-click | Select agent → show stats + NN panel |
| Right-click drag | Pan camera |
| Scroll wheel | Zoom |

Rebuild overlay in egui. Render world geometry with wgpu instanced drawing. Preserve:
- Left/right panel layout
- Stats panel (step, population, births/deaths, species counts)
- Population / complexity / energy charts (egui plots)
- NN topology panel for selected agent (node/edge graph)
- Energy distribution bars

### Phase 9 — Remove Remaining C++ Host Code

Once Rust headless and GUI both work:

1. Delete `src/main.cpp`, `src/app/`, `src/core/*.cpp`, `src/simulation/simulation.cpp`, `src/evolution/*.cpp` (except `inference_cache.cu`), `src/metrics/*.cpp`, `src/visualization/`.
2. Remove CMake `moonai_app`, `moonai_core`, `moonai_simulation`, `moonai_evolution`, `moonai_metrics`, `moonai_visualization` targets.
3. Remove `src/*/CMakeLists.txt` files.
4. Remove `src/` CMake add_subdirectory calls from root `CMakeLists.txt`.
5. Delete `src/` directory entirely.
6. Keep only Rust source tree + CUDA crate + Python analysis package.

## Verification Gates

| Gate | Command | Success Criteria |
|------|---------|-----------------|
| Core parity | `cargo test` | All evolution/config tests pass |
| Build parity | `cargo build` | Rust + CUDA build without CMake |
| Config parity | `cargo run -- --validate config.lua` | Output matches current behavior |
| Headless runtime parity | Run small seeded experiment in both | `stats.csv`/`species.csv` shape and trends match |
| GUI smoke test | Open window, interact | Pause, step, speed, zoom, pan, select, charts all work |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `build.rs` CUDA arch selection less convenient than CMake `CUDA_ARCHITECTURES native` | Drive arch via `MOONAI_CUDA_ARCHS` env var |
| RNG parity across language boundary | Use `mt19937_64` seeded identically; do not assume exact float distribution parity — compare trends not bit-exactness |
| Egui rendering world entities | Use wgpu instancing for all agent/food drawing; egui only for panels |
| Unsafe FFI boundary grows unbounded | Keep ABI narrow; all host ↔ GPU traffic goes through the two exported handles |
| Large population defaults (24k+96k+240k) | wgpu instancing required from day one; do not try egui for world drawing |

## Implementation Order for Next Prompt

2. Phase 1 — Add Cargo workspace and `moonai_cuda_sys` crate.
3. Phase 2 — Introduce CUDA C ABI and `build.rs`.
4. Phase 3 (partial) — Port CLI/config/logger scaffolding to Rust.

This establishes the permanent build/runtime boundary before touching any large host-side modules.
