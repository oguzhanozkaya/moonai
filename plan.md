# MoonAI Rust Migration Plan

## Overview

This document describes the incremental migration of MoonAI from C++ to Rust, maintaining bit-for-bit reproducibility and keeping C++ CUDA kernels via cxx FFI.

**Strategy**: Incremental port, module by module, validating at each step.

**What stays C++**:
- CUDA kernels (`batch.cu`, `inference_cache.cu`, `buffers.cu`) - called via cxx bridge
- SFML visualization (optional, can be ported later)

**Estimated split**: ~70% Rust, ~30% C++ (GPU + optional viz)

---

## Phase 1: Infrastructure

**Goal**: Establish Rust build alongside C++, prove cxx FFI interop works.

### Tasks

1. Create Cargo workspace structure:
   ```
   rust/
   ├── moonai-ffi/        # cxx bridge crate
   ├── moonai-core/       # Core types, config, RNG, state
   ├── moonai-evo/        # NEAT evolution
   ├── moonai-metrics/    # Logging and metrics
   └── moonai-app/        # Application orchestration
   ```

2. Set up `moonai-ffi` crate with cxx dependency

3. Define initial FFI boundary: expose C++ RNG functions to Rust

4. Create `build.rs` for cxx code generation

5. Verify: Call C++ RNG from Rust, produce identical sequence

### Validation

```bash
# Baseline (C++)
./build/linux-release/moonai --seed 42 --steps 100 --headless

# Rust (calls C++ RNG via cxx)
./rust/target/release/moonai --seed 42 --steps 100 --headless

# Compare outputs - must be identical
```

---

## Phase 2: Core Primitives

**Goal**: Port `src/core/` to Rust.

### Tasks

1. **Port types** (`types.hpp` → `types.rs`)
   - `Vec2` struct
   - Constants: `SENSOR_COUNT = 35`, `OUTPUT_COUNT = 2`, `INVALID_ENTITY`

2. **Port random** (`random.cpp/hpp` → `random.rs`)
   - MT19937-64 implementation
   - Must produce **exactly** the same sequence for a given seed

3. **Port config** (`config.hpp/cpp` → `config.rs`)
   - `SimulationConfig` struct with all fields
   - JSON serialization/deserialization (serde, serde_json)
   - Validation logic

4. **Port Lua runtime** (`lua_runtime.cpp/hpp` → `lua.rs`)
   - Use `mlua` crate instead of `sol2`
   - Load `config.lua`, parse into SimulationConfig
   - **Critical**: Must parse identically to C++ version

5. **Port profiler macros** (`profiler_macros.hpp` → `profiler_macros.rs`)
   - `MOONAI_PROFILE_SCOPE` - no-op in non-profiler builds

### Dependencies

- `serde`, `serde_json` for config
- `mlua` for Lua parsing
- `log`, `spdlog` or `tracing` for logging

### Validation

- Load `config.lua` and compare parsed `SimulationConfig` byte-for-byte
- Run 100 steps with fixed seed, compare all agent positions/energies

---

## Phase 3: NEAT Evolution

**Goal**: Port `src/evolution/` to Rust.

### Tasks

1. **Port genome** (`genome.hpp/cpp` → `genome.rs`)
   - `NodeGene` struct (id, node_type)
   - `ConnectionGene` struct (in_node, out_node, weight, enabled, innovation)
   - `Genome` struct with methods: `add_node`, `add_connection`, `has_connection`, `has_node`, `compatibility_distance`
   - Innovation tracking

2. **Port mutation** (`mutation.hpp/cpp` → `mutation.rs`)
   - Weight perturbation
   - Add connection mutation
   - Add node mutation (split connection, create new node)
   - Delete connection mutation
   - `InnovationTracker` struct

3. **Port crossover** (`crossover.hpp/cpp` → `crossover.rs`)
   - Multi-point crossover
   - Disabled gene handling (75% chance)
   - Matching/non-matching gene logic

4. **Port species** (`species.hpp/cpp` → `species.rs`)
   - Compatibility distance formula
   - Species struct and management
   - `refresh_species` logic

5. **Port neural network** (`neural_network.hpp/cpp` → `network.rs`)
   - `NeuralNetwork` struct
   - CPU evaluation with topological sort
   - tanh activation
   - `CompiledNetwork` for GPU-ready representation

6. **Port network cache** (`network_cache.hpp/cpp` → `cache.rs`)
   - `NetworkCache` - NeuralNetwork per agent
   - `CompiledNetwork` array management
   - Entity-to-cache mapping

7. **Port inference cache** (`inference_cache.hpp/cu` → partially in `cache.rs`)
   - GPU inference launching (via cxx to C++)
   - `NetworkDescriptor` management
   - Repacking logic when capacity exhausted

8. **Port evolution manager** (`evolution_manager.hpp/cpp` → `evolution.rs`)
   - `seed_initial_population`
   - `run_inference` (via cxx to GPU)
   - `post_step` - reproduction with crossover + mutation
   - `refresh_species`

### Dependencies

- Phase 2 core (types, config, random)

### Validation

- Run evolution for N generations with fixed seed
- Compare:
  - Same number of genomes
  - Same species assignments
  - Same genome structures (node/connection counts)
  - Same neural network outputs for identical inputs

---

## Phase 4: Metrics and Logging

**Goal**: Port `src/metrics/` to Rust.

### Tasks

1. **Port metrics** (`metrics.hpp/cpp` → `metrics.rs`)
   - `MetricsSnapshot` struct
   - `refresh` function - compute population statistics
   - Aggregations: predator_count, prey_count, avg_complexity, avg_energy, etc.

2. **Port logger** (`logger.hpp/cpp` → `logger.rs`)
   - `Logger` struct
   - Write `stats.csv` (use `csv` crate)
   - Write `genomes.json` (serde_json)
   - Write `species.csv`

### Dependencies

- Phase 2 core (config)

### Validation

- Compare output files line-by-line after 100 steps
- Float values must match within epsilon (1e-6)

---

## Phase 5: Agent Registry and Simulation State

**Goal**: Port `src/core/app_state.hpp/cpp` to Rust.

### Tasks

1. **Port AgentRegistry** (`app_state.hpp` → `state/agent.rs`)
   - Structure-of-Arrays layout for cache efficiency
   - Entity management (create, destroy, compact)
   - `move_entity`, `swap_remove_entity` operations

2. **Port Food struct** (`app_state.hpp` → `state/food.rs`)
   - Food position, active state
   - Respawn logic

3. **Port AppState** (`app_state.hpp` → `state/mod.rs`)
   - Compose: `AgentRegistry<predator>`, `AgentRegistry<prey>`, `Food`
   - `MetricsSnapshot`, `RuntimeState`, `StepBuffers`
   - `UiState` (paused, speed, selected_agent)

4. **Port deterministic respawn** (`deterministic_respawn.hpp` → `state/respawn.rs`)
   - Food respawn logic that depends only on step count and RNG state

### Dependencies

- Phase 2 (core types, random, config)
- Phase 4 (metrics)

### Design Decision

Keep SoA layout for agent data (positions, velocities, energy as separate vectors) for CUDA compatibility and cache efficiency. This mirrors the C++ design.

### Validation

- Run 100 steps, compare:
  - Same predator/prey counts
  - Same energy distributions
  - Same positions (within float epsilon)

---

## Phase 6: FFI Integration and Application Orchestration

**Goal**: Wire Rust modules together, bridge to C++ GPU code via cxx.

### Tasks

1. **Define cxx bridge interface** (`moonai-ffi/src/lib.rs`)
   ```rust
   #[cxx::bridge]
   mod ffi {
       unsafe extern "C++" {
           include!("batch.cuh");
           include!("inference_cache.cuh");

           // Buffer management
           fn upload_async(state: &AppState);
           fn download_async(state: &mut AppState);

           // GPU kernels
           fn build_sensors_cuda(state: &mut AppState);
           fn run_inference_cuda(state: &mut AppState);
           fn resolve_step_cuda(state: &mut AppState);
       }
   }
   ```

2. **Port App** (`app.hpp/cpp` → `app.rs`)
   - Constructor: initialize all subsystems
   - `step()` - main loop iteration
   - `run()` - application entry point
   - `record_and_log()` - metrics recording

3. **Port main** (`main.cpp` → `main.rs`)
   - CLI argument parsing
   - Experiment selection from config.lua
   - App construction and run

4. **Port profiler main** (`profiler_main.cpp` → `profiler_main.rs`)
   - GPU profiling instrumentation
   - Report generation

### Dependencies

- All previous phases
- cxx bridge to C++ CUDA code

### Validation

- Full simulation run, compare:
  - `stats.csv` identical
  - `genomes.json` identical (within epsilon)
  - `species.csv` identical

---

## Phase 7: Visualization

**Goal**: Port or wrap SFML rendering.

### Option A: Port to Rust SFML (Recommended for long-term)

**Tasks**:
- Use `sfml` crate bindings
- Port `renderer.cpp/hpp` → `renderer.rs`
- Port `overlay.cpp/hpp` → `overlay.rs`
- Port `visualization_manager.cpp/hpp` → `viz_manager.rs`
- Port `frame_snapshot.cpp/hpp` → `frame_snapshot.rs`

### Option B: Keep C++ via cxx (Faster for this phase)

**Tasks**:
- Wrap C++ visualization module in cxx bridge
- Call from Rust application orchestration

### Dependencies

- Phase 6 (app orchestration)

### Validation

- Visual comparison of rendered frames
- Verify agent rendering, selection, stats panel all work

---

## Phase 8: Testing and Validation

**Goal**: Ensure bit-for-bit reproducibility, clean up C++ dependencies.

### Tasks

1. **Create comparison test suite**
   - Run C++ and Rust with identical seeds
   - Compare all output files programmatically
   - Report any divergences

2. **Validate GPU-CPU boundary**
   - Ensure sensor data passed to C++ CUDA matches exactly
   - Ensure results downloaded from CUDA applied correctly

3. **Remove C++ dead code**
   - After each phase validates, remove corresponding C++ files
   - Clean up CMakeLists.txt

4. **Final integration test**
   - Run full experiment matrix (all conditions × seeds)
   - Compare analysis reports

5. **Documentation**
   - FFI boundary contract
   - Build instructions (cargo build, not cmake)

### Validation

- All `stats.csv`, `genomes.json`, `species.csv` files identical across C++ and Rust runs
- Full experiment pipeline completes without errors

---

## Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | Week 1-2 | Infrastructure and cxx setup |
| Phase 2 | Week 3-4 | Core primitives (types, RNG, config, Lua) |
| Phase 3 | Week 5-8 | NEAT evolution |
| Phase 4 | Week 6-7 | Metrics and logging (parallel with Phase 3) |
| Phase 5 | Week 8-11 | Agent registry and simulation state |
| Phase 6 | Week 11-13 | FFI integration and app orchestration |
| Phase 7 | Week 13-15 | Visualization |
| Phase 8 | Week 16-17 | Testing and validation |

---

## Module Mapping

| C++ Module | Rust Crate | Phase |
|------------|------------|-------|
| `src/core/types.hpp` | `moonai-core/types.rs` | 2 |
| `src/core/random.cpp` | `moonai-core/random.rs` | 2 |
| `src/core/config.cpp` | `moonai-core/config.rs` | 2 |
| `src/core/lua_runtime.cpp` | `moonai-core/lua.rs` | 2 |
| `src/core/app_state.cpp` | `moonai-core/state/` | 5 |
| `src/evolution/genome.cpp` | `moonai-evo/genome.rs` | 3 |
| `src/evolution/mutation.cpp` | `moonai-evo/mutation.rs` | 3 |
| `src/evolution/crossover.cpp` | `moonai-evo/crossover.rs` | 3 |
| `src/evolution/species.cpp` | `moonai-evo/species.rs` | 3 |
| `src/evolution/neural_network.cpp` | `moonai-evo/network.rs` | 3 |
| `src/evolution/network_cache.cpp` | `moonai-evo/cache.rs` | 3 |
| `src/evolution/evolution_manager.cpp` | `moonai-evo/evolution.rs` | 3 |
| `src/metrics/metrics.cpp` | `moonai-metrics/metrics.rs` | 4 |
| `src/metrics/logger.cpp` | `moonai-metrics/logger.rs` | 4 |
| `src/app/app.cpp` | `moonai-app/app.rs` | 6 |
| `src/main.cpp` | `moonai-app/main.rs` | 6 |
| `src/visualization/*` | `moonai-viz/` | 7 |
| `src/simulation/*.cu` | C++ (keep via cxx) | 6 |
| `src/evolution/inference_cache.cu` | C++ (keep via cxx) | 6 |

---

## What Stays C++

| Component | File | Reason |
|----------|------|--------|
| Spatial grid + sensors | `batch.cu` | CUDA kernels |
| Simulation step | `batch.cu` | CUDA kernels |
| Neural inference | `inference_cache.cu` | CUDA kernels |
| Buffer management | `buffers.cu` | CUDA memory |
| Visualization | `visualization/` | SFML (if not ported) |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `mlua` parsing differs from `sol2` | High | Build compatibility layer, extensive config.lua testing |
| Float divergence over time | Medium | Strict epsilon comparison (1e-6), deterministic RNG |
| cxx performance overhead | Low | Bridge is thin, no hot-path penalty |
| SFML Rust bindings immature | Medium | Fall back to C++ via cxx |
| Bit-for-bit reproducibility | High | Validate every phase before proceeding |

## Status

All phases complete. See [rust/README.md](rust/README.md) for current project state.

| Phase | Status | Crate | Tests |
|-------|--------|-------|-------|
| Phase 1: Infrastructure | ✅ Complete | `moonai-ffi` | 6 |
| Phase 2: Core Primitives | ✅ Complete | `moonai-core` | 11 |
| Phase 3: NEAT Evolution | ✅ Complete | `moonai-evo` | 31 |
| Phase 4: Metrics & Logging | ✅ Complete | `moonai-metrics` | 9 |
| Phase 5: Agent Registry | ✅ Complete | `moonai-state` | 20 |
| Phase 6: App Orchestration | ✅ Complete | `moonai-app` | 2 |
| Phase 7: Visualization | ⏸️ Deferred | - | - |
| Phase 8: Testing | ✅ Complete | `moonai-app` | 9 integration |

**Total: 88 tests passing**
