---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Rewrite Plan: MoonAI — GPU-First Architecture

## Overview

Complete rewrite in Rust with GPU-first simulation design. The GPU runs all simulation compute (agents, sensors, neural inference, physics, spatial grid, genome crossover and mutation, birth activation) as a **persistent kernel** that executes N steps per tick. The CPU orchestrates: metrics export, UI data delivery, and initial population seeding. No per-step CPU iteration over agents. No CPU-side genome compilation.

## Design Principles

1. **GPU owns all simulation state** — positions, velocities, energy, age, alive flags, genomes, innovation counters, all live in GPU memory.
2. **CPU is orchestrator only** — never iterates the agent population except for initial population seeding and metrics export.
3. **Tick-based cadence** — GPU runs N simulation steps per tick; CPU handles metrics logging between ticks.
4. **GPU-native reproduction** — crossover, mutation, and network compilation happen entirely on GPU every step. No CPU thread pool needed.
5. **Buffer expansion** — buffers grow by 2x when capacity threshold is reached. No artificial ceiling.

## Assumptions

- `config.lua` remains the config format.
- Output schema stays unchanged so Python analysis keeps working.
- Behavioral parity is the goal.
- Predator and prey use separate GPU buffers; no `AgentType` enum needed.
- `config.lua` is loaded via `mlua`. `moonai_defaults` is injected as a global table.
- Reproduction is **sexual** — two parent genomes crossover on GPU, mutation applied on GPU, network compiled on GPU.
- FPS target: 120fps. Speed multiplier: 1x-1024x steps per frame. Every frame renders everything live.
- UI needs fresh data every frame: population counts, positions, velocities, all of it.

## Technology Choices

| Concern | C++ | Rust/GPU-First |
|---------|-----|----------------|
| Logging | spdlog | `tracing` + `tracing-subscriber` |
| JSON | nlohmann/json | `serde` + `serde_json` |
| Lua binding | Lua C API | `mlua` crate |
| GUI framework | SFML | winit + egui + wgpu |
| GPU rendering | SFML shapes | wgpu instanced rendering |
| Atomic counters | — | CUDA atomics for GPU-to-CPU events |
| Genome compilation | CPU (rayon) | GPU (persistent kernel) |

## Architecture

### Tick Execution Model

```
CPU tick loop:
  TICK N
    GPU: simulation_kernel(N_STEPS_PER_TICK)
      for each step 0..N-1:
        grid_build
        sensor_compute
        neural_inference
        update_vitals
        resolve_food
        resolve_combat
        apply_movement
        reproduction:
          evaluate_reproduction_eligibility
          find_mate (DenseReproductionGrid)
          write (slot, parent_a, parent_b) to birth buffer
          gpu_crossover_and_mutate(offspring)
          gpu_compile_network(offspring)
          activate_slot(offspring)
        write_atomics(births, deaths, kills, food_eaten)
        write_ui_stats(alive_pred, alive_prey, ...)

    GPU signals completion

    CPU:
      if report_interval: GPU_reduce_metrics -> log CSV/JSON
      if headless: wgpu_render()
      else: wgpu_render() + egui_overlay()

```

### GPU-Side Innovation Tracking (Atomic Counter, NOT Hash Map)

NEAT innovation tracking requires assigning globally unique innovation IDs to new structural mutations. A GPU hash map (open addressing) suffers from bank conflicts under heavy concurrent insert from thousands of threads. Instead, use **atomic counter + direct assignment**:

```
Global GPU state:
  innovation_counter: atomic<uint32>   // monotonic, starts at (num_inputs + num_outputs + 1)
  next_node_id: atomic<uint32>        // monotonic for hidden nodes

Per-step innovation log (append-only):
  innovation_log[step][innovation_id] = {from_node, to_node, innovation_type}
  // Used for matching homologues during crossover
```

**Mutation -- add_connection**:
1. Pick random `(from_node, to_node)` pair
2. Check if connection exists by scanning this agent's connection array (O(C), typically <500)
3. If not found and `num_connections < max_connections`: atomically increment `innovation_counter` -> new ID -> insert connection

**Mutation -- add_node**:
1. Pick random enabled connection `(a, b)` with innovation `I`
2. Atomically increment `next_node_id` -> hidden node `h`
3. Atomically increment `innovation_counter` twice -> `I1`, `I2`
4. Disable connection `(a, b)`, insert `(a, h): I1`, `(h, b): I2`

**Why this is fast**:
- No hash map contention -- atomics only on counter increments (1-2 ops each)
- Connection existence check is a simple linear scan -- O(C) is fine since most connections do not mutate
- All other mutations (weight perturbation, enable/disable) are data movement, no atomics

### GPU-Side Crossover

Each offspring gets one GPU thread. No warp divergence since each thread handles one offspring independently.

```
gpu_crossover(parent_a, parent_b) -> child:
  1. Read parent A and parent B connection arrays
  2. Sort both by innovation number (warp-level bitonic sort, 32 threads cooperate)
  3. Merge: for each innovation present in either parent:
       - in both: 75% chance inherit from A, 25% from B (matching NEAT behavior)
       - in one only: 50% chance inherit
  4. Disable mismatched connections with 75% probability
  5. Result = child genome (stored in offspring slot's genome arrays)
```

### GPU-Side Network Compilation

After crossover, the child genome's connection arrays are ready. Compile to inference format within the same kernel:

```
gpu_compile_network(offspring_slot):
  1. Topological sort of nodes -> eval_order[]
  2. Build conn_ptr[] -- offset into conn_from[] for each node
  3. Build output_indices[] -- indices of output nodes
  4. Copy weights, enabled flags into inference arrays
```

### GPU-Free Agent Slot Management

GPU maintains a **free list** (ring buffer of dead slot indices).

- When agent dies: push slot index onto free list
- When offspring activates: pop from free list
- If free list empty: **buffer expansion** (see below)

### Buffer Expansion

```
Trigger: when live_count > capacity * 0.9

Expansion:
  new_capacity = capacity * 2
  allocate new buffer (all SoA arrays)
  gpu_copy_all(old_buffer, new_buffer, live_count)
  swap buffer pointers

No artificial ceiling. Buffers grow as needed.
```

### Compaction (GPU-Native, Two-Pass)

Compaction is NOT for ceiling avoidance -- it is for reclaiming dead slots when expansion is undesirable (e.g., nearing max GPU memory). It runs lazily when free list is empty but births are pending.

```
Trigger: free_list empty AND births pending AND we want to reclaim slots

Pass 1 -- Mark:
  for each slot i:
    if alive[i]:
      remap[i] = atomic_counter++

Pass 2 -- Scatter:
  for each slot i:
    if alive[i]:
      new_pos = remap[i]
      copy agent[i] -> buffer[new_pos]

Swap buffer pointers
Reset free_list to dead slots at end of new buffer
```

### UI Data Path (Every Frame)

GPU writes a compact `UiStats` struct to a **pinned host-mapped buffer** every step. CPU reads it with a single `memcpy`. No kernel launch needed.

```
UiStats (pinned, written every step):
  step, predator_count, prey_count
  predator_births, prey_births
  predator_deaths, prey_deaths
  kills, food_eaten
  avg_predator_energy, avg_prey_energy

render pass (wgpu, no CPU readback):
  predator positions -> GPU buffer -> instanced draw
  prey positions -> GPU buffer -> instanced draw
  food positions -> GPU buffer -> instanced draw
  vision circle, sensor lines -> computed on GPU on-demand (click), read via staging buffer
```

### Selected Agent Readback (On Demand)

```
User clicks agent:
  GPU: kernel_compute_selected_agent_features(slot_id, staging_buffer)
    - sensor lines (5 nearest predators, prey, food)
    - vision circle
    - node activations (forward pass)
  CPU: cudaMemcpy async -> read staging buffer -> update NN panel
```

## Crate Architecture

```
Cargo.toml
crates/
  moonai-types/
    src/lib.rs              # Vec2, INVALID_ENTITY, SENSOR_COUNT (35),
                            # OUTPUT_COUNT (2), NodeType, NodeGene,
                            # ConnectionGene, SimulationConfig, ConfigError,
                            # deterministic_respawn, tracing setup
  moonai-evolution/
    src/lib.rs              # Genome, NeuralNetwork, Mutation, Crossover,
                            # Species, InnovationTracker, EvolutionManager,
                            # CompiledNetwork
    src/crossover.cu        # CUDA kernel: genome crossover
    src/mutation.cu         # CUDA kernel: weight mutate, add_connection, add_node
    src/network_compilation.cu  # CUDA kernel: compile genome to inference format
    build.rs                # Compiles .cu files via cxx or raw nvcc
  moonai-simulation/
    src/lib.rs              # SimulationState, GpuHandles, TickResult,
                            # run_tick, read_metrics, read_selected_agent,
                            # init_from_config, init_from_genomes
    src/kernel.cu           # Persistent simulation kernel
                            # NOTE: Calls evolution kernels from moonai-evolution.
                            # Does NOT implement crossover/mutation/reproduction.
    src/buffers.rs          # GPU SoA buffers (agents, food)
    src/checks.rs           # CUDA_CHECK macro
    build.rs                # Compiles kernel.cu
  moonai-metrics/
    src/lib.rs              # Logger, stats.csv, species.csv, genomes.json
  moonai-ui/
    src/lib.rs              # App, winit event loop, egui overlay
    src/render.rs           # wgpu world renderer
    src/types.rs            # OverlayStats, RenderFood, RenderAgent, RenderLine
  moonai/
    src/main.rs             # binary entrypoint
    src/cli.rs              # argument parsing
    src/lua_config.rs        # mlua loading with moonai_defaults injection
    src/signal.rs           # SIGINT/SIGTERM graceful shutdown
```

### Crate Responsibilities

| Crate | Owns | Depends on |
|-------|------|------------|
| `moonai-types` | Shared types, config, deterministic respawn, tracing | -- |
| `moonai-evolution` | ALL NEAT evolution logic: CUDA kernels for crossover, mutation, network compilation, InnovationTracker, Species, EvolutionManager | `moonai-types` |
| `moonai-simulation` | GPU SoA buffers, persistent simulation kernel (calls evolution kernels), spatial grid, inference, metrics reduce, UI stats write | `moonai-types`, `moonai-evolution` |
| `moonai-metrics` | CSV/JSON file logging | `moonai-types` |
| `moonai-ui` | winit, egui panels, wgpu world renderer | `moonai-types`, `moonai-evolution`, `moonai-simulation` |
| `moonai` | CLI, Lua config, signal handling | All above |

## Execution Flow

```
main:
  parse CLI -> load config via mlua -> init tracing -> init CUDA
  init moonai-simulation (allocate buffers)
  init moonai-metrics (open output files)
  init moonai-ui (winit + wgpu) if not headless

  // Seed initial population
  for each initial agent:
    compile_network(genome) -> upload to GPU buffer
    activate_slot(agent)

  // Tick loop
  while running && (max_steps == 0 || step < max_steps):
    tick_result = moonai_simulation.run_tick(steps_per_frame)

    if step % report_interval == 0:
      gpu_reduce_and_log()

    wgpu_render_frame()
    if not headless:
      egui_overlay_draw(ui_stats_from_pinned_buffer)

  flush logs -> output saved
```

## Phases

### Phase 1 -- Workspace Skeleton

1. Create `Cargo.toml` with workspace members.
2. Create `moonai-types/src/lib.rs` with `Vec2`, constants only.
3. Stub all other crates with empty `src/lib.rs` and `build.rs` so `cargo build --workspace` succeeds.
4. Verify build.

### Phase 2 -- Core Types and Infrastructure

Build in dependency order (verify at each step):

1. **`moonai-types`** -- Vec2, INVALID_ENTITY, SENSOR_COUNT (35), OUTPUT_COUNT (2), NodeType, NodeGene, ConnectionGene, SimulationConfig (all fields), ConfigError, validate_config, serde Serialize/Deserialize.
2. **`moonai-types` / deterministic_respawn** -- hash-based food respawn. `#[repr(C)]`, `#[no_mangle]` for CUDA compatibility.
3. **`moonai-types` / tracing** -- subscriber setup (fmt + env filter).
4. **`moonai-evolution`** -- stub.
5. **`moonai-simulation`** -- stub.
6. **`moonai-metrics`** -- stub.
7. **`moonai-ui`** -- stub.
8. **`moonai` / cli** -- all current flags preserved.
9. **`moonai` / signal** -- graceful shutdown (`ctrlc` crate).
10. **`moonai` / lua_config** -- mlua loading, `moonai_defaults` injection, `table_to_config`.

### Phase 3 -- Evolution (GPU CUDA Kernels)

Implement evolution CUDA kernels in `moonai-evolution`. These are the production implementations — no CPU fallback, no duplication in `moonai-simulation`.

1. **`moonai-evolution` / Genome** -- nodes, connections, add_node, add_connection, has_connection, has_node, max_node_id, complexity, compatibility_distance, serde JSON.
2. **`moonai-evolution` / InnovationTracker** -- get_innovation, get_split_node_id, init_from_population.
3. **`moonai-evolution` / NeuralNetwork** -- activate (CPU reference forward pass), `activate_into`.
4. **`moonai-evolution` / Mutation** -- mutate_weights, add_connection, add_node, delete_connection.
5. **`moonai-evolution` / Crossover** -- crossover (sexual, two parents, matching by innovation number).
6. **`moonai-evolution` / Species** -- compatibility check, add_member, refresh_summary, average_complexity. Static species ID via `AtomicI32`.
7. **`moonai-evolution` / EvolutionManager** -- initialize, seed_initial_population, create_initial_genome, create_child_genome, reproduce_population, refresh_species.
8. **`moonai-evolution` / CompiledNetwork** -- struct that maps Genome -> GPU-uploadable arrays.
9. **`moonai-evolution` / .cu files** -- `crossover.cu`, `mutation.cu`, `network_compilation.cu` in `src/`. Compiled via `build.rs` at crate root.
10. GPU kernel tests deferred (per user request).

### Phase 4 -- GPU Simulation Kernel

1. **`moonai-simulation` / buffers** -- GPU SoA buffers for predator, prey, food. Genome arrays in-place (connection_from, connection_to, connection_weight, connection_innovation, connection_enabled, node_types, num_connections, num_nodes). Pinned host-mapped `UiStats` buffer. Free list ring buffer.
2. **`moonai-simulation` / kernel** -- persistent simulation kernel. Each tick runs N steps with all phases fused:
   - grid_build, sensor_compute, neural_inference, update_vitals, resolve_food, resolve_combat, apply_movement
   - reproduction (evaluate -> find mate -> **calls moonai-evolution kernels** for crossover, mutate, compile -> activate)
   - write_atomics, write_ui_stats
3. **`moonai-simulation` / inference** -- neural inference kernel (forward pass from in-place genome arrays).
4. **`moonai-simulation` / reproduction** -- evaluation, mate finding, birth buffer management. Calls `moonai-evolution` CUDA kernels for actual crossover/mutation/compilation.
5. **`moonai-simulation` / metrics_reduce** -- launched at report_interval. One-pass reduction over agent arrays. Atomic counters + sums -> averages. Output compact struct.
6. **`moonai-simulation` / lib** -- `SimulationState`, `run_tick(n_steps) -> TickResult`, `read_metrics() -> MetricsSnapshot`, `read_selected_agent(id) -> AgentState`, `init_from_config`, `init_from_genomes(Vec<Genome>)`.
7. **`moonai-simulation` / compaction** -- two-pass GPU defragment (mark, scatter, swap). Buffer expansion (2x doubling).

### Phase 5 -- Metrics

1. **`moonai-metrics` / Logger** -- stats.csv, species.csv, genomes.json.
   - Run directory: `YYYYMMDD_HHMMSS_seedN`, explicit name override, conflict suffix.
   - stats.csv header: `step,predator_count,prey_count,predator_births,prey_births,predator_deaths,prey_deaths,predator_species,prey_species,avg_predator_complexity,avg_prey_complexity,avg_predator_energy,avg_prey_energy,max_predator_generation,avg_predator_generation,max_prey_generation,avg_prey_generation`.
2. Wire `moonai-simulation` -> `moonai-metrics` at report intervals.

### Phase 6 -- First Milestone: Headless Runtime

Before touching GUI:

1. `cargo test` -- evolution and config tests pass.
2. `cargo build --workspace` -- all crates compile cleanly.
3. `cargo run -- --validate config.lua` -- matches current behavior.
4. Run small seeded headless experiment -> produce:
   - `config.json`
   - `stats.csv`
   - `species.csv`
   - `genomes.json`

### Phase 7 -- UI

1. **`moonai-ui` / types** -- OverlayStats (fps, step, alive counts, avg complexity/energy), RenderFood, RenderAgent, RenderLine.
2. **`moonai-ui`** -- winit window, event loop, egui context.
3. **`moonai-ui` / render** -- wgpu instanced rendering from GPU simulation buffers. Reads predator/prey/food positions directly from GPU -- no CPU readback. Vision circle and sensor lines computed in GPU kernel on demand (click).
4. Input semantics:

   | Key/Action | Behavior |
   |------------|----------|
   | Space | Pause / resume (stops GPU tick loop) |
   | Up / Down or + / - | Increase / decrease simulation speed (steps per tick) |
   | . | Step one tick (while paused) |
   | S | Save screenshot |
   | Esc | Quit |
   | Left-click | GPU computes agent features -> read staging buffer -> show stats + NN panel |
   | Right-click drag | Pan camera |
   | Scroll wheel | Zoom |

5. Rebuild overlay in egui. Preserve:
   - Left/right panel layout
   - Stats panel (step, population, births/deaths, species counts)
   - Population / complexity / energy charts
   - NN topology panel for selected agent (node/edge graph, blue-gray-orange edge coloring)
   - Energy distribution bars

### Phase 8 -- Cleanup

Once headless and GUI both work:

1. Delete `src/` directory (all C++ code).
2. Delete `CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json`, `src/*/CMakeLists.txt`.
3. Delete `justfile` CMake-related recipes.
4. Update `justfile` to pure Cargo commands.
5. Remove `.clang-format`, `.clang-tidy`.

## GPU Kernel Design

### Sensor Layout (35 inputs, unchanged)

Same as current C++:
- 5 nearest predators x 2 values (dx, dy)
- 5 nearest prey x 2 values
- 5 nearest food x 2 values
- Self energy, vel x, vel y (3 values)
- Wall proximity x, y (2 values)

### Simulation Kernel Phases (per step, inside persistent loop)

1. **Grid build** -- count-scan-scatter or single-pass hash into spatial cells
2. **Sensor build** -- for each agent, search neighboring cells, find 5 nearest of each type, encode
3. **Inference** -- forward pass through NEAT network (tanh activation)
4. **Vitals** -- energy drain, age increment, death check
5. **Food resolution** -- prey within interaction range claim food, energy transfer
6. **Combat resolution** -- predator within range claim prey kill, energy transfer
7. **Movement** -- NN output (dx, dy) scaled by speed -> new position (with wall wrap/bounce)
8. **Reproduction** -- evaluate eligibility -> find mate -> crossover -> mutate -> compile -> activate
9. **UI stats write** -- write UiStats to pinned memory (no sync needed)

### Atomic Counters

Written every step to pinned UiStats buffer (no kernel launch, just global memory write):
- `predator_count`, `prey_count`
- `predator_births`, `prey_births` (cumulative)
- `predator_deaths`, `prey_deaths` (cumulative)
- `kills`, `food_eaten`
- `avg_predator_energy`, `avg_prey_energy`

### Metrics Reduce Kernel

Launched once per `report_interval` steps. One-pass warp reduction -> compact struct -> CPU reads once.

## GPU Memory Layout

All agent data in SoA layout with genome data stored in-place:

```
PredatorBuffer:
  pos_x[N], pos_y[N], vel_x[N], vel_y[N]
  energy[N], age[N], alive[N]
  species_id[N], entity_id[N], generation[N]

  // Genome data (variable-length, in-place)
  connection_from[N * max_connections]
  connection_to[N * max_connections]
  connection_weight[N * max_connections]
  connection_innovation[N * max_connections]
  connection_enabled[N * max_connections]
  node_types[N * (max_hidden_nodes + FIXED_NODES)]
  num_connections[N], num_nodes[N]
  birth_state[N]  // DEAD=0, ACTIVE=1
  rng_state[N]     // per-agent RNG seed

PreyBuffer: (same layout)

FoodBuffer:
  pos_x[N], pos_y[N], active[N]

UiStats (pinned host-mapped):
  step, predator_count, prey_count, births, deaths, kills, food_eaten,
  avg_predator_energy, avg_prey_energy
```

## Verification Gates

| Gate | Command | Success Criteria |
|------|---------|------------------|
| Core parity | `cargo test` | All evolution/config tests pass |
| Build parity | `cargo build --workspace` | All crates compile, no CMake |
| Config parity | `cargo run -- --validate config.lua` | Output matches current behavior |
| Headless runtime parity | Run small seeded experiment | `stats.csv`/`species.csv` shape and trends match |
| GUI smoke test | Open window, interact | Pause, step, speed, zoom, pan, select, charts all work |
| Cleanup | `rg "C++\|#include" src/` | No matches |
| Cleanup | `ls src/` | Directory does not exist |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Persistent kernel register pressure | Template on N_STEPS; tune block size per GPU |
| Innovation counter atomics contention | Only 1-2 atomics per offspring mutation; not per-agent-per-step |
| Buffer expansion copying large arrays | Doubling is infrequent; add 20% headroom to delay next expansion |
| Genome crossover memory access | Use shared memory for parent genomes; warp-level sort |
| Large population defaults (24k+96k+240k) | wgpu instancing; no per-frame CPU iteration |

## Implementation Order for Next Prompt

1. Phase 1 -- Create workspace skeleton with all crate stubs.
2. Phase 2 -- Build `moonai-types` (types, config, deterministic_respawn, tracing) and `moonai` (cli, signal, lua_config).
3. Phase 3 -- Implement `moonai-evolution` with tests (CPU reference).

This establishes the complete crate structure and CPU-side NEAT reference logic. Phase 4 implements the GPU kernels that mirror the CPU reference.
