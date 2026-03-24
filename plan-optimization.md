# Performance Audit: MoonAI Simulation Engine

## Context

Each generation takes ~3.6s (baseline: 2000 agents, 1500 ticks/gen). The user wants architectural optimization recommendations, not micro-optimizations. This audit is based on profiler data from `output/profiles/` and full source code analysis.

---

## Current Performance Breakdown (per generation avg, 2000 agents)

| Phase | Time (ms) | % of Gen | Location |
|-------|-----------|----------|----------|
| **Sensor Building** | 2,813 | **77.6%** | `physics.cpp:build_sensors` via `write_sensors_flat` |
| ↳ Spatial Grid Queries | 2,221 | 61.3% | `spatial_grid.cpp:query_radius_into` |
| **Simulation Tick** | 572 | **15.8%** | `simulation_manager.cpp:tick` |
| ↳ Process Food | 334 | 9.2% | `simulation_manager.cpp:process_food` (sequential) |
| ↳ Process Attacks | 145 | 4.0% | `physics.cpp:process_attacks` (sequential) |
| ↳ Boundary Apply | 43 | 1.2% | sequential loop |
| **GPU NN Inference** | 164 | **4.5%** | kernel + D2H wait |
| ↳ Kernel Launch | 6 | 0.2% | `neural_inference.cu` |
| **Evolution** | 37 | **1.0%** | speciate + reproduce |
| Other | ~37 | 1.0% | grid rebuild, energy, death, logging |

**Key insight**: GPU accelerates the wrong thing. NN inference is only 6ms (0.2%), while sensor building (2,813ms / 77.6%) and simulation tick (572ms / 15.8%) dominate. The GPU is idle 99.8% of the time.

---

## Bottleneck Analysis

### Bottleneck #1: Sensor Building (77.6% — 2,813 ms/gen)

**Root cause**: Every tick, every alive agent queries the spatial grid twice (once for agents, once for food). That's ~2000 agents × 1500 ticks × 2 queries = **~6M spatial grid queries per generation**. Profiler confirms: 7.9M grid_query_calls scanning 544M candidates.

**Why it's slow**:
- `query_radius_into` checks `ceil(vision_range/cell_size)²` cells — with default `vision_range=150` and `cell_size=150`, that's 9 cells (3×3). But the profiler shows 544M candidates scanned for 7.9M queries = ~69 candidates per query, meaning cells are moderately populated.
- Each query does distance² comparison for every candidate in range.
- `write_sensors_flat` is OpenMP parallelized, but the spatial grid structure itself isn't thread-friendly (all threads read the same grid, causing cache contention).

**Where the time goes** (nested): `gpu_sensor_flatten` (2813ms) ⊃ `physics_build_sensors_accumulated` (2751ms) ⊃ `spatial_query_radius_accumulated` (2221ms). So 79% of sensor time is in grid queries, and 21% is the distance/angle math.

### Bottleneck #2: Simulation Tick (15.8% — 572 ms/gen)

**Root cause**: `process_food()` and `process_attacks()` are sequential (no OpenMP) and each does their own spatial grid queries.

- `process_food`: 1.88M food_eat_attempts per gen, each queries food grid
- `process_attacks`: 796K attack_checks per gen, each queries agent grid
- These run after the spatial grid is rebuilt fresh each tick

**The grid is rebuilt every tick**: `rebuild_spatial_grid()` + `rebuild_food_grid()` = 25ms/gen (cheap), but it means the grid is rebuilt 1500 times per generation.

### Bottleneck #3: Redundant Spatial Queries

The spatial grid is queried in **three separate phases** per tick:
1. Sensor building: queries agent grid + food grid for every agent (OpenMP parallel)
2. `process_food()`: queries food grid again for every prey (sequential)
3. `process_attacks()`: queries agent grid again for every predator (sequential)

Phases 2 and 3 repeat spatial queries that phase 1 already performed (for food and nearby agents). The data from sensor building could be cached and reused.

---

## Optimization Recommendations

### Recommendation 1: GPU-Accelerated Sensor Building (HIGH IMPACT — estimated 3-5x speedup)

**What**: Move the entire sensor computation to the GPU. Currently only NN inference runs on GPU (6ms), while sensor building (2,813ms) runs on CPU.

**How**:
- Upload agent positions, velocities, energies, food positions to GPU once per tick
- CUDA kernel: each thread handles one agent, reads positions from global memory, computes nearest-neighbor distances using a GPU-side spatial grid (or brute-force for small populations, which is fine up to ~5K on modern GPUs)
- Output: flat sensor buffer already on GPU → feed directly into NN inference kernel (eliminates CPU→GPU sensor transfer entirely)

**Pros**:
- Eliminates the 77.6% bottleneck
- Eliminates CPU↔GPU data transfer for sensors (currently sensors are computed on CPU, copied to pinned memory, then DMA'd to GPU)
- GPU spatial queries scale better with population (2048+ CUDA cores vs 8-16 CPU cores)
- The entire sense→think pipeline becomes GPU-resident

**Cons**:
- GPU spatial grid is more complex to implement (hash grid or sort-based approach)
- For small populations (<1000), GPU overhead may not be worth it
- Increases GPU memory usage (need positions + food on GPU)
- Debugging GPU spatial queries is harder than CPU

**Implementation approach**:
- **Simple first version**: brute-force O(N²) distance check per agent on GPU. For 2000 agents this is 4M distance comparisons — trivial for a GPU. Each thread scans all other agents. This avoids implementing a GPU spatial grid entirely.
- **Scaled version** (for 5K+ agents): GPU hash grid — upload positions, build cell lists on GPU, query in parallel. Libraries like CUB can help with the sort/prefix-sum needed for grid construction.

**Files to modify**:
- `src/gpu/neural_inference.cu` — add sensor kernel or new file `sensor_kernel.cu`
- `src/gpu/gpu_batch.hpp/cu` — add position upload, sensor output buffer
- `src/evolution/evolution_manager.cpp` — restructure GPU path to: upload positions → sensor kernel → NN kernel → download actions
- `src/simulation/simulation_manager.cpp` — expose position/food data in GPU-friendly flat arrays

### Recommendation 2: Fused Tick Processing (MEDIUM IMPACT — estimated 20-30% of remaining time)

**What**: Merge `process_food()`, `process_attacks()`, and sensor building into a single spatial query pass per agent per tick, and parallelize the simulation tick.

**How**:
- During sensor building, we already query the spatial grid for nearby agents and food. Cache these neighbor lists.
- Use the cached neighbors for food pickup and attack resolution instead of querying the grid again.
- Parallelize the independent parts of `tick()`: energy drain, boundary apply, death check are all per-agent with no cross-agent dependencies → trivially parallelizable with OpenMP.

**Current waste**:
- Sensor build queries agent grid (parallel) → discards neighbor list
- `process_attacks` queries agent grid again (sequential) for predators
- Sensor build queries food grid (parallel) → discards neighbor list
- `process_food` queries food grid again (sequential) for prey

**Estimated savings**: eliminates ~50% of grid queries (the ones in process_food and process_attacks), plus parallelizes the remaining sequential loops. The simulation tick could drop from 572ms to ~200ms.

**Files to modify**:
- `src/simulation/simulation_manager.cpp` — refactor tick to accept cached neighbor data
- `src/simulation/physics.cpp` — return neighbor lists alongside sensor data
- `src/evolution/evolution_manager.cpp` — pass cached data to tick

### Recommendation 3: Incremental Spatial Grid Updates (MEDIUM IMPACT)

**What**: Instead of clearing and rebuilding the spatial grid from scratch every tick (2 grids × 1500 ticks = 3000 rebuilds), update it incrementally.

**How**:
- Track agent movement deltas. Most agents move a small fraction of a cell per tick.
- Only re-insert agents that crossed cell boundaries.
- For food: only update when food is eaten or respawned (rare events: ~1000/gen out of 3M ticks×food).

**Current cost**: 25ms/gen (grid rebuild) — not a bottleneck itself, but incremental updates also improve cache locality since the grid structure stays warm across ticks.

**Risk**: More complex code, potential for stale data bugs. The current approach is clean and correct.

### Recommendation 4: GPU-Resident Simulation Tick (HIGH IMPACT for large populations, COMPLEX)

**What**: Move the entire tick (movement, energy, food, attacks, death) to GPU, making the simulation fully GPU-resident between generations.

**How**:
- Agent state lives on GPU: positions, velocities, energies, alive flags
- Per-tick: sensor kernel → NN kernel → action kernel → physics kernel (food/attack/energy/death)
- Only download statistics at end of generation (fitness inputs: age, kills, food_eaten, energy, distance)

**Pros**:
- Eliminates ALL CPU↔GPU transfers during a generation (only transfer once at start, once at end)
- All per-tick work is parallel
- Scales linearly with GPU cores for large populations

**Cons**:
- Major architectural change — nearly full rewrite of simulation tick
- Food/attack resolution has sequential dependencies (one kill per predator per tick, food consumption is a race condition)
- Need atomic operations for food eating and kill tracking
- Harder to debug, profile, and maintain
- CPU fallback becomes a separate code path that must stay in sync

**Verdict**: This is the "endgame" optimization for 10K+ populations but is a large engineering effort. Recommendation 1 (GPU sensors) gets 80% of the benefit at 20% of the effort.

### Recommendation 5: Skip Dead Agents Early (LOW IMPACT but trivial)

**What**: The profiler shows 500 predators + 1500 prey = 2000 agents, but by mid-generation many are dead. Currently, dead agents are still iterated in several loops (with `alive()` checks that short-circuit).

**How**:
- Maintain a compact alive-agent index list, updated when agents die
- Use this for all per-agent loops (sensor build, tick processing)
- Avoids iterating over dead agents entirely

**Estimated savings**: Depends on death rate. If 30% die by mid-generation, this saves ~15% of per-agent iteration overhead.

### Recommendation 6: Batch Spatial Queries (MEDIUM IMPACT, complements Rec 1)

**What**: Instead of calling `query_radius_into` once per agent per tick, batch all queries together.

**How**:
- Collect all query positions + radii into arrays
- Single-pass grid traversal that bins results by querier
- Better cache utilization: grid cells are read once for all nearby queries instead of once per query

**This naturally emerges from the GPU approach** (Rec 1), where the kernel inherently batches all queries.

---

## Recommended Implementation Order

1. **GPU Sensor Building** (Rec 1) — highest ROI, attacks the 77.6% bottleneck
2. **Fused Tick Processing** (Rec 2) — moderate effort, removes redundant queries
3. **Skip Dead Agents** (Rec 5) — trivial, incremental improvement
4. **GPU-Resident Tick** (Rec 4) — only if targeting 10K+ populations

**Expected combined result** (Recs 1+2+5): Generation time drops from ~3.6s to ~0.5-0.8s for 2000 agents.

---

## What's Already Well-Optimized

- **Spatial grid design**: Cell size = vision range is correct (minimizes cells checked)
- **NN inference on GPU**: Async pipeline with pinned memory is textbook
- **OpenMP parallelization**: Sensor build uses `schedule(dynamic)` which is correct for varying agent aliveness
- **Thread-local buffers**: `thread_local std::vector<AgentId> nearby_ids` avoids allocation contention
- **Precomputed evaluation order**: NN topological sort is done once per generation, not per tick
- **Network data packing**: CSR format with capacity tracking avoids GPU reallocation

---

## Verification Plan

After implementing optimizations:
1. Run `just profile` and compare generation times against this baseline (3.6s/gen avg)
2. Verify simulation determinism: run with same seed before/after, compare `stats.csv` output
3. Test at multiple scales: 2K (baseline), 5K, 10K, 20K agents
4. Run `just test` to ensure no regressions
5. Compare GPU vs CPU paths to ensure identical behavior
