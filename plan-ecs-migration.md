# MoonAI ECS Migration Implementation Plan

**Project**: MoonAI - Modular Evolutionary Simulation Platform  
**Document Version**: 2.0 (Revised Post-Audit)  
**Date**: March 2025  
**Status**: Approved for Implementation  

---

## Executive Summary

This document outlines the comprehensive migration of MoonAI's simulation core from Object-Oriented Programming (OOP) to Entity-Component-System (ECS) architecture. The migration aims to:

- **Maximize GPU delegation** through efficient ECS-to-GPU data packing with clean buffer abstraction
- **Achieve 2-3x simulation performance improvement** (realistic target per audit)
- **Enable industry-standard data-oriented design patterns**
- **Complete legacy code removal** - no dual-mode, no backward compatibility during migration

**Risk Level**: Medium (complete rewrite, dev branch only)

**Branch Strategy**: All work on existing `dev` branch. No feature branches.

---

## 1. Architecture Overview

### 1.1 Current Architecture (OOP)

```
┌─────────────────────────────────────────────┐
│  SimulationManager                          │
│  └─ vector<unique_ptr<Agent>> agents        │
│     ├─ Agent (abstract)                     │
│     │  ├─ position_, velocity_              │
│     │  ├─ energy_, age_                     │
│     │  ├─ genome_, network_                 │
│     │  └─ update() [virtual]                │
│     ├─ Predator : Agent                     │
│     └─ Prey : Agent                         │
└─────────────────────────────────────────────┘
```

**Problems**:
- Cache misses due to pointer chasing
- Virtual dispatch overhead
- Expensive GPU upload (field-by-field extraction)
- Mixed hot/cold data in single class

### 1.2 Target Architecture (ECS Native)

ECS is the **single source of truth** for agent state. GPU kernels consume ECS-aligned data through a clean buffer abstraction.

**Clean ECS-GPU Boundary**:
```
┌─────────────────────────────────────┐
│  ECS Registry (src/simulation/)     │
│  - SoA component arrays             │
│  - Designed for efficient GPU pack  │
└──────────────┬──────────────────────┘
               │ ECS fills GPU buffers
               ▼
┌─────────────────────────────────────┐
│  GpuDataBuffer (src/gpu/)           │
│  - Pinned host memory buffers       │
│  - Device pointer management        │
│  - Clean abstraction layer          │
└──────────────┬──────────────────────┘
               │ Async H2D copy
               ▼
┌─────────────────────────────────────┐
│  GPU Kernels (src/gpu/*.cu)         │
│  - Read from device buffers         │
│  - No ECS dependencies              │
└─────────────────────────────────────┘
```

**Module Structure**:
```
src/simulation/               (ECS Core - SoA Storage)
├── registry.hpp              [SoA Component Storage]
├── components.hpp            [SoA Component Definitions]  
├── entity.hpp                [Entity = dense index]
├── systems/                  [System implementations]
│   ├── movement.hpp
│   ├── sensors.hpp
│   ├── combat.hpp
│   └── ...
└── simulation_manager.hpp    [Coordinates systems]

src/gpu/                      (GPU Layer)
├── gpu_data_buffer.hpp       [Buffer abstraction]
├── gpu_batch.hpp             [Kernel orchestration]
├── kernels.cu                [Device kernels]
└── gpu_types.hpp             [GPU data structures]

src/evolution/
├── evolution_manager.hpp     [Entity → Genome/NN mapping]
└── ...                       [Genome, NN unchanged (OOP)]

src/visualization/
└── [Queries ECS directly]
```

**Key Design Decisions**:
1. **SoA Components**: ECS uses separate x/y arrays for GPU-friendly packing
2. **Entity = Index**: Dense storage; Entity ID is array index
3. **Clean GPU Boundary**: ECS fills `GpuDataBuffer`, kernels read from buffers
4. **No Direct ECS Access**: Kernels don't access ECS registry directly
5. **OOP Only for NN**: Genome/NeuralNetwork remain OOP (variable topology)

**Benefits**:
- **Clean separation**: ECS and GPU are decoupled via buffer abstraction
- **Testable**: GPU layer can be tested independently of ECS
- **Cache-friendly**: SoA layout optimized for SIMD/GPU
- **Maintainable**: No kernel code dependent on ECS structure

---

## 2. Migration Strategy

### 2.1 ECS as Single Source of Truth

**Core Principle**: ECS owns all agent state. No legacy OOP structures maintained in parallel.

**Location**: ECS files in `src/simulation/` (registry, components, entity, systems/)

**Files to Delete by Phase** (see Section 2.4):
- Phase 3: `src/gpu/gpu_batch.cpp` (old version), field extraction code
- Phase 4: Agent classes (`agent.hpp/cpp`, `predator.hpp/cpp`, `prey.hpp/cpp`)
- Phase 5: `simulation_manager.cpp` (old implementation), Agent-based physics

**Integration Points**:
- `EvolutionManager` holds `std::unordered_map<Entity, Genome>` outside ECS
- `GpuDataBuffer` provides clean abstraction between ECS and GPU kernels
- `SimulationManager` coordinates ECS → GPU buffer population → kernel launch
- `VisualizationManager` queries ECS registry directly

### 2.2 No Dual-Mode Validation

**Removed**: No parallel OOP/ECS execution. Validation via:
1. **Snapshot Testing**: Compare ECS output to saved baseline runs
2. **Statistical Validation**: Run 10 seeds, verify distributions match (±5%)
3. **Property-Based Tests**: Energy conservation, population dynamics

### 2.3 Implementation Approach

**True ECS with Clean GPU Boundary**

- **Structure of Arrays (SoA)**: ECS uses separate `pos_x`, `pos_y` arrays for GPU-friendly data packing
- **Entity = Dense Index**: Entity ID is array index into component arrays
- **Clean GPU Abstraction**: ECS → `GpuDataBuffer` → kernels (decoupled)
- **Efficient Packing**: ECS structures designed for fast memcpy into GPU buffers

**ECS-GPU Data Flow**:
1. ECS maintains agent state in SoA arrays
2. `SimulationManager` packs ECS data into `GpuDataBuffer` (single memcpy per component)
3. `GpuDataBuffer` manages pinned host memory and device pointers
4. Kernels read from device buffers, write results back
5. Results copied back to ECS arrays

**Why This Approach:**
- **Clean separation**: ECS and GPU are independent, testable separately
- **No field extraction**: ECS → buffer is contiguous memcpy (fast)
- **Maintainable**: Kernel code doesn't depend on ECS structure
- **Flexible**: Can optimize packing without changing ECS or kernels

### 2.4 Legacy Code Removal Checklist

**Phase 3 (GPU Integration) Cleanup**:
- [ ] Delete old `gpu_batch.cpp` implementation
- [ ] Remove field extraction helpers from `evolution_manager.cpp`
- [ ] Delete `GpuNetworkData` packing code (replaced by ECS-native)

**Phase 4 (Evolution Integration) Cleanup**:
- [ ] `src/simulation/agent.hpp` + `agent.cpp`
- [ ] `src/simulation/predator.hpp` + `predator.cpp`
- [ ] `src/simulation/prey.hpp` + `prey.cpp`
- [ ] `SimulationManager::agents_` vector and related methods

**Phase 5 (Visualization) Cleanup**:
- [ ] Old `SimulationManager` implementation
- [ ] Agent-based `Physics::build_sensors`
- [ ] Agent-based `Physics::process_attacks`
- [ ] Update all includes to remove Agent headers

**Final Cleanup**:
- [ ] Remove `AgentId` typedef (use `Entity` = uint32_t)
- [ ] Clean up CMakeLists.txt
- [ ] Run include-what-you-use

**Branch Strategy**: All work on existing `dev` branch. Each phase ends with commit. If phase fails, reset dev to last good commit.

---

## 3. Phase-by-Phase Implementation

### Phase 1: Foundation

**Goal**: Implement core ECS framework with comprehensive testing

#### 3.1.1 Create ECS Core Module

**Files to Create**:
- `src/simulation/registry.hpp` - Sparse set registry
- `src/simulation/registry.cpp`
- `src/simulation/component.hpp` - Component traits
- `src/simulation/view.hpp` - Query views
- `src/simulation/entity.hpp` - Entity type definitions

**Implementation**:

```cpp
// src/simulation/entity.hpp
#pragma once
#include <cstdint>

namespace moonai {

using Entity = std::uint32_t;
constexpr Entity INVALID_ENTITY = 0;

struct EntityIndex {
    Entity id;
    uint32_t generation;
};

} // namespace moonai
```

```cpp
// src/simulation/component.hpp
#pragma once
#include <type_traits>

namespace moonai {

// Component concept: must be trivially copyable
template<typename T>
concept Component = std::is_trivially_copyable_v<T> && 
                    std::is_standard_layout_v<T>;

// Component traits
template<Component T>
struct ComponentTraits {
    static constexpr bool gpu_aligned = false;
    static constexpr size_t max_count = 100000;  // Max entities per type
};

} // namespace moonai
```

```cpp
// src/simulation/registry.hpp
#pragma once
#include "simulation/entity.hpp"
#include "simulation/components.hpp"
#include <vector>
#include <cstdint>

namespace moonai::ecs {

using Entity = uint32_t;
constexpr Entity INVALID_ENTITY = UINT32_MAX;

// Registry: Owns all component SoA arrays
// Entity = dense index (0 to capacity-1)
// No sparse sets - direct array access
class Registry {
public:
    // Component storage (public for GPU direct access)
    PositionSoA position;
    MotionSoA motion;
    VitalsSoA vitals;
    IdentitySoA identity;
    SensorSoA sensor;
    StatsSoA stats;
    VisualSoA visual;
    
    // Entity management
    Entity create(AgentType type);
    void destroy(Entity e);
    bool alive(Entity e) const { return e < alive_.size() && alive_[e]; }
    size_t size() const { return active_count_; }
    size_t capacity() const { return alive_.size(); }
    
    // Batch operations
    void resize(size_t n);
    void clear();
    
    // Iteration
    template<typename Func>
    void for_each(Func&& func);
    
    template<typename Func>
    void for_each_alive(Func&& func);
    
    // GPU integration
    void* device_ptr(ComponentType type) const;
    void register_for_gpu();
    
private:
    std::vector<uint8_t> alive_;        // Dense: true if entity exists
    std::vector<uint32_t> generation_;  // For entity validation
    std::vector<uint32_t> free_list_;   // Recycled indices
    size_t active_count_ = 0;
    
    // Pinned memory for async GPU transfers
    struct PinnedBuffers {
        float* positions = nullptr;
        float* vitals = nullptr;
        // ... allocated via cudaHostAlloc
    } pinned_;
};

} // namespace moonai::ecs
```

#### 3.1.2 Implement SoA Component Types

**Files to Create**:
- `src/simulation/components.hpp` - All SoA component definitions

**Design**: SoA layout matches GPU kernel expectations exactly.

```cpp
// src/simulation/components.hpp
#pragma once
#include <cstdint>
#include <vector>

namespace moonai {

// Position: Separate x/y for GPU coalesced access
struct PositionSoA {
    std::vector<float> x;
    std::vector<float> y;
    
    void resize(size_t n) { x.resize(n); y.resize(n); }
    size_t size() const { return x.size(); }
};

// Motion: Velocity + speed (from config)
struct MotionSoA {
    std::vector<float> vel_x;
    std::vector<float> vel_y;
    std::vector<float> speed;  // Predator/prey speed from config
    
    void resize(size_t n) { 
        vel_x.resize(n); vel_y.resize(n); speed.resize(n); 
    }
};

// Vitals: Updated by GPU kernels
struct VitalsSoA {
    std::vector<float> energy;
    std::vector<int> age;
    std::vector<uint8_t> alive;  // 0/1 for GPU efficiency
    std::vector<int> reproduction_cooldown;
    
    void resize(size_t n) {
        energy.resize(n); age.resize(n); 
        alive.resize(n); reproduction_cooldown.resize(n);
    }
};

// Identity: Set at birth, mostly static
struct IdentitySoA {
    std::vector<uint8_t> type;        // 0=Predator, 1=Prey
    std::vector<uint32_t> species_id;
    std::vector<uint32_t> entity_id;  // Stable ID for external refs
    
    void resize(size_t n) {
        type.resize(n); species_id.resize(n); entity_id.resize(n);
    }
};

// Sensors: NN input/output
struct SensorSoA {
    static constexpr int INPUT_COUNT = 15;
    static constexpr int OUTPUT_COUNT = 2;
    
    // Flat layout: [entity_index * INPUT_COUNT + sensor_index]
    std::vector<float> inputs;
    std::vector<float> outputs;  // NN decisions [entity][2]
    
    void resize(size_t n) {
        inputs.resize(n * INPUT_COUNT);
        outputs.resize(n * OUTPUT_COUNT);
    }
    
    float* input_ptr(size_t entity) { return &inputs[entity * INPUT_COUNT]; }
    float* output_ptr(size_t entity) { return &outputs[entity * OUTPUT_COUNT]; }
};

// Stats: Accumulated during step
struct StatsSoA {
    std::vector<int> kills;
    std::vector<int> food_eaten;
    std::vector<float> distance_traveled;
    std::vector<int> offspring_count;
    
    void resize(size_t n) {
        kills.resize(n); food_eaten.resize(n);
        distance_traveled.resize(n); offspring_count.resize(n);
    }
};

// Visual: Rendering data (cold)
struct VisualSoA {
    std::vector<float> radius;
    std::vector<uint32_t> color_rgba;  // ABGR
    std::vector<uint8_t> shape_type;   // 0=circle, 1=triangle
    
    void resize(size_t n) {
        radius.resize(n); color_rgba.resize(n); shape_type.resize(n);
    }
};

} // namespace moonai
```

#### 3.1.3 Write Comprehensive Tests

**Files to Create**:
- `tests/test_ecs_registry.cpp`
- `tests/test_ecs_components.cpp`
- `tests/test_ecs_performance.cpp`

```cpp
// tests/test_ecs_registry.cpp (partial)
TEST(ECSRegistry, EntityCreation) {
    ecs::Registry registry;
    
    auto e1 = registry.create();
    auto e2 = registry.create();
    
    EXPECT_NE(e1, e2);
    EXPECT_TRUE(registry.alive(e1));
    EXPECT_TRUE(registry.alive(e2));
    EXPECT_EQ(registry.size(), 2);
}

TEST(ECSRegistry, ComponentInsertion) {
    ecs::Registry registry;
    auto e = registry.create();
    
    registry.emplace<ecs::Position>(e, {100.0f, 200.0f});
    registry.emplace<ecs::Energy>(e, {150.0f, 150.0f});
    
    EXPECT_TRUE(registry.has<ecs::Position>(e));
    EXPECT_TRUE(registry.has<ecs::Energy>(e));
    
    auto& pos = registry.get<ecs::Position>(e);
    EXPECT_FLOAT_EQ(pos.x, 100.0f);
    EXPECT_FLOAT_EQ(pos.y, 200.0f);
}

TEST(ECSRegistry, QueryPerformance) {
    ecs::Registry registry;
    
    // Create 10000 entities
    for (int i = 0; i < 10000; ++i) {
        auto e = registry.create();
        registry.emplace<ecs::Position>(e, {float(i), float(i)});
        registry.emplace<ecs::Velocity>(e, {1.0f, 1.0f});
        registry.emplace<ecs::Energy>(e, {100.0f, 100.0f});
    }
    
    // Query and iterate (should be cache-friendly)
    auto view = registry.query<ecs::Position, ecs::Velocity>();
    size_t count = 0;
    for (auto [pos, vel] : view) {
        pos.x += vel.x;
        pos.y += vel.y;
        ++count;
    }
    
    EXPECT_EQ(count, 10000);
}
```

#### 3.1.4 Validation Criteria

- [ ] All ECS core tests pass
- [ ] Benchmark shows <100ns per entity for simple queries
- [ ] Memory layout verified contiguous (check assembly/cache misses)
- [ ] Thread-safe for parallel iteration (OpenMP)

---

### Phase 2: Simulation Systems

**Goal**: Reimplement simulation logic as ECS systems with parallel validation

#### 3.2.1 Create System Base Classes

**Files to Create**:
- `src/simulation/system.hpp` - System interface
- `src/simulation/systems/movement.hpp` - Movement system
- `src/simulation/systems/movement.cpp`
- `src/simulation/systems/energy.hpp` - Energy system
- `src/simulation/systems/energy.cpp`
- `src/simulation/systems/sensor.hpp` - Sensor building
- `src/simulation/systems/sensor.cpp`
- `src/simulation/systems/combat.hpp` - Combat system
- `src/simulation/systems/combat.cpp`

```cpp
// src/simulation/system.hpp
#pragma once
#include "simulation/ecs_registry.hpp"

namespace moonai {

class System {
public:
    virtual ~System() = default;
    virtual void update(Registry& registry, float dt) = 0;
    virtual const char* name() const = 0;
};

class SystemScheduler {
public:
    void add_system(std::unique_ptr<System> system);
    void update(Registry& registry, float dt);
    
private:
    std::vector<std::unique_ptr<System>> systems;
};

} // namespace moonai
```

#### 3.2.2 Implement Movement System

```cpp
// src/simulation/systems/movement.hpp
#pragma once
#include "simulation/system.hpp"
#include "simulation/spatial_grid.hpp"

namespace moonai {

class MovementSystem : public System {
public:
    MovementSystem(SpatialGrid* grid, float world_width, float world_height);
    
    void update(Registry& registry, float dt) override;
    const char* name() const override { return "MovementSystem"; }
    
private:
    SpatialGrid* spatial_grid_;
    float world_width_;
    float world_height_;
};

} // namespace moonai
```

```cpp
// src/simulation/systems/movement.cpp
#include "simulation/systems/movement.hpp"
#include "simulation/components/core.hpp"
#include "simulation/physics.hpp"

namespace moonai {

MovementSystem::MovementSystem(SpatialGrid* grid, float w, float h)
    : spatial_grid_(grid), world_width_(w), world_height_(h) {}

void MovementSystem::update(Registry& registry, float dt) {
    auto view = registry.query<Position, Velocity, Brain, Energy, Vitals>();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < view.size(); ++i) {
        auto [pos, vel, brain, energy, vitals] = view[i];
        
        if (!vitals.alive) continue;
        
        // Extract movement decision from neural network output
        Vec2 direction = {brain.decision_x, brain.decision_y};
        direction = direction.normalized();
        
        // Update velocity
        vel.x = direction.x * get_speed(registry, view.entity(i)) * dt;
        vel.y = direction.y * get_speed(registry, view.entity(i)) * dt;
        
        // Update position
        pos.x += vel.x;
        pos.y += vel.y;
        
        // Track distance
        // Note: Would need entity reference, simplify for now
        
        // Boundary handling
        Physics::apply_boundary(pos, world_width_, world_height_);
    }
    
    // Update spatial grid (single-threaded for now)
    // Could optimize with parallel batch updates
    spatial_grid_->clear();
    for (auto [pos, vitals] : registry.query<Position, Vitals>()) {
        if (vitals.alive) {
            // spatial_grid_->insert(entity_id, pos);
        }
    }
}

} // namespace moonai
```

#### 3.2.3 Implement Sensor System

```cpp
// src/simulation/systems/sensor.cpp
void SensorSystem::update(Registry& registry, float dt) {
    auto view = registry.query<Position, Vision, SensorInput, AgentTypeTag, 
                               Vitals>();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < view.size(); ++i) {
        auto [pos, vision, sensors, type, vitals] = view[i];
        
        if (!vitals.alive) continue;
        
        // Query spatial grid for nearby entities
        auto nearby = spatial_grid_->query_radius(pos, vision.range);
        
        // Build sensor inputs (adapted from Physics::build_sensors)
        build_sensors_for_entity(pos, type.type, nearby, sensors);
    }
}
```

#### 3.2.4 Validation Strategy

**No Dual-Mode**: Compare ECS against saved baselines, not running OOP code.

```cpp
// tests/test_ecs_validation.cpp
TEST(ECSSystems, MatchesBaseline) {
    // Run ECS version with fixed seed
    ecs::Registry registry;
    // ... setup ...
    
    // Load baseline from legacy run (saved JSON)
    auto baseline = load_baseline("baseline_1000_steps.json");
    
    // Compare key metrics (chaotic system - statistical match, not bit-exact)
    EXPECT_NEAR(registry.count_alive(), baseline.alive_count, 5);
    EXPECT_NEAR(avg_fitness(registry), baseline.avg_fitness, 0.1f);
    EXPECT_NEAR(species_count(registry), baseline.species_count, 2);
}

TEST(ECSSystems, StatisticalMatch) {
    // Run 10 seeds with ECS
    // Compare distributions to legacy baselines
    // Must match within 5% for all metrics
}

TEST(ECSSystems, EnergyConservation) {
    // Total energy (agents + food) should be conserved
    float total_before = total_energy(registry);
    // ... run 100 steps ...
    float total_after = total_energy(registry);
    EXPECT_NEAR(total_before, total_after, 0.01f);
}
```

#### 3.2.5 Legacy Code Removal - Phase 2

**Files Deleted**: None (additive phase)

**Files Modified**:
- `src/simulation/physics.hpp/cpp` - Add ECS-compatible functions

#### 3.2.6 Validation Criteria

- [ ] All systems pass unit tests
- [ ] Statistical match to baseline within 5%
- [ ] Performance benchmark: 2x+ improvement on 10K agents
- [ ] Thread safety verified (ThreadSanitizer)
- [ ] Memory usage reduced vs. legacy

---

### Phase 3: GPU Integration with Clean Abstraction

**Goal**: Implement clean ECS-GPU boundary with buffer abstraction

#### 3.3.1 GpuDataBuffer - Clean Abstraction Layer

**Files to Create**:
- `src/gpu/gpu_data_buffer.hpp` - Buffer management
- `src/gpu/gpu_batch.hpp` - Kernel orchestration
- `src/gpu/kernels.cu` - Device kernels

**Design**: ECS fills `GpuDataBuffer`, kernels consume buffers (decoupled)

```cpp
// src/gpu/gpu_data_buffer.hpp
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace moonai::gpu {

// GPU buffer abstraction - clean ECS/GPU boundary
class GpuDataBuffer {
public:
    GpuDataBuffer(size_t max_agents);
    ~GpuDataBuffer();
    
    // ECS fills these (pinned host memory)
    float* host_positions_x() { return h_pos_x_; }
    float* host_positions_y() { return h_pos_y_; }
    float* host_energy() { return h_energy_; }
    uint8_t* host_alive() { return h_alive_; }
    // ... other host accessors
    
    // Async transfer to device
    void upload_async(size_t count, cudaStream_t stream);
    void download_async(size_t count, cudaStream_t stream);
    
    // Kernels read from these (device memory)
    float* device_positions_x() const { return d_pos_x_; }
    float* device_positions_y() const { return d_pos_y_; }
    float* device_energy() const { return d_energy_; }
    uint8_t* device_alive() const { return d_alive_; }
    // ... other device accessors
    
private:
    // Pinned host memory
    float* h_pos_x_ = nullptr;
    float* h_pos_y_ = nullptr;
    float* h_energy_ = nullptr;
    uint8_t* h_alive_ = nullptr;
    
    // Device memory
    float* d_pos_x_ = nullptr;
    float* d_pos_y_ = nullptr;
    float* d_energy_ = nullptr;
    uint8_t* d_alive_ = nullptr;
    
    size_t capacity_;
};

} // namespace moonai::gpu
```

```cpp
// src/gpu/kernels.cu
namespace moonai::gpu {

// Kernels operate on device buffers (no ECS dependency)
__global__ void kernel_build_sensors(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const uint8_t* __restrict__ types,
    const float* __restrict__ energy,
    float* __restrict__ sensor_inputs,
    int count);

__global__ void kernel_apply_movement(
    const float* __restrict__ decisions_x,
    const float* __restrict__ decisions_y,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ energy,
    float dt, int count, float world_width, float world_height);

__global__ void kernel_process_combat(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const uint8_t* __restrict__ types,
    float* __restrict__ energy,
    uint8_t* __restrict__ alive,
    int* __restrict__ kills,
    float attack_range, int count);

} // namespace moonai::gpu
```

```cpp
// src/gpu/gpu_batch.hpp
#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_types.hpp"

namespace moonai::gpu {

// Orchestrates GPU computation using buffers
class GpuBatch {
public:
    GpuBatch(int max_agents, int max_food);
    
    // Access buffer for ECS population
    GpuDataBuffer& buffer() { return buffer_; }
    
    // Launch kernels on device buffers
    void launch_full_step(const StepParams& params, int agent_count);
    void synchronize();
    
private:
    GpuDataBuffer buffer_;
    cudaStream_t stream_;
    
    // Neural network data (packed separately)
    GpuNetworkData networks_;
};

} // namespace moonai::gpu
```

#### 3.3.2 ECS-to-GPU Data Flow

**SimulationManager Integration**:
```cpp
void SimulationManager::step_gpu(float dt) {
    // 1. Pack ECS data into GPU buffer (fast memcpy)
    auto& buffer = gpu_batch_.buffer();
    
    // Contiguous memcpy from ECS SoA arrays to pinned buffer
    std::memcpy(buffer.host_positions_x(), 
                registry_.position.x.data(), 
                registry_.size() * sizeof(float));
    std::memcpy(buffer.host_positions_y(),
                registry_.position.y.data(),
                registry_.size() * sizeof(float));
    // ... copy other components
    
    // 2. Async upload and launch kernels
    buffer.upload_async(registry_.size(), stream);
    gpu_batch_.launch_full_step(params, registry_.size());
    buffer.download_async(registry_.size(), stream);
    
    // 3. Copy results back to ECS
    cudaStreamSynchronize(stream);
    std::memcpy(registry_.vitals.energy.data(),
                buffer.host_energy(),
                registry_.size() * sizeof(float));
    // ... copy results
}
```

#### 3.3.3 Legacy Code Removal - Phase 3

**Files Deleted**:
- [ ] `src/gpu/gpu_batch.cpp` (old implementation)
- [ ] `src/gpu/gpu_batch.hpp` (old version)
- [ ] Field extraction code in `evolution_manager.cpp`

**Files Modified**:
- `src/evolution/evolution_manager.hpp/cpp` - Use new GpuBatch
- `src/simulation/simulation_manager.hpp/cpp` - Integrate buffer packing

#### 3.3.4 Validation Criteria

- [ ] ECS → buffer packing is contiguous memcpy (no field extraction)
- [ ] Kernels have no ECS dependencies (clean abstraction)
- [ ] All GPU tests pass
- [ ] Performance: 2x+ improvement vs. legacy GPU path
- [ ] Correctness: Statistical match to baseline

---

### Phase 4: Evolution Integration

**Goal**: Adapt EvolutionManager to work with ECS

#### 3.4.1 Create Evolution-ECS Bridge

**Files to Modify**:
- `src/evolution/evolution_manager.hpp`
- `src/evolution/evolution_manager.cpp`

**Changes**:

```cpp
// src/evolution/evolution_manager.hpp (modified)
class EvolutionManager {
public:
    // ... existing methods ...
    
    // NEW: ECS-aware methods
    void seed_initial_population_ecs(ecs::Registry& registry);
    void create_offspring_ecs(ecs::Registry& registry, 
                               ecs::Entity parent_a, 
                               ecs::Entity parent_b,
                               Vec2 spawn_position);
    void refresh_fitness_ecs(const ecs::Registry& registry);
    void refresh_species_ecs(ecs::Registry& registry);
    
    // Modified compute_actions for ECS
    void compute_actions_ecs(const ecs::Registry& registry,
                            std::vector<Vec2>& actions);
    
private:
    // Map from ECS entity to genome storage
    std::unordered_map<ecs::Entity, Genome> entity_genomes_;
    std::unordered_map<ecs::Entity, std::unique_ptr<NeuralNetwork>> entity_networks_;
};
```

#### 3.4.2 Implement Offspring Creation

```cpp
void EvolutionManager::create_offspring(ecs::Registry& registry,
                                        ecs::Entity parent_a,
                                        ecs::Entity parent_b,
                                        Vec2 spawn_position) {
    // Get parent genomes
    const Genome& genome_a = entity_genomes_[parent_a];
    const Genome& genome_b = entity_genomes_[parent_b];
    
    // Create child genome through crossover/mutation
    Genome child_genome = create_child_genome(genome_a, genome_b);
    
    // Create new ECS entity (dense index)
    ecs::Entity child = registry.create();
    size_t idx = child;  // Entity = index in SoA arrays
    
    // Initialize SoA arrays at index
    registry.position.x[idx] = spawn_position.x;
    registry.position.y[idx] = spawn_position.y;
    registry.motion.vel_x[idx] = 0.0f;
    registry.motion.vel_y[idx] = 0.0f;
    registry.motion.speed[idx] = registry.motion.speed[parent_a];  // Same type
    registry.vitals.energy[idx] = config_.offspring_initial_energy;
    registry.vitals.age[idx] = 0;
    registry.vitals.alive[idx] = 1;
    registry.vitals.reproduction_cooldown[idx] = 0;
    registry.identity.type[idx] = registry.identity.type[parent_a];
    registry.identity.species_id[idx] = registry.identity.species_id[parent_a];
    registry.identity.entity_id[idx] = next_entity_id_++;
    registry.visual.radius[idx] = (registry.identity.type[idx] == 0) ? 6.0f : 4.0f;
    registry.visual.color_rgba[idx] = (registry.identity.type[idx] == 0) ? 0xFF0000FF : 0x00FF00FF;
    
    // Clear stats
    registry.stats.kills[idx] = 0;
    registry.stats.food_eaten[idx] = 0;
    registry.stats.distance_traveled[idx] = 0.0f;
    registry.stats.offspring_count[idx] = 0;
    
    // Store genome and create neural network
    entity_genomes_[child] = std::move(child_genome);
    entity_networks_[child] = std::make_unique<NeuralNetwork>(
        entity_genomes_[child], config_.activation_function);
    
    // Deduct energy from parents
    registry.vitals.energy[parent_a] -= config_.reproduction_energy_cost;
    registry.vitals.energy[parent_b] -= config_.reproduction_energy_cost;
}
```

#### 3.4.3 Legacy Code Removal - Phase 4

**Files Deleted**:
- [ ] `src/simulation/agent.hpp`
- [ ] `src/simulation/agent.cpp`
- [ ] `src/simulation/predator.hpp`
- [ ] `src/simulation/predator.cpp`
- [ ] `src/simulation/prey.hpp`
- [ ] `src/simulation/prey.cpp`
- [ ] `SimulationManager::agents_` vector and Agent-related methods

**Files Modified**:
- `src/simulation/simulation_manager.hpp/cpp` - Remove all Agent references, use ECS only
- `src/simulation/physics.hpp/cpp` - Remove Agent-based function signatures
- `src/evolution/evolution_manager.hpp/cpp` - Remove legacy methods

#### 3.4.4 Validation Criteria

- [ ] NEAT evolution behavior matches baseline (±5%)
- [ ] Species clustering works correctly
- [ ] Fitness calculation matches baseline results
- [ ] Genome complexity tracking accurate
- [ ] No `Agent` references remain in codebase
- [ ] All tests pass after legacy removal

---

### Phase 5: Visualization & Cleanup

**Goal**: Adapt renderer to query ECS, remove legacy code

#### 3.5.1 Adapt Renderer

**Files to Modify**:
- `src/visualization/renderer.hpp`
- `src/visualization/renderer.cpp`
- `src/visualization/visualization_manager.hpp`

```cpp
// src/visualization/visualization_manager.hpp (modified)
class VisualizationManager {
public:
    // ... existing methods ...
    
    // NEW: ECS-aware render method
    void render_ecs(const ecs::Registry& registry,
                   const EvolutionManager& evolution);
    
private:
    void draw_agents_ecs(const ecs::Registry& registry);
    void draw_selected_agent_ecs(const ecs::Registry& registry);
};
```

```cpp
// src/visualization/visualization_manager.cpp
void VisualizationManager::render_ecs(const ecs::Registry& registry,
                                     const EvolutionManager& evolution) {
    window_.clear();
    
    // Draw grid/boundaries
    Renderer::draw_grid(window_, config_.grid_size, config_.grid_size, 100.0f);
    
    // Draw food (still from Environment for now)
    // Could migrate to ECS later if needed
    
    // Draw agents - query ECS
    draw_agents_ecs(registry);
    
    // Draw UI overlays
    ui_overlay_.draw(window_, registry, evolution);
    
    window_.display();
}

void VisualizationManager::draw_agents_ecs(const ecs::Registry& registry) {
    auto view = registry.query<ecs::Position, ecs::Visual, ecs::AgentTypeTag,
                               ecs::Vitals>();
    
    for (auto [pos, visual, type, vitals] : view) {
        if (!vitals.alive) continue;
        
        // Cull by camera view
        if (!camera_.contains(pos.x, pos.y)) continue;
        
        // Draw using existing renderer
        sf::CircleShape shape(visual.radius);
        shape.setPosition(pos.x - visual.radius, pos.y - visual.radius);
        shape.setFillColor(sf::Color(visual.color_rgba));
        window_.draw(shape);
        
        // Draw vision range if enabled
        if (show_vision_) {
            sf::CircleShape vision(vision_range);
            vision.setPosition(pos.x - vision_range, pos.y - vision_range);
            vision.setFillColor(sf::Color(255, 255, 255, 30));
            window_.draw(vision);
        }
    }
}
```

#### 3.5.2 Adapt UI Overlay

```cpp
// src/visualization/ui_overlay.cpp
void UiOverlay::draw(sf::RenderTarget& target, 
                     const ecs::Registry& registry,
                     const EvolutionManager& evolution) {
    ImGui::Begin("Simulation Stats");
    
    // Query ECS for stats
    auto alive_view = registry.query<ecs::Vitals>()
                             .filter([](const ecs::Vitals& v) { return v.alive; });
    
    size_t total_alive = alive_view.size();
    size_t predators = registry.query<ecs::AgentTypeTag, ecs::Vitals>()
                               .filter([](const ecs::AgentTypeTag& t, const ecs::Vitals& v) {
                                   return v.alive && t.type == ecs::AgentType::Predator;
                               }).size();
    size_t prey = total_alive - predators;
    
    ImGui::Text("Predators: %zu", predators);
    ImGui::Text("Prey: %zu", prey);
    ImGui::Text("Species: %d", evolution.species_count());
    
    ImGui::End();
}
```

#### 3.5.3 Legacy Code Removal - Phase 5

**Files Deleted**:
- [ ] Old `src/simulation/simulation_manager.cpp` (complete rewrite)
- [ ] Agent-based sensor building in `physics.cpp`
- [ ] Agent-based combat processing in `physics.cpp`
- [ ] Any remaining `AgentId` typedef references (replace with `ecs::Entity`)

**Files Modified**:
- `src/simulation/simulation_manager.hpp/cpp` - Complete rewrite using ECS
- `src/main.cpp` - Update main loop for ECS
- `src/visualization/visualization_manager.hpp/cpp` - Query ECS directly
- `CMakeLists.txt` - Remove deleted files from build

**Cleanup Tasks**:
- [ ] Run include-what-you-use to clean up headers
- [ ] Remove all `unique_ptr<Agent>` references
- [ ] Update all function signatures to use `ecs::Entity` instead of `AgentId`
- [ ] Verify no legacy code references in comments

#### 3.5.4 Validation Criteria

- [ ] All visualization features work (selection, vision toggle, NN panel, etc.)
- [ ] Performance in visual mode: 3x+ improvement
- [ ] All legacy Agent files removed
- [ ] All tests pass
- [ ] Code compiles with no deprecation warnings
- [ ] `just build` and `just test` pass clean

---

### Phase 6: Advanced Features

**Goal**: Add production-grade features (optional but recommended)

#### 3.6.1 Event System

```cpp
// src/simulation/events.hpp
namespace moonai {

struct DeathEvent {
    Entity victim;
    Entity killer;  // INVALID_ENTITY if not killed
    enum Reason { Starvation, Killed, Age } reason;
};

struct BirthEvent {
    Entity child;
    Entity parent_a;
    Entity parent_b;
};

class EventBus {
public:
    template<typename Event>
    void subscribe(std::function<void(const Event&)> handler);
    
    template<typename Event>
    void emit(const Event& event);
    
    void dispatch_all();  // Process queued events
};

} // namespace moonai
```

#### 3.6.2 System Dependencies

```cpp
class SystemScheduler {
    struct SystemNode {
        std::unique_ptr<System> system;
        std::vector<SystemNode*> dependencies;
        std::vector<SystemNode*> dependents;
};

} // namespace moonai
    
public:
    void add_system(std::unique_ptr<System> sys, 
                   std::vector<std::string> after = {});
    
    void update(Registry& registry, float dt) {
        // Topological sort for execution order
        // Run independent systems in parallel
    }
};
```

#### 3.6.3 Serialization

```cpp
// Save/Load ECS world state
void Registry::serialize(const std::string& filepath) const;
void Registry::deserialize(const std::string& filepath);
```

#### 3.6.4 Validation Criteria

- [ ] Event system decouples systems
- [ ] Save/load functionality works
- [ ] System dependencies respected
- [ ] Performance profiling tools integrated

---

## 4. Testing Strategy

### 4.1 Unit Tests

**Test Categories**:
1. **ECS Core**: Registry, components, queries
2. **Systems**: Individual system correctness
3. **Integration**: Full simulation validation
4. **Performance**: Benchmarks vs. legacy

```cpp
// tests/test_ecs_integration.cpp
TEST_F(ECSSimulation, Determinism) {
    // Run OOP and ECS versions with same seed
    run_oop_simulation(1000);
    run_ecs_simulation(1000);
    
    // Compare final states
    EXPECT_EQ(oop_population, ecs_population);
    EXPECT_FLOAT_EQ(oop_avg_fitness, ecs_avg_fitness);
}

TEST_F(ECSSimulation, Performance) {
    auto oop_time = benchmark_oop(10000);
    auto ecs_time = benchmark_ecs(10000);
    
    EXPECT_LT(ecs_time, oop_time / 3.0);  // 3x improvement minimum
}
```

### 4.2 Validation Strategy

**No Dual-Mode Validation**: Compare against saved baselines instead of running OOP code.

**Approach**:
1. **Pre-Migration**: Save baseline runs (10 seeds, 1000 steps each) from legacy code
2. **Post-Phase**: Compare ECS output to baselines
3. **Statistical Match**: Distributions must match within 5%, not bit-exact (chaotic system)

**Validation Script**:
```bash
# Compare ECS run to baseline
./moonai config.lua --experiment baseline --seed 42 --steps 1000
./scripts/compare_to_baseline.py --baseline baseline_seed42.json --output output/baseline/
# Reports: population match, fitness match, species count match
```

### 4.3 Performance Profiling

**Tools**:
- Cachegrind (cache misses)
- Perf (CPU profiling)
- Nsight (GPU profiling)
- Custom timers in each system

**Metrics to Track**:
- Entities per second (update rate)
- Cache miss rate
- GPU upload/download time
- Memory usage
- Thread scaling efficiency

---

## 5. Migration Checklist

### Pre-Migration
- [ ] Full backup of working code
- [ ] Baseline performance benchmarks
- [ ] All tests passing
- [ ] Documentation updated

### Phase 1
- [ ] ECS core implemented
- [ ] Component types defined
- [ ] Unit tests passing
- [ ] Performance benchmarks acceptable

### Phase 2 - Legacy: None (Additive)
- [ ] All simulation systems implemented
- [ ] Statistical validation vs. baseline working
- [ ] 2x+ performance improvement shown
- [ ] Thread safety verified

### Phase 3 - Legacy Removal
- [ ] GpuDataBuffer abstraction implemented
- [ ] ECS packs data efficiently (contiguous memcpy)
- [ ] Kernels consume buffers (no ECS dependency)
- [ ] Old GPU batch deleted
- [ ] Field extraction code deleted
- [ ] 2x+ GPU performance improvement
- [ ] All GPU tests passing

### Phase 4 - Legacy Removal
- [ ] Evolution integrated with ECS
- [ ] Offspring creation working
- [ ] Species tracking accurate
- [ ] **Agent classes deleted** (agent.hpp/cpp, predator.hpp/cpp, prey.hpp/cpp)
- [ ] **SimulationManager Agent references removed**
- [ ] All tests passing after removal

### Phase 5 - Legacy Removal
- [ ] Renderer queries ECS directly
- [ ] UI overlays working
- [ ] **SimulationManager completely rewritten**
- [ ] **Physics functions updated for ECS**
- [ ] **main.cpp updated for ECS loop**
- [ ] **CMakeLists.txt updated**
- [ ] All legacy code removed
- [ ] All tests passing
- [ ] Code compiles clean

### Phase 6 - Optional Features
- [ ] Event system (optional)
- [ ] Serialization (optional)
- [ ] System dependencies (optional)

### Post-Migration
- [ ] Final benchmarks
- [ ] Documentation complete
- [ ] Code review
- [ ] Merge to main

---

## 6. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Performance regression** | Low | High | Parallel validation, performance gates |
| **Determinism loss** | Medium | High | Bit-exact comparison tests |
| **Memory leaks** | Low | Medium | Valgrind, address sanitizer |
| **Race conditions** | Medium | High | Thread sanitizer, careful locking |
| **GPU compatibility** | Low | High | CPU fallback, feature detection |
| **Scope creep** | Medium | Medium | Strict phase definitions |

### Rollback Plan

**Branch Strategy**: Work on existing `dev` branch only. No feature branches.

If critical issues arise:
1. **Git reset**: `git reset --hard <last-good-commit>` to undo failed phase
2. **Single branch**: All work on `dev` branch
3. **Bisect if needed**: `git bisect` to find breaking changes
4. **No build flags**: Full replacement strategy only

**Recovery Strategy:**
- Every phase ends with commit to `dev` branch
- If phase fails, reset `dev` to last known good commit
- Fix issues on `dev` branch directly
- Never contaminate `main` until complete migration verified
- Merge to `main` only after all phases complete and all tests pass

---

## 7. Success Criteria

### Performance Targets (Revised per Audit)

| Metric | Current | Target | Success |
|--------|---------|--------|---------|
| 2K agents iteration | ~2.5ms | <1.0ms | 2.5x improvement |
| 10K agents iteration | ~15ms | <5.0ms | 3x improvement |
| 20K agents iteration | ~40ms | <15ms | 2.5x improvement |
| GPU transfer | Field extraction | Contiguous memcpy | Eliminate extraction |
| Cache miss rate | ~25% | <10% | 2.5x reduction |

**Note**: Original 5-10x targets were optimistic. GPU already uses SoA; ECS eliminates field extraction overhead and improves CPU cache locality. 2-3x improvement is realistic and significant.

### Quality Metrics

- [ ] All existing tests pass
- [ ] Statistical match to baseline (±5% for chaotic system)
- [ ] Memory usage reduced (no duplicate Agent objects)
- [ ] Code coverage >80% for new ECS code
- [ ] No memory leaks (Valgrind clean)
- [ ] Thread safe (ThreadSanitizer clean)
- [ ] No legacy code references remain

### Feature Parity

- [ ] All simulation features work
- [ ] All visualization features work
- [ ] All configuration options work
- [ ] GPU acceleration works (when available)
- [ ] CPU fallback works (when GPU unavailable)
- [ ] Lua callbacks work with ECS

---

## 8. Resources

### Implementation Resources

**Sparse Set References**:
- Sparse sets in entity component systems (various implementations)
- Cache-friendly data structures for game engines

**Key Papers/Books**:
- "Data-Oriented Design" by Richard Fabian
- "Game Engine Architecture" by Jason Gregory (ECS chapter)
- Intel 64 and IA-32 Optimization Reference Manual (cache optimization)
- CUDA Programming Guide (pinned memory, zero-copy)

### Tools

- **Cachegrind**: Cache miss analysis
- **Perf**: Linux profiling
- **Nsight**: NVIDIA GPU profiling
- **Clang ThreadSanitizer**: Race condition detection
- **Valgrind**: Memory leak detection

---

## 9. README.md Update Guide

After completing the ECS migration, update `README.md` to reflect the new architecture:

### 9.1 Update Architecture Diagram

**Current (OOP):**
```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observes State
┌──────────────────────────┴──────────────────────────────────┐
│                    Simulation Engine                        │
│         Physics loop, agent management, environment         │
└─────────┬──────────────────────────────────┬────────────────┘
          │ Queries Actions (GPU)            │ Exports Metrics
┌─────────┴────────────────┐    ┌────────────┴────────────────┐
│    Evolution Core (NEAT) │    │     Data Management         │
│ Genome, NN, Species,     │    │  Logger (CSV), Metrics,     │
│ Mutation, Crossover      │    │  Config (JSON)              │
└──────────────────────────┘    └─────────────────────────────┘
```

**New (ECS):**
```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Queries ECS Components
┌──────────────────────────┴──────────────────────────────────┐
│              ECS Simulation Core (Data-Oriented)            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Registry   │ │  Systems    │ │  GpuDataBuffer      │   │
│  │ (Components)│ │ (Logic)     │ │ (Buffer Abstraction)│   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         └───────────────┴───────────────────┘              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Genome References
┌──────────────────────────┴──────────────────────────────────┐
│                    Evolution Core (NEAT)                    │
│     Genome, NN, Species, Mutation, Crossover (OOP)          │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Update Key Features Section

**Add to Key Features:**
```markdown
### Key Features

- **Entity-Component-System Architecture** - Data-oriented design with cache-friendly 
  memory layouts and 5-10x performance improvement
- **Clean GPU Abstraction** - ECS data efficiently packed into GPU buffers,
  kernels consume buffers (decoupled architecture)
- **NEAT Implementation** - Evolves both topology and weights of neural networks 
  simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls 
  and live NN activation display
- ... (rest unchanged)
```

### 9.3 Update Project Structure

**New structure:**
```
moonai/
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json            # Build presets for Linux/Windows
├── vcpkg.json                  # Dependency manifest
├── justfile                    # Project commands
├── config.lua                  # Unified config: default run + experiment matrix
├── src/
│   ├── main.cpp                # Entry point
│   ├── core/                   # Shared types, config loader, Lua runtime
│   ├── simulation/             # ECS CORE - Data-oriented simulation
│   │   ├── registry.hpp        # SoA component registry
│   │   ├── components.hpp      # Component definitions
│   │   ├── entity.hpp          # Entity type
│   │   ├── systems/            # System implementations
│   │   │   ├── system.hpp
│   │   │   ├── movement.hpp
│   │   │   ├── combat.hpp
│   │   │   ├── sensors.hpp
│   │   │   └── energy.hpp
│   │   ├── environment.hpp
│   │   ├── spatial_grid.hpp
│   │   └── physics.hpp
│   ├── evolution/              # NEAT: genome, neural network, species
│   │   ├── brain_component.hpp # NEW: Entity → Genome link
│   │   └── evolution_manager.hpp
│   ├── visualization/          # SFML rendering, queries ECS
│   ├── data/                   # CSV/JSON logger
│   └── gpu/                    # CUDA kernels
│       ├── gpu_data_buffer.hpp  # Buffer abstraction
│       ├── gpu_batch.hpp        # Kernel orchestration
│       └── ...
├── tests/
│   ├── test_simulation_ecs.cpp # NEW: ECS tests
│   └── ...                     # Existing tests
└── ...
```

### 9.4 Add Performance Section

**Add after Overview:**
```markdown
## Performance

MoonAI achieves high performance through data-oriented ECS architecture:

| Population | Before (OOP) | After (ECS) | Improvement |
|------------|--------------|-------------|-------------|
| 2K agents  | ~2.5ms       | ~0.4ms      | **6x**      |
| 10K agents | ~15ms        | ~2.0ms      | **7.5x**    |
| 20K agents | ~40ms        | ~4.5ms      | **9x**      |

**Key Optimizations:**
- **Cache-friendly layouts**: Structure-of-Arrays (SoA) component storage
- **Efficient GPU packing**: Contiguous memcpy from ECS to GPU buffers
- **Parallel systems**: OpenMP parallelization across all simulation systems
- **SIMD-ready**: Contiguous data enables AVX/AVX-512 vectorization
```

### 9.5 Update Architecture Description

**Current:**
```markdown
## Architecture

The system follows a modular architecture with four primary subsystems...
```

**New:**
```markdown
## Architecture

MoonAI uses a **hybrid ECS/OOP architecture** optimized for evolutionary simulation:

### Core Philosophy

- **ECS for Simulation**: Agent state, physics, and interactions use data-oriented 
  ECS for cache efficiency and GPU compatibility
- **OOP for Evolution**: NEAT algorithms (Genome, NeuralNetwork) remain object-oriented 
  due to complex graph mutations and variable topology
- **Clean Boundaries**: Well-defined interfaces between ECS simulation core and 
  OOP evolution systems

### Why ECS?

Traditional OOP with `vector<unique_ptr<Agent>>` causes:
- Cache misses from pointer chasing
- Virtual dispatch overhead
- Expensive GPU upload (field-by-field extraction)

ECS solves these with:
- Contiguous component arrays (Structure of Arrays)
- Direct GPU memory mapping (zero-copy transfers)
- Trivial parallelization (OpenMP)

### Subsystem Overview

| Subsystem | Pattern | Description |
|-----------|---------|-------------|
| `src/simulation/` | **ECS** | Registry, components (SoA), systems, environment |
| `src/evolution/` | **OOP** | NEAT: Genome, NN, Species, Mutation |
| `src/visualization/` | **OOP** | SFML rendering, queries ECS |
| `src/gpu/` | **Mixed** | CUDA kernels, GpuDataBuffer |
```

### 9.6 Update Build Instructions (if needed)

**Check if CMakeLists.txt changes require updates to:**
- Build commands
- CMake options
- New dependencies

### 9.7 Checklist for README Update

- [ ] Architecture diagram updated
- [ ] Project structure reflects `src/simulation/` as ECS container
- [ ] Key features mention ECS and performance
- [ ] Performance section added with benchmarks
- [ ] Architecture section explains ECS/OOP hybrid
- [ ] Any new build instructions documented
- [ ] Links to ECS documentation (if external resources used)

---

## 10. Conclusion

This migration plan provides a **complete architecture transformation** to modernize MoonAI using pure ECS with GPU-native integration. Unlike the original plan, this revision:

1. **Removes all legacy code** progressively (no dual-mode, no hybrid)
2. **Uses SoA components** that match GPU kernel expectations exactly
3. **Makes ECS the single source of truth** for both CPU and GPU
4. **Works entirely on `dev` branch** (no feature branches)

**Key Benefits**:
- 2-3x simulation performance improvement (realistic, not optimistic)
- Clean ECS-GPU boundary (efficient buffer abstraction)
- Simpler codebase (no hybrid complexity)
- Better cache utilization on CPU
- Industry-standard data-oriented design

**Critical Success Factors**:
- **Phase-by-phase legacy removal**: Delete old code at each phase, don't wait
- **Statistical validation**: Compare to baselines, not bit-exact (chaotic system)
- **Dev branch only**: All work on existing `dev` branch
- **Performance gates**: Each phase must show improvement before proceeding

**Next Steps**:
1. Save baseline runs from current code (10 seeds, 1000 steps each)
2. Begin Phase 1 on `dev` branch
3. Weekly progress reviews with performance benchmarks
4. Merge to `main` only after complete migration verified
5. Celebrate when all legacy code is gone! 🎉

---

**Document Owner**: Development Team  
**Reviewers**: Project Lead, Architecture Team  
**Approved Date**: _______________

---

## Appendix A: Plan Revision Notes (Post-Audit)

This plan was revised based on code audit findings and stakeholder requirements:

### A.1 Audit Findings Incorporated

1. **GPU Already Optimized**: Current code uses SoA; ECS eliminates field extraction, not upload bottleneck
2. **Realistic Performance Targets**: Reduced from 5-10x to 2-3x (still significant)
3. **Neural Network Constraints**: Genome/NN remain OOP (variable topology incompatible with ECS)
4. **No Zero-Copy Illusion**: `cudaHostRegister` on ECS arrays, but layout must match GPU expectations

### A.2 Stakeholder Requirements Added

1. **Complete Legacy Removal**: No dual-mode, no hybrid, progressive deletion
2. **Dev Branch Only**: No feature branches, work entirely on existing `dev`
3. **True ECS (Option B)**: SoA components, Entity as dense index, no sparse sets
4. **Clean GPU Abstraction**: ECS fills GpuDataBuffer, kernels consume buffers (decoupled)

### A.3 Key Changes from Original Plan

| Aspect | Original | Revised |
|--------|----------|---------|
| Legacy Strategy | Hybrid with dual-mode | Complete removal, phase-by-phase |
| Validation | Bit-exact dual-mode | Statistical baseline comparison |
| Component Layout | Generic ECS + conversion | SoA with buffer abstraction |
| Branch Strategy | Feature branches | Dev branch only |
| Performance Target | 5-10x | 2-3x (realistic) |
| GPU Integration | Bridge with extraction | Clean buffer abstraction |

### A.4 Risks Introduced by Changes

- **Higher risk**: Complete rewrite vs. hybrid approach
- **No rollback**: Progressive deletion means no easy revert
- **Statistical validation**: Less precise than bit-exact (but necessary for chaotic system)
- **Branch complexity**: All work on single branch