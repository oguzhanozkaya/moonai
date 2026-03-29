#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "simulation/entity.hpp"
#include "simulation/food_store.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace moonai {

class Registry;
class EvolutionManager;
namespace gpu {
class GpuBatch;
}

struct SimEvent {
  enum Type : uint8_t { Kill, Food, Birth, Death };
  Type type;
  Entity agent_id;  // predator (kill) or prey (food)
  Entity target_id; // prey (kill), food (food), or self (death)
  Vec2 position;    // where the event occurred
};

class SimulationManager {
public:
  struct ReproductionPair {
    Entity parent_a = INVALID_ENTITY;
    Entity parent_b = INVALID_ENTITY;
    Vec2 spawn_position;
  };

  struct SimulationStepResult {
    std::vector<SimEvent> events;
    std::vector<ReproductionPair> reproduction_pairs;
  };

  explicit SimulationManager(const SimulationConfig &config);
  ~SimulationManager();

  void initialize();
  SimulationStepResult step(Registry &registry, EvolutionManager &evolution);
  SimulationStepResult step_gpu(Registry &registry,
                                EvolutionManager &evolution);
  void reset();

  void enable_gpu(bool enable);
  bool gpu_enabled() const {
    return gpu_enabled_;
  }

  const FoodStore &food_store() const {
    return food_store_;
  }

private:
  void initialize(bool log_initialization);
  void ensure_gpu_capacity(std::size_t agent_count, std::size_t food_count);

  void compact_registry(Registry &registry, EvolutionManager &evolution);
  void collect_gpu_step_events(Registry &registry,
                               const std::vector<uint8_t> &was_alive,
                               const std::vector<uint8_t> &was_food_active,
                               std::vector<SimEvent> &events);
  std::vector<ReproductionPair>
  find_reproduction_pairs(const Registry &registry) const;
  void refresh_world_state_after_step();

  SimulationConfig config_;
  Random rng_;
  FoodStore food_store_;
  int current_step_ = 0;

  // GPU support
  bool gpu_enabled_ = false;
  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
};

} // namespace moonai
