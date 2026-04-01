#pragma once

#include "core/config.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace moonai {

struct AppState;
struct AgentRegistry;
struct PendingOffspring;
class EvolutionManager;

namespace gpu {

class GpuBatch;

}

class SimulationManager {
public:
  explicit SimulationManager(const SimulationConfig &config);
  ~SimulationManager();

  void initialize(AppState &state);
  void step(AppState &state, EvolutionManager &evolution);
  void reset(AppState &state);

  void enable_gpu(AppState &state, bool enable);
  void disable_gpu(AppState &state);

private:
  void step_cpu(AppState &state, EvolutionManager &evolution);
  void step_gpu(AppState &state, EvolutionManager &evolution);

  void ensure_gpu_capacity(std::size_t predator_count, std::size_t prey_count, std::size_t food_count);
  void collect_gpu_step_events(AppState &state, const std::vector<uint8_t> &was_food_active);
  void reproduction(AppState &state, EvolutionManager &evolution, AgentRegistry &registry);
  void refresh_world_state_after_step(AppState &state);

  const SimulationConfig &config_;

  // GPU support
  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
};

} // namespace moonai
