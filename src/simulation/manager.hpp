#pragma once

#include "core/config.hpp"

#include <cstddef>
#include <memory>

namespace moonai {

struct AppState;
struct AgentRegistry;
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

  void enable_gpu(AppState &state, bool enable);
  void disable_gpu(AppState &state);

private:
  const SimulationConfig &config_;

  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
};

} // namespace moonai
