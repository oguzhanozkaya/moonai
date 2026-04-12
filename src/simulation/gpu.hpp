#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "gpu/gpu_batch.hpp"

#include <memory>

namespace moonai {

class EvolutionManager;

namespace simulation {
namespace gpu {

void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config);
void disable(std::unique_ptr<moonai::gpu::GpuBatch> &batch);

} // namespace gpu
} // namespace simulation

} // namespace moonai
