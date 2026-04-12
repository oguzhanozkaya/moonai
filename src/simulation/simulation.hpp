#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "gpu/gpu_batch.hpp"

#include <cstddef>
#include <memory>

namespace moonai {

class EvolutionManager;

namespace simulation {

void initialize(AppState &state, const SimulationConfig &config);
void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config);

void enable_gpu(AppState &state, bool enable, const SimulationConfig &config);
void disable_gpu(AppState &state);

} // namespace simulation

} // namespace moonai
