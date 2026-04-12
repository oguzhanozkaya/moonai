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

} // namespace simulation

} // namespace moonai
