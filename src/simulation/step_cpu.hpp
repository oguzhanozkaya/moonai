#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

class EvolutionManager;

namespace cpu_backend {

void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config);

} // namespace cpu_backend

} // namespace moonai
