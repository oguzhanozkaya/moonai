#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

class EvolutionManager;

namespace simulation {
namespace cpu {

void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config);

} // namespace cpu
} // namespace simulation

} // namespace moonai
