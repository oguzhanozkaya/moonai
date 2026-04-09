#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

class EvolutionManager;

namespace reproduction {

void run(AppState &state, EvolutionManager &evolution, AgentRegistry &registry, const SimulationConfig &config);

} // namespace reproduction

} // namespace moonai