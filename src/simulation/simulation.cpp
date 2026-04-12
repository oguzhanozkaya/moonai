#include "simulation/simulation.hpp"

#include "evolution/evolution_manager.hpp"
#include "simulation/common.hpp"
#include "simulation/cpu.hpp"
#include "simulation/gpu.hpp"

#include <spdlog/spdlog.h>

namespace moonai {

namespace simulation {

void initialize(AppState &state, const SimulationConfig &config) {
  state.food.initialize(config, state.runtime.rng);
}

void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config) {
  if (state.runtime.gpu_enabled) {
    gpu::step(state, evolution, state.gpu_batch, config);
  } else {
    cpu::step(state, evolution, config);
  }

  common::run(state, evolution, config);
}

} // namespace simulation

} // namespace moonai
