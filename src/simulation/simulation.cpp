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

void enable_gpu(AppState &state, bool enable, const SimulationConfig &config) {
  if (enable) {
    gpu::ensure_capacity(state.gpu_batch, static_cast<std::size_t>(config.predator_count),
                         static_cast<std::size_t>(config.prey_count), static_cast<std::size_t>(config.food_count));
    state.runtime.gpu_enabled = true;
  } else {
    gpu::disable(state.gpu_batch);
    state.runtime.gpu_enabled = false;
  }
}

} // namespace simulation

} // namespace moonai
