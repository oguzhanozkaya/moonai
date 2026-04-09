#include "simulation/manager.hpp"

#include "core/app_state.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/reproduction.hpp"
#include "simulation/step_cpu.hpp"
#include "simulation/step_gpu.hpp"

#include <spdlog/spdlog.h>

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config) : config_(config) {}

SimulationManager::~SimulationManager() = default;

void SimulationManager::initialize(AppState &state) {
  state.food.initialize(config_, state.runtime.rng);
}

void SimulationManager::step(AppState &state, EvolutionManager &evolution) {
  if (state.runtime.gpu_enabled) {
    gpu_backend::step(state, evolution, gpu_batch_, config_);
  } else {
    cpu_backend::step(state, evolution, config_);
  }

  state.predator.compact();
  state.prey.compact();
  state.food.respawn_step(config_, state.runtime.step, state.runtime.rng.seed());

  reproduction::run(state, evolution, state.predator, config_);
  reproduction::run(state, evolution, state.prey, config_);
  if (config_.species_update_interval_steps > 0 && (state.runtime.step % config_.species_update_interval_steps) == 0) {
    evolution.refresh_species(state);
  }
}

void SimulationManager::reset(AppState &state) {
  initialize(state);
}

void SimulationManager::enable_gpu(AppState &state, bool enable) {
  if (enable) {
    gpu_backend::ensure_capacity(gpu_batch_, static_cast<std::size_t>(config_.predator_count),
                                 static_cast<std::size_t>(config_.prey_count),
                                 static_cast<std::size_t>(config_.food_count));
    state.runtime.gpu_enabled = true;
  } else {
    disable_gpu(state);
  }
}

void SimulationManager::disable_gpu(AppState &state) {
  gpu_backend::disable(gpu_batch_);
  state.runtime.gpu_enabled = false;
}

} // namespace moonai