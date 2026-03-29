#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "gpu/gpu_network_cache.hpp"

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig &config)
    : config_(config) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize(AppState &state, int num_inputs,
                                  int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;

  state.evolution.innovation_tracker = InnovationTracker();
  state.evolution.innovation_tracker.set_counters(
      0, static_cast<std::uint32_t>(num_inputs_ + num_outputs_ + 1));
  state.evolution.species.clear();
  state.evolution.entity_genomes.clear();
  state.evolution.network_cache.clear();

  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

} // namespace moonai
