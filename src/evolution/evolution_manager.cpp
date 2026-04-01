#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "gpu/gpu_network_cache.hpp"

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig &config) : config_(config) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize_population(AgentRegistry &registry) const {
  registry.innovation_tracker = InnovationTracker();
  registry.innovation_tracker.set_counters(0, static_cast<std::uint32_t>(num_inputs_ + num_outputs_ + 1));
  registry.species.clear();
  registry.genomes.clear();
  registry.network_cache.clear();
}

void EvolutionManager::initialize(AppState &state, int num_inputs, int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;

  Species::reset_id_counter();
  initialize_population(state.predator);
  initialize_population(state.prey);

  if (predator_gpu_network_cache_) {
    predator_gpu_network_cache_->invalidate();
  }
  if (prey_gpu_network_cache_) {
    prey_gpu_network_cache_->invalidate();
  }
}

} // namespace moonai
