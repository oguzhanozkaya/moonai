#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"

namespace moonai {

void EvolutionManager::compute_actions_for_population(
    PopulationEvolutionState &population, AgentRegistry &agents) const {
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());

  std::vector<float> all_inputs;
  all_inputs.reserve(entity_count * AgentRegistry::INPUT_COUNT);

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    const float *input_ptr = agents.input_ptr(idx);
    all_inputs.insert(all_inputs.end(), input_ptr,
                      input_ptr + AgentRegistry::INPUT_COUNT);
  }

  std::vector<float> all_outputs;
  population.network_cache.activate_batch(entity_count, all_inputs, all_outputs,
                                          AgentRegistry::INPUT_COUNT,
                                          AgentRegistry::OUTPUT_COUNT);

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    agents.decision_x[idx] = all_outputs[idx * AgentRegistry::OUTPUT_COUNT];
    agents.decision_y[idx] = all_outputs[idx * AgentRegistry::OUTPUT_COUNT + 1];
  }
}

void EvolutionManager::compute_actions(AppState &state) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  compute_actions_for_population(state.evolution.predators, state.predators);
  compute_actions_for_population(state.evolution.prey, state.prey);
}

} // namespace moonai