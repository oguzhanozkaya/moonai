#include "simulation/step_cpu.hpp"

#include "core/metrics.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/systems.hpp"

#include <vector>

namespace moonai::cpu_backend {

void step(AppState &state, EvolutionManager &evolution, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");

  const std::vector<uint8_t> was_food_active = state.food.active;

  std::vector<int> food_consumed_by(state.food.size(), -1);
  std::vector<int> killed_by(state.prey.size(), -1);
  std::vector<uint32_t> kill_counts(state.predator.size(), 0U);

  std::vector<float> predator_sensors;
  std::vector<float> prey_sensors;
  systems::build_sensors(state.predator, state.predator, state.prey, state.food, config, config.predator_speed,
                         predator_sensors);
  systems::build_sensors(state.prey, state.predator, state.prey, state.food, config, config.prey_speed, prey_sensors);

  std::vector<float> predator_decisions;
  std::vector<float> prey_decisions;
  evolution.compute_actions(state, predator_sensors, prey_sensors, predator_decisions, prey_decisions);

  systems::update_vitals(state.predator, config);
  systems::update_vitals(state.prey, config);
  systems::process_food(state.prey, state.food, config, food_consumed_by);
  systems::process_combat(state.predator, state.prey, config, killed_by, kill_counts);
  systems::apply_movement(state.predator, config, config.predator_speed, predator_decisions);
  systems::apply_movement(state.prey, config, config.prey_speed, prey_decisions);

  systems::collect_food_events(state.prey, state.food, was_food_active, food_consumed_by, state.metrics.step_delta);
  systems::collect_combat_events(state.predator, state.prey, killed_by, kill_counts, state.metrics.step_delta);
  systems::collect_death_events(state.predator, state.metrics.step_delta);
  systems::collect_death_events(state.prey, state.metrics.step_delta);
}

} // namespace moonai::cpu_backend
