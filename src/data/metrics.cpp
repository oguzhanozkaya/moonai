#include "data/metrics.hpp"

#include "core/app_state.hpp"

namespace moonai::metrics {

namespace {

int count_active_food(const Food &food) {
  int active_food = 0;
  for (uint8_t active : food.active) {
    active_food += active ? 1 : 0;
  }
  return active_food;
}

} // namespace

void begin_step(AppState &state) {
  state.metrics.step_delta.clear();
}

void refresh_live(AppState &state) {
  MetricsSnapshot &live = state.metrics.live;
  live.clear();
  live.step = state.runtime.step;
  live.predator_count = static_cast<int>(state.predator.size());
  live.prey_count = static_cast<int>(state.prey.size());
  live.active_food = count_active_food(state.food);
  live.predator_species = static_cast<int>(state.predator.species.size());
  live.prey_species = static_cast<int>(state.prey.species.size());

  // Calculate average energy for predators
  float predator_energy_sum = 0.0f;
  for (float energy : state.predator.energy) {
    predator_energy_sum += energy;
  }
  if (!state.predator.energy.empty()) {
    live.avg_predator_energy = predator_energy_sum / static_cast<float>(state.predator.energy.size());
  }

  // Calculate average energy for prey
  float prey_energy_sum = 0.0f;
  for (float energy : state.prey.energy) {
    prey_energy_sum += energy;
  }
  if (!state.prey.energy.empty()) {
    live.avg_prey_energy = prey_energy_sum / static_cast<float>(state.prey.energy.size());
  }

  // Calculate average complexity for predators
  float predator_complexity_sum = 0.0f;
  for (const auto &genome : state.predator.genomes) {
    predator_complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
  }
  if (!state.predator.genomes.empty()) {
    live.avg_predator_complexity = predator_complexity_sum / static_cast<float>(state.predator.genomes.size());
  }

  // Calculate average complexity for prey
  float prey_complexity_sum = 0.0f;
  for (const auto &genome : state.prey.genomes) {
    prey_complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
  }
  if (!state.prey.genomes.empty()) {
    live.avg_prey_complexity = prey_complexity_sum / static_cast<float>(state.prey.genomes.size());
  }

  // Calculate generation metrics for predators
  int max_pred_gen = 0;
  int64_t pred_gen_sum = 0;
  for (int gen : state.predator.generation) {
    max_pred_gen = std::max(max_pred_gen, gen);
    pred_gen_sum += gen;
  }
  if (!state.predator.generation.empty()) {
    live.max_predator_generation = max_pred_gen;
    live.avg_predator_generation =
        static_cast<float>(pred_gen_sum) / static_cast<float>(state.predator.generation.size());
  }

  // Calculate generation metrics for prey
  int max_prey_gen = 0;
  int64_t prey_gen_sum = 0;
  for (int gen : state.prey.generation) {
    max_prey_gen = std::max(max_prey_gen, gen);
    prey_gen_sum += gen;
  }
  if (!state.prey.generation.empty()) {
    live.max_prey_generation = max_prey_gen;
    live.avg_prey_generation = static_cast<float>(prey_gen_sum) / static_cast<float>(state.prey.generation.size());
  }
}

void finalize_step(AppState &state) {
  refresh_live(state);
  state.metrics.report_window.step = state.metrics.live.step;
  state.metrics.report_window.add_events(state.metrics.step_delta);
  state.metrics.totals.step = state.metrics.live.step;
  state.metrics.totals.add_events(state.metrics.step_delta);
}

void record_report(AppState &state) {
  MetricsSnapshot report;
  report.step = state.metrics.live.step;
  report.predator_count = state.metrics.live.predator_count;
  report.prey_count = state.metrics.live.prey_count;
  report.active_food = state.metrics.live.active_food;
  report.kills = state.metrics.report_window.kills;
  report.food_eaten = state.metrics.report_window.food_eaten;
  report.predator_births = state.metrics.report_window.predator_births;
  report.prey_births = state.metrics.report_window.prey_births;
  report.predator_deaths = state.metrics.report_window.predator_deaths;
  report.prey_deaths = state.metrics.report_window.prey_deaths;
  report.predator_species = state.metrics.live.predator_species;
  report.prey_species = state.metrics.live.prey_species;
  report.avg_predator_complexity = state.metrics.live.avg_predator_complexity;
  report.avg_prey_complexity = state.metrics.live.avg_prey_complexity;
  report.avg_predator_energy = state.metrics.live.avg_predator_energy;
  report.avg_prey_energy = state.metrics.live.avg_prey_energy;
  report.max_predator_generation = state.metrics.live.max_predator_generation;
  report.avg_predator_generation = state.metrics.live.avg_predator_generation;
  report.max_prey_generation = state.metrics.live.max_prey_generation;
  report.avg_prey_generation = state.metrics.live.avg_prey_generation;

  state.metrics.last_report = report;
  state.metrics.has_last_report = true;
  state.metrics.report_window.clear_events();
}

} // namespace moonai::metrics
