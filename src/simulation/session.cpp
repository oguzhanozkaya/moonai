#include "simulation/session.hpp"

#include "data/logger.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/genome.hpp"
#include "simulation/components.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>

namespace moonai {

Session::Session(const SessionConfig &cfg)
    : cfg_(cfg),
      rng_(
          cfg.seed == 0
              ? static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count())
              : cfg.seed),
      simulation_(cfg.sim_config), evolution_(cfg.sim_config, rng_),
      steps_executed_(0), births_in_window_(0), deaths_in_window_(0) {
  // Validate config
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  // Initialize simulation and evolution
  simulation_.initialize();
  evolution_.initialize(SensorSoA::INPUT_COUNT, 2);
  evolution_.seed_initial_population_ecs(registry_);
  simulation_.refresh_state_ecs(registry_);

  // Initialize logger if enabled
  if (cfg_.enable_logger) {
    std::string run_name =
        cfg_.run_name_override.value_or(cfg_.experiment_name);
    logger_ = std::make_unique<Logger>(cfg_.output_dir, cfg_.seed, run_name);
    logger_->initialize(cfg_.sim_config);
  }

  // Initialize visualization if not headless
  if (!cfg_.headless) {
    visualization_ = std::make_unique<VisualizationManager>(cfg_.sim_config);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  // Enable GPU if available
  evolution_.enable_gpu(cfg_.enable_gpu);
}

Session::~Session() = default;

Registry &Session::registry() {
  return registry_;
}
SimulationManager &Session::simulation() {
  return simulation_;
}
EvolutionManager &Session::evolution() {
  return evolution_;
}
MetricsCollector &Session::metrics() {
  return metrics_;
}
Logger *Session::logger() {
  return logger_.get();
}
VisualizationManager *Session::visualization() {
  return visualization_.get();
}

const Registry &Session::registry() const {
  return registry_;
}
const SimulationManager &Session::simulation() const {
  return simulation_;
}
const EvolutionManager &Session::evolution() const {
  return evolution_;
}
const MetricsCollector &Session::metrics() const {
  return metrics_;
}

int Session::steps_executed() const {
  return steps_executed_;
}
int Session::births_in_window() const {
  return births_in_window_;
}
int Session::deaths_in_window() const {
  return deaths_in_window_;
}

void Session::record_birth() {
  ++births_in_window_;
}
void Session::record_death() {
  ++deaths_in_window_;
}
void Session::reset_window_counters() {
  births_in_window_ = 0;
  deaths_in_window_ = 0;
}

void Session::step(float dt) {
  // Compute actions
  actions_buffer_.clear();
  evolution_.compute_actions_ecs(registry_, actions_buffer_);

  // Apply actions to entities
  size_t action_idx = 0;
  for (Entity e : registry_.living_entities()) {
    size_t idx = registry_.index_of(e);
    if (!registry_.vitals().alive[idx]) {
      continue;
    }

    if (action_idx < actions_buffer_.size()) {
      float dx = actions_buffer_[action_idx].x;
      float dy = actions_buffer_[action_idx].y;
      float speed = registry_.motion().speed[idx];

      registry_.motion().vel_x[idx] = dx * speed;
      registry_.motion().vel_y[idx] = dy * speed;
      registry_.positions().x[idx] += registry_.motion().vel_x[idx] * dt;
      registry_.positions().y[idx] += registry_.motion().vel_y[idx] * dt;
      registry_.stats().distance_traveled[idx] +=
          std::sqrt(dx * dx + dy * dy) * speed * dt;

      ++action_idx;
    }
  }

  // Step simulation
  simulation_.step_ecs(registry_, dt);

  // Handle reproduction
  auto pairs = simulation_.find_reproduction_pairs_ecs(registry_);
  for (const auto &pair : pairs) {
    Entity child = evolution_.create_offspring_ecs(
        registry_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      ++births_in_window_;
    }
  }

  // Refresh fitness
  evolution_.refresh_fitness_ecs(registry_);

  // Update species periodically
  if (cfg_.sim_config.species_update_interval_steps > 0 &&
      (steps_executed_ % cfg_.sim_config.species_update_interval_steps) == 0) {
    evolution_.refresh_species_ecs(registry_);
  }

  // Record deaths
  for (const auto &event : simulation_.last_events()) {
    if (event.type == SimEvent::Death) {
      ++deaths_in_window_;
    }
  }

  ++steps_executed_;

  // Log step if enabled
  if (logger_) {
    if (cfg_.sim_config.step_log_enabled &&
        (steps_executed_ % cfg_.sim_config.step_log_interval) == 0) {
      logger_->log_step(steps_executed_, registry_);
    }
    logger_->log_events(steps_executed_, simulation_.last_events());
  }
}

StepMetrics Session::record_and_log(int births, int deaths) {
  evolution_.refresh_species_ecs(registry_);

  float best_predator = 0.0f, avg_predator = 0.0f;
  float best_prey = 0.0f, avg_prey = 0.0f;
  evolution_.get_fitness_by_type_ecs(registry_, best_predator, avg_predator,
                                     best_prey, avg_prey);

  int alive_predators = 0, alive_prey = 0;
  for (Entity e : registry_.living_entities()) {
    size_t idx = registry_.index_of(e);
    if (registry_.vitals().alive[idx]) {
      if (registry_.identity().type[idx] == IdentitySoA::TYPE_PREDATOR) {
        ++alive_predators;
      } else {
        ++alive_prey;
      }
    }
  }

  auto snapshot =
      metrics_.collect_ecs(steps_executed_, registry_, evolution_, births,
                           deaths, evolution_.species_count());

  if (logger_) {
    logger_->log_report(snapshot);

    // Find and log best genome
    const Genome *best_genome = nullptr;
    float best_fitness = -std::numeric_limits<float>::infinity();
    for (Entity e : registry_.living_entities()) {
      const auto *genome = evolution_.genome_for(e);
      if (genome && genome->fitness() > best_fitness) {
        best_fitness = genome->fitness();
        best_genome = genome;
      }
    }
    if (best_genome) {
      logger_->log_best_genome(steps_executed_, *best_genome);
    }

    logger_->log_species(steps_executed_, evolution_.species());
    logger_->flush();
  }

  reset_window_counters();
  return snapshot;
}

} // namespace moonai
