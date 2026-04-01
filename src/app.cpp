#include "app.hpp"

#include "core/profiler_macros.hpp"
#include "data/metrics.hpp"
#include "simulation/simulation_step_systems.hpp"
#include "visualization/frame_snapshot.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace moonai {

volatile std::sig_atomic_t App::g_running_ = 1;

void App::signal_handler(int) {
  g_running_ = 0;
}

void App::register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    registered = true;
  }
  g_running_ = 1;
}

std::uint64_t App::generate_seed() {
  return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
}

App::App(AppConfig cfg)
    : cfg_([&cfg] {
        if (cfg.sim_config.seed == 0) {
          cfg.sim_config.seed = App::generate_seed();
        }
        return std::move(cfg);
      }()),
      state_(cfg_.sim_config.seed), simulation_(cfg_.sim_config), evolution_(cfg_.sim_config),
      logger_(cfg_.sim_config.output_dir, cfg_.sim_config.seed, cfg_.run_name_override.value_or(cfg_.experiment_name)) {
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  state_.ui.speed_multiplier = cfg_.speed_multiplier;

  simulation_.initialize(state_);
  evolution_.initialize(state_, simulation_detail::SENSOR_COUNT, simulation_detail::OUTPUT_COUNT);
  evolution_.seed_initial_population(state_);
  state_.runtime.gpu_enabled = cfg_.enable_gpu;
  metrics::refresh_live(state_);

  logger_.initialize(cfg_.sim_config);

  if (!cfg_.headless) {
    visualization_ = std::make_unique<VisualizationManager>(cfg_.sim_config, state_.ui);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  if (state_.runtime.gpu_enabled) {
    evolution_.enable_gpu(state_, true);
    simulation_.enable_gpu(state_, true);
    spdlog::info("GPU acceleration enabled");
  }

  register_signal_handlers();
}

void App::step() {
  MOONAI_PROFILE_SCOPE("step");

  if (state_.runtime.gpu_enabled) {
    simulation_.step_gpu(state_, evolution_);
  } else {
    simulation_.step(state_, evolution_);
  }

  for (const auto &pair : state_.runtime.pending_predator_offspring) {
    const uint32_t child =
        evolution_.create_predator_offspring(state_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      ++state_.runtime.step_events.births;
    }
  }

  for (const auto &pair : state_.runtime.pending_prey_offspring) {
    const uint32_t child = evolution_.create_prey_offspring(state_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      ++state_.runtime.step_events.births;
    }
  }
  state_.runtime.pending_predator_offspring.clear();
  state_.runtime.pending_prey_offspring.clear();

  state_.runtime.report_events.add(state_.runtime.step_events);
  state_.runtime.total_events.add(state_.runtime.step_events);

  if (cfg_.sim_config.species_update_interval_steps > 0 &&
      (state_.runtime.step % cfg_.sim_config.species_update_interval_steps) == 0) {
    evolution_.refresh_species(state_);
  }

  ++state_.runtime.step;
  metrics::refresh_live(state_);
}

void App::record_and_log() {
  evolution_.refresh_species(state_);
  metrics::record_report(state_);

  logger_.log_report(state_.metrics.last_report);

  const Genome *best_genome = nullptr;
  std::size_t best_complexity = 0;
  for (const auto &genome : state_.predator.genomes) {
    const std::size_t complexity = genome.nodes().size() + genome.connections().size();
    if (!best_genome || complexity > best_complexity) {
      best_genome = &genome;
      best_complexity = complexity;
    }
  }
  for (const auto &genome : state_.prey.genomes) {
    const std::size_t complexity = genome.nodes().size() + genome.connections().size();
    if (!best_genome || complexity > best_complexity) {
      best_genome = &genome;
      best_complexity = complexity;
    }
  }

  if (best_genome) {
    logger_.log_best_genome(state_.runtime.step, *best_genome);
  }

  logger_.log_species(state_.runtime.step, state_.predator.species, "predator");
  logger_.log_species(state_.runtime.step, state_.prey.species, "prey");
  logger_.flush();
  state_.runtime.report_events.clear();

  spdlog::info("Step {:6d}: predators={} prey={} births={} deaths={} "
               "pred_species={} prey_species={}",
               state_.metrics.last_report.step, state_.metrics.last_report.predator_count,
               state_.metrics.last_report.prey_count, state_.metrics.last_report.births,
               state_.metrics.last_report.deaths, state_.metrics.last_report.predator_species,
               state_.metrics.last_report.prey_species);
}

bool App::should_continue() const {
  return !(g_running_ == 0) && !(cfg_.sim_config.max_steps > 0 && state_.runtime.step >= cfg_.sim_config.max_steps);
}

bool App::run() {
  if (!cfg_.headless && std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr) {
    spdlog::error("No display server found. GUI mode requires a display.");
    return false;
  }
  if (!cfg_.headless && !visualization_) {
    spdlog::error("Visualization requested but not initialized");
    return true;
  }

  bool completed = true;

  if (cfg_.headless) {
    while (should_continue()) {
      step();

      if (state_.runtime.step % cfg_.sim_config.report_interval_steps == 0) {
        record_and_log();
      }
    }

    completed = (g_running_ != 0);
  } else {
    while (should_continue()) {
      MOONAI_PROFILE_SCOPE("frame_total");

      visualization_->handle_events();
      if (visualization_->should_close()) {
        completed = false;
        break;
      }

      int steps_to_run = state_.ui.paused ? state_.ui.step_requested : std::max(1, state_.ui.speed_multiplier);
      state_.ui.step_requested = false;
      for (int i = 0; i < steps_to_run && should_continue(); ++i) {
        step();

        if (state_.runtime.step % cfg_.sim_config.report_interval_steps == 0) {
          record_and_log();
        }
      }

      visualization_->render(build_frame_snapshot(state_, cfg_));
    }
  }

  if (state_.metrics.history.empty() || state_.metrics.history.back().step != state_.runtime.step) {
    record_and_log();
  }

  logger_.flush();

  if (!completed) {
    spdlog::info("Simulation stopped by user (window closed)");
  }

  spdlog::info("Output saved to: {}", logger_.run_dir());

  return completed;
}

} // namespace moonai
