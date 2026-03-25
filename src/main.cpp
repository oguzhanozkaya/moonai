#include "core/config.hpp"
#include "core/random.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/physics.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
  g_running = 0;
}

int run_experiment(const std::string &name, moonai::SimulationConfig config,
                   const moonai::CLIArgs &args) {
  if (args.seed_override != 0) {
    config.seed = args.seed_override;
  }
  if (args.max_steps_override != 0) {
    config.max_steps = args.max_steps_override;
  }

  const auto errors = moonai::validate_config(config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    return 1;
  }

  if (config.seed == 0) {
    config.seed = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }

  moonai::Random rng(config.seed);
  moonai::SimulationManager simulation(config);
  moonai::EvolutionManager evolution(config, rng);
  moonai::Logger logger(config.output_dir, config.seed,
                        args.run_name.empty() ? name : args.run_name);
  moonai::MetricsCollector metrics;

  simulation.initialize();
  evolution.initialize(moonai::SensorInput::SIZE, 2);
  evolution.enable_gpu(!args.no_gpu);
  evolution.seed_initial_population(simulation);
  logger.initialize(config);

  bool headless = args.headless;
  if (!headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    headless = true;
    spdlog::warn("No display server found; switching to headless mode.");
  }

  moonai::VisualizationManager visualization(config);
  if (!headless) {
    visualization.initialize();
  }

  const float dt = 1.0f / static_cast<float>(config.target_fps);
  int steps_executed = 0;
  int births_in_window = 0;
  int deaths_in_window = 0;
  std::vector<moonai::Vec2> actions;

  auto record_window = [&]() {
    evolution.refresh_species(simulation);
    auto snapshot =
        metrics.collect(steps_executed, simulation.agents(), births_in_window,
                        deaths_in_window, evolution.species_count());
    logger.log_report(snapshot);
    const auto &agents = simulation.agents();
    const auto best = std::max_element(
        agents.begin(), agents.end(), [](const auto &lhs, const auto &rhs) {
          return lhs->genome().fitness() < rhs->genome().fitness();
        });
    if (best != agents.end()) {
      logger.log_best_genome(steps_executed, (*best)->genome());
    }
    logger.log_species(steps_executed, evolution.species());
    logger.flush();
    births_in_window = 0;
    deaths_in_window = 0;
    return snapshot;
  };

  auto update_selected_visualization = [&]() {
    const int selected_id = visualization.selected_agent();
    if (selected_id < 0) {
      return;
    }
    const std::size_t slot =
        simulation.slot_for_id(static_cast<moonai::AgentId>(selected_id));
    moonai::NeuralNetwork *network =
        evolution.network_at(simulation, static_cast<int>(slot));
    if (slot < simulation.agents().size() &&
        simulation.agents()[slot]->alive() && network != nullptr) {
      network->activate(simulation.get_sensors(slot).to_vector());
      visualization.set_selected_activations(network->last_activations(),
                                             network->node_index_map());
    }
  };

  auto advance_one_step = [&]() {
    // Try GPU full ecology first, fall back to CPU
    bool used_gpu = false;
    if (evolution.gpu_enabled()) {
      used_gpu = evolution.step_gpu(simulation, steps_executed);
    }

    if (!used_gpu) {
      // CPU path: compute actions and step simulation
      evolution.compute_actions(simulation, actions);
      for (std::size_t idx : simulation.alive_agent_indices()) {
        simulation.apply_action(idx, actions[idx], dt);
      }
      simulation.step(dt);
    }

    // Reproduction and species management always on CPU
    const auto pairs = simulation.find_reproduction_pairs();
    for (const auto &pair : pairs) {
      evolution.create_offspring(simulation, pair.parent_a, pair.parent_b,
                                 pair.spawn_position);
    }

    evolution.refresh_fitness(simulation);
    if (config.species_update_interval_steps > 0 &&
        (steps_executed % config.species_update_interval_steps) == 0) {
      evolution.refresh_species(simulation);
    }

    int step_births = 0;
    int step_deaths = 0;
    for (const auto &event : simulation.last_events()) {
      if (event.type == moonai::SimEvent::Birth) {
        ++step_births;
      } else if (event.type == moonai::SimEvent::Death) {
        ++step_deaths;
      }
    }
    births_in_window += step_births;
    deaths_in_window += step_deaths;

    ++steps_executed;

    if (config.step_log_enabled &&
        (steps_executed % config.step_log_interval) == 0) {
      logger.log_step(steps_executed, simulation.agents());
    }
    logger.log_events(steps_executed, simulation.last_events());

    if ((steps_executed % config.report_interval_steps) == 0) {
      const auto snapshot = record_window();
      spdlog::info(
          "Step {:>6d}: predators={} prey={} births={} deaths={} species={}",
          steps_executed, simulation.alive_predators(), simulation.alive_prey(),
          snapshot.births, snapshot.deaths, evolution.species_count());
    }
  };

  while (g_running &&
         (config.max_steps == 0 || steps_executed < config.max_steps)) {
    if (headless) {
      advance_one_step();
      continue;
    }

    visualization.handle_events();
    if (visualization.should_close()) {
      break;
    }

    if (visualization.is_paused() && !visualization.should_step()) {
      visualization.render(simulation, evolution);
      continue;
    }
    visualization.clear_step();

    const int frame_steps = std::max(1, visualization.speed_multiplier());
    for (int i = 0;
         i < frame_steps &&
         (config.max_steps == 0 || steps_executed < config.max_steps) &&
         g_running;
         ++i) {
      advance_one_step();
    }

    update_selected_visualization();

    visualization.render(simulation, evolution);
  }

  if (births_in_window > 0 || deaths_in_window > 0 ||
      metrics.history().empty()) {
    record_window();
  }

  logger.flush();
  spdlog::info("Output saved to: {}", logger.run_dir());
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  const auto args = moonai::parse_args(argc, argv);
  if (args.help) {
    moonai::print_usage(argv[0]);
    return 0;
  }

  spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  auto configs = moonai::load_all_configs_lua(args.config_path);
  if (configs.empty()) {
    spdlog::error("No configs loaded from '{}'", args.config_path);
    return 1;
  }

  if (args.list_experiments) {
    for (const auto &[name, _] : configs) {
      std::printf("%s\n", name.c_str());
    }
    return 0;
  }

  if (args.validate_only) {
    moonai::SimulationConfig config;
    if (!args.experiment_name.empty()) {
      config = configs.at(args.experiment_name);
    } else {
      config = configs.begin()->second;
    }
    const auto errors = moonai::validate_config(config);
    if (errors.empty()) {
      std::printf("OK\n");
      return 0;
    }
    for (const auto &error : errors) {
      std::fprintf(stderr, "ERROR [%s]: %s\n", error.field.c_str(),
                   error.message.c_str());
    }
    return 1;
  }

  if (args.run_all) {
    int failures = 0;
    for (const auto &[name, config] : configs) {
      failures += run_experiment(name, config, args);
    }
    return failures == 0 ? 0 : 1;
  }

  std::string selected = args.experiment_name;
  if (selected.empty()) {
    selected = configs.begin()->first;
    if (configs.size() > 1) {
      spdlog::warn("Multiple experiments found; using '{}'.", selected);
    }
  }

  auto it = configs.find(selected);
  if (it == configs.end()) {
    spdlog::error("Experiment '{}' not found.", selected);
    return 1;
  }

  return run_experiment(selected, it->second, args);
}
