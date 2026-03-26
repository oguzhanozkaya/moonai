#include "core/config.hpp"
#include "core/random.hpp"
#include "data/logger.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/components.hpp"
#include "simulation/session.hpp"
#include "visualization/visualization_manager.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
  g_running = 0;
}

int run_experiment(const std::string &name, moonai::SimulationConfig config,
                   const moonai::CLIArgs &args) {
  // Check for display
  bool headless = args.headless;
  if (!headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    headless = true;
    spdlog::warn("No display server found; switching to headless mode.");
  }

  // Build SessionConfig
  moonai::SessionConfig session_cfg;
  session_cfg.sim_config = config;
  session_cfg.experiment_name = name;
  session_cfg.output_dir = config.output_dir;
  session_cfg.seed = args.seed_override != 0 ? args.seed_override : config.seed;
  session_cfg.headless = headless;
  session_cfg.enable_gpu = !args.no_gpu;
  session_cfg.enable_logger = true;
  session_cfg.run_name_override =
      args.run_name.empty() ? std::nullopt : std::optional(args.run_name);

  if (args.max_steps_override != 0) {
    session_cfg.sim_config.max_steps = args.max_steps_override;
  }

  // Create Session
  moonai::Session session(session_cfg);

  const float dt = 1.0f / static_cast<float>(config.target_fps);
  auto *visualization = session.visualization();

  auto update_selected_visualization = [&session, visualization]() -> void {
    if (!visualization)
      return;

    moonai::Entity selected = visualization->selected_entity();
    auto &registry = session.registry();
    if (selected == moonai::INVALID_ENTITY || !registry.valid(selected)) {
      return;
    }

    size_t idx = registry.index_of(selected);
    const auto *genome = session.evolution().genome_for(selected);
    if (!genome || !registry.vitals().alive[idx]) {
      return;
    }

    const float *sensors = registry.sensors().input_ptr(idx);
    std::vector<float> sensor_vec(sensors,
                                  sensors + moonai::SensorSoA::INPUT_COUNT);

    moonai::NeuralNetwork *network =
        session.evolution().network_cache().get_network(selected);
    if (network) {
      network->activate(sensor_vec);
      visualization->set_selected_activations(network->last_activations(),
                                              network->node_index_map());
    }
  };

  auto advance_one_step = [&session, dt, &config]() -> void {
    session.step(dt);

    if (session.steps_executed() % config.report_interval_steps == 0) {
      auto snapshot = session.record_and_log(session.births_in_window(),
                                             session.deaths_in_window());
      spdlog::info(
          "Step {:>6d}: predators={} prey={} births={} deaths={} species={}",
          session.steps_executed(), snapshot.predator_count,
          snapshot.prey_count, snapshot.births, snapshot.deaths,
          snapshot.num_species);
    }
  };

  while (g_running && (config.max_steps == 0 ||
                       session.steps_executed() < config.max_steps)) {
    if (headless) {
      advance_one_step();
      continue;
    }

    visualization->handle_events();
    if (visualization->should_close()) {
      break;
    }

    if (visualization->is_paused() && !visualization->should_step()) {
      visualization->render_ecs(session.registry(), session.evolution(),
                                session.simulation(), session.steps_executed());
      continue;
    }
    visualization->clear_step();

    const int frame_steps = std::max(1, visualization->speed_multiplier());
    for (int i = 0; i < frame_steps &&
                    (config.max_steps == 0 ||
                     session.steps_executed() < config.max_steps) &&
                    g_running;
         ++i) {
      advance_one_step();
    }

    update_selected_visualization();

    visualization->render_ecs(session.registry(), session.evolution(),
                              session.simulation(), session.steps_executed());
  }

  // Final record if needed
  if (session.births_in_window() > 0 || session.deaths_in_window() > 0 ||
      session.metrics().history().empty()) {
    session.record_and_log(session.births_in_window(),
                           session.deaths_in_window());
  }

  if (session.logger()) {
    session.logger()->flush();
    spdlog::info("Output saved to: {}", session.logger()->run_dir());
  }
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
    for (const auto &[config_name, _] : configs) {
      std::printf("%s\n", config_name.c_str());
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
    for (const auto &[config_name, config] : configs) {
      failures += run_experiment(config_name, config, args);
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
