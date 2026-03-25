#include "core/config.hpp"
#include "core/profiler.hpp"
#include "core/profiler_suite.hpp"
#include "core/random.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/physics.hpp"
#include "simulation/simulation_manager.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace {

struct Args {
  std::string profiler_config = "profiler.lua";
  std::string suite_name;
  bool no_gpu = false;
};

Args parse_args(int argc, const char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--suite" && i + 1 < argc) {
      args.suite_name = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    } else if (!arg.empty() && arg[0] != '-') {
      args.profiler_config = arg;
    }
  }
  return args;
}

struct SuiteRunSummary {
  std::uint64_t seed = 0;
  std::string run_dir;
  std::string profile_path;
  double avg_window_ms = 0.0;
  double run_total_ms = 0.0;
  int window_count = 0;
  std::string disposition = "kept";
  nlohmann::json profile;
};

double json_number(const nlohmann::json &value) {
  return value.is_number() ? value.get<double>() : 0.0;
}

template <typename Extractor>
nlohmann::json aggregate_named_stats(const std::vector<SuiteRunSummary *> &runs,
                                     Extractor extractor) {
  nlohmann::json aggregated = nlohmann::json::object();
  if (runs.empty()) {
    return aggregated;
  }

  const auto &first = extractor(runs.front()->profile);
  if (!first.is_object()) {
    return aggregated;
  }

  for (auto it = first.begin(); it != first.end(); ++it) {
    if (!it.value().is_object()) {
      continue;
    }
    nlohmann::json stats = nlohmann::json::object();
    for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
      double sum = 0.0;
      for (const auto *run : runs) {
        const auto &source = extractor(run->profile);
        if (source.contains(it.key()) && source[it.key()].contains(jt.key())) {
          sum += json_number(source[it.key()][jt.key()]);
        }
      }
      stats[jt.key()] = sum / static_cast<double>(runs.size());
    }
    aggregated[it.key()] = std::move(stats);
  }

  return aggregated;
}

SuiteRunSummary run_suite_member(const moonai::ProfilerSuiteConfig &suite,
                                 moonai::SimulationConfig config,
                                 std::uint64_t seed, bool no_gpu) {
  config.seed = seed;
  config.max_steps = suite.windows * config.report_interval_steps;

  moonai::Random rng(config.seed);
  moonai::SimulationManager simulation(config);
  moonai::EvolutionManager evolution(config, rng);
  moonai::MetricsCollector metrics;

  simulation.initialize();
  evolution.initialize(moonai::SensorInput::SIZE, 2);
  evolution.seed_initial_population(simulation);
  evolution.enable_gpu(!no_gpu);

  const std::string suite_dir = suite.output_dir + "/" + suite.name + "_suite";

  moonai::ProfileRunSpec run_spec;
  run_spec.experiment_name =
      suite.experiment_name + "_seed" + std::to_string(config.seed);
  run_spec.output_root_dir = suite_dir + "/raw";
  run_spec.seed = config.seed;
  run_spec.predator_count = config.predator_count;
  run_spec.prey_count = config.prey_count;
  run_spec.food_count = config.food_count;
  run_spec.total_steps = config.max_steps;
  run_spec.report_interval_steps = config.report_interval_steps;
  run_spec.gpu_allowed = !no_gpu;
  run_spec.cuda_compiled = false;
  run_spec.openmp_compiled = true;
  run_spec.suite_name = suite.name;
  run_spec.base_experiment_name = suite.experiment_name;
  run_spec.config_fingerprint = moonai::fingerprint_config(config);
  run_spec.profiler_entry_point = "moonai_profiler";

  auto &profiler = moonai::Profiler::instance();
  profiler.set_enabled(true);
  profiler.start_run(run_spec);

  const auto run_start = std::chrono::steady_clock::now();
  const float dt = 1.0f / static_cast<float>(config.target_fps);
  std::vector<moonai::Vec2> actions;
  int steps_executed = 0;

  for (int window = 0; window < suite.windows; ++window) {
    profiler.start_window(window);
    const int window_end = std::min(
        config.max_steps, steps_executed + config.report_interval_steps);

    while (steps_executed < window_end) {
      evolution.compute_actions(simulation, actions);
      for (std::size_t idx : simulation.alive_agent_indices()) {
        simulation.apply_action(idx, actions[idx], dt);
      }
      simulation.step(dt);

      const auto pairs = simulation.find_reproduction_pairs();
      for (const auto &pair : pairs) {
        evolution.create_offspring(simulation, pair.parent_a, pair.parent_b,
                                   pair.spawn_position);
      }

      evolution.refresh_fitness(simulation);
      ++steps_executed;
      MOONAI_PROFILE_INC(moonai::ProfileCounter::StepsExecuted);
    }

    evolution.refresh_species(simulation);
    const auto snapshot = metrics.collect(steps_executed, simulation.agents(),
                                          0, 0, evolution.species_count());
    profiler.finish_window({window, snapshot.predator_count,
                            snapshot.prey_count, snapshot.num_species,
                            snapshot.best_fitness, snapshot.avg_fitness,
                            snapshot.avg_genome_complexity});
  }

  const auto run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - run_start)
                          .count();
  profiler.finish_run(run_ns);

  SuiteRunSummary summary;
  summary.seed = seed;
  summary.run_dir = profiler.output_dir();
  summary.profile_path = summary.run_dir + "/profile.json";

  std::ifstream file(summary.profile_path);
  file >> summary.profile;
  const auto &profile_summary = summary.profile["summary"];
  summary.avg_window_ms = json_number(
      profile_summary["events"]["window_total"]["avg_ms_per_window"]);
  summary.run_total_ms = json_number(profile_summary["run_total_ms"]);
  summary.window_count = profile_summary["window_count"].get<int>();
  return summary;
}

void write_suite_manifest(const moonai::ProfilerSuiteConfig &suite,
                          const moonai::SimulationConfig &config,
                          std::vector<SuiteRunSummary> runs) {
  const auto suite_dir =
      std::filesystem::path(suite.output_dir) / (suite.name + "_suite");
  std::sort(runs.begin(), runs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.avg_window_ms < rhs.avg_window_ms;
  });

  std::vector<SuiteRunSummary *> kept;
  if (runs.size() > 2) {
    runs.front().disposition = "dropped_fastest";
    runs.back().disposition = "dropped_slowest";
    for (std::size_t i = 1; i + 1 < runs.size(); ++i) {
      kept.push_back(&runs[i]);
    }
  } else {
    for (auto &run : runs) {
      kept.push_back(&run);
    }
  }

  const double avg_window_ms =
      kept.empty() ? 0.0
                   : std::accumulate(kept.begin(), kept.end(), 0.0,
                                     [](double sum, const auto *run) {
                                       return sum + run->avg_window_ms;
                                     }) /
                         static_cast<double>(kept.size());
  const double avg_run_total_ms =
      kept.empty() ? 0.0
                   : std::accumulate(kept.begin(), kept.end(), 0.0,
                                     [](double sum, const auto *run) {
                                       return sum + run->run_total_ms;
                                     }) /
                         static_cast<double>(kept.size());
  double variance = 0.0;
  for (const auto *run : kept) {
    const double diff = run->avg_window_ms - avg_window_ms;
    variance += diff * diff;
  }
  const double stddev =
      kept.empty() ? 0.0
                   : std::sqrt(variance / static_cast<double>(kept.size()));

  nlohmann::json manifest;
  manifest["schema_version"] = 1;
  manifest["suite"] = {
      {"name", suite.name},
      {"config_path", suite.config_path},
      {"experiment_name", suite.experiment_name},
      {"windows", suite.windows},
      {"config_fingerprint", moonai::fingerprint_config(config)}};

  nlohmann::json run_rows = nlohmann::json::array();
  for (const auto &run : runs) {
    const auto run_dir =
        std::filesystem::relative(run.run_dir, suite_dir).generic_string();
    const auto profile_path =
        std::filesystem::relative(run.profile_path, suite_dir).generic_string();
    run_rows.push_back({{"seed", run.seed},
                        {"run_dir", run_dir},
                        {"profile_path", profile_path},
                        {"avg_window_ms", run.avg_window_ms},
                        {"run_total_ms", run.run_total_ms},
                        {"window_count", run.window_count},
                        {"disposition", run.disposition}});
  }
  manifest["runs"] = std::move(run_rows);

  manifest["aggregate"] = {
      {"avg_window_ms", avg_window_ms},
      {"avg_window_ms_stddev", stddev},
      {"avg_run_total_ms", avg_run_total_ms},
      {"cpu_window_count_avg",
       kept.empty()
           ? 0.0
           : std::accumulate(
                 kept.begin(), kept.end(), 0.0,
                 [](double sum, const auto *run) {
                   return sum +
                          json_number(
                              run->profile["summary"]["cpu_window_count"]);
                 }) /
                 static_cast<double>(kept.size())},
      {"gpu_window_count_avg",
       kept.empty()
           ? 0.0
           : std::accumulate(
                 kept.begin(), kept.end(), 0.0,
                 [](double sum, const auto *run) {
                   return sum +
                          json_number(
                              run->profile["summary"]["gpu_window_count"]);
                 }) /
                 static_cast<double>(kept.size())},
      {"events",
       aggregate_named_stats(kept,
                             [](const auto &profile) -> const nlohmann::json & {
                               return profile["summary"]["events"];
                             })},
      {"counters",
       aggregate_named_stats(kept,
                             [](const auto &profile) -> const nlohmann::json & {
                               return profile["summary"]["counters"];
                             })},
      {"gpu_stage_timings",
       aggregate_named_stats(kept,
                             [](const auto &profile) -> const nlohmann::json & {
                               return profile["summary"]["gpu_stage_timings"];
                             })},
  };

  std::filesystem::create_directories(suite_dir);
  const auto manifest_path = suite_dir / "profile_suite.json";
  std::ofstream file(manifest_path);
  file << manifest.dump(2) << '\n';
}

} // namespace

int main(int argc, const char *argv[]) {
  const Args args = parse_args(argc, argv);
  if (args.suite_name.empty()) {
    std::fprintf(stderr,
                 "Usage: moonai_profiler profiler.lua --suite <name>\n");
    return 1;
  }

  auto suites = moonai::load_profiler_suites_lua(args.profiler_config);
  auto suite_it = suites.find(args.suite_name);
  if (suite_it == suites.end()) {
    spdlog::error("Profiler suite '{}' not found", args.suite_name);
    return 1;
  }

  const auto &suite = suite_it->second;
  auto configs = moonai::load_all_configs_lua(suite.config_path);
  auto config_it = configs.find(suite.experiment_name);
  if (config_it == configs.end()) {
    spdlog::error("Experiment '{}' not found in '{}'", suite.experiment_name,
                  suite.config_path);
    return 1;
  }

  std::vector<SuiteRunSummary> runs;
  runs.reserve(suite.seeds.size());
  for (std::uint64_t seed : suite.seeds) {
    runs.push_back(
        run_suite_member(suite, config_it->second, seed, args.no_gpu));
  }

  write_suite_manifest(suite, config_it->second, std::move(runs));
  return 0;
}
