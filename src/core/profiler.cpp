#include "core/profiler.hpp"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

template <typename Enum> constexpr std::size_t enum_index(Enum value) {
  return static_cast<std::size_t>(value);
}

std::string sanitize_path_component(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char ch : value) {
    const bool valid = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
    sanitized.push_back(valid ? ch : '_');
  }
  return sanitized.empty() ? std::string{"profile"} : sanitized;
}

std::string utc_timestamp_for_path(std::chrono::system_clock::time_point now) {
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &time);
#else
  gmtime_r(&time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

std::string utc_timestamp_iso(std::chrono::system_clock::time_point now) {
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &time);
#else
  gmtime_r(&time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

const char *profile_event_measurement(ProfileEvent event) {
  switch (event) {
    case ProfileEvent::WindowTotal:
      return "window_wall_clock";
    case ProfileEvent::PrepareGpuWindow:
    case ProfileEvent::GpuSensorFlatten:
    case ProfileEvent::GpuPackInputs:
    case ProfileEvent::GpuLaunch:
    case ProfileEvent::GpuStartUnpack:
    case ProfileEvent::GpuFinishUnpack:
    case ProfileEvent::GpuOutputConvert:
    case ProfileEvent::CpuEvalTotal:
    case ProfileEvent::CpuSensorBuild:
    case ProfileEvent::CpuNnActivate:
    case ProfileEvent::ApplyActions:
    case ProfileEvent::SimulationStep:
    case ProfileEvent::RebuildSpatialGrid:
    case ProfileEvent::RebuildFoodGrid:
    case ProfileEvent::AgentUpdate:
    case ProfileEvent::ProcessEnergy:
    case ProfileEvent::ProcessFood:
    case ProfileEvent::ProcessAttacks:
    case ProfileEvent::BoundaryApply:
    case ProfileEvent::DeathCheck:
    case ProfileEvent::CountAlive:
    case ProfileEvent::ComputeFitness:
    case ProfileEvent::Speciate:
    case ProfileEvent::Reproduce:
    case ProfileEvent::Logging:
      return "accumulated";
    case ProfileEvent::Count:
      return "unknown";
  }
  return "unknown";
}

const char *profile_event_description(ProfileEvent event) {
  switch (event) {
    case ProfileEvent::WindowTotal:
      return "Wall-clock report-window duration measured in headless mode.";
    case ProfileEvent::PrepareGpuWindow:
      return "Time spent uploading GPU network state before a report window.";
    case ProfileEvent::GpuSensorFlatten:
      return "Time spent preparing CPU-side sensor input buffers for the "
             "legacy "
             "GPU inference path.";
    case ProfileEvent::GpuPackInputs:
      return "Time spent packing flattened inputs into GPU buffers.";
    case ProfileEvent::GpuLaunch:
      return "Time spent running GPU neural inference for the legacy hybrid "
             "path.";
    case ProfileEvent::GpuStartUnpack:
      return "Time spent starting GPU output unpacking.";
    case ProfileEvent::GpuFinishUnpack:
      return "Time spent waiting for GPU output unpacking to finish.";
    case ProfileEvent::GpuOutputConvert:
      return "Time spent converting GPU outputs into agent actions.";
    case ProfileEvent::CpuEvalTotal:
      return "Total CPU inference path time across a report window.";
    case ProfileEvent::CpuSensorBuild:
      return "Time spent building CPU-side sensor inputs.";
    case ProfileEvent::CpuNnActivate:
      return "Time spent running CPU neural network activations.";
    case ProfileEvent::ApplyActions:
      return "Time spent applying movement actions to agents.";
    case ProfileEvent::SimulationStep:
      return "Accumulated time spent inside simulation steps.";
    case ProfileEvent::RebuildSpatialGrid:
      return "Time spent rebuilding the agent spatial grid.";
    case ProfileEvent::RebuildFoodGrid:
      return "Time spent rebuilding the food spatial grid.";
    case ProfileEvent::AgentUpdate:
      return "Time spent updating agent age and per-step state.";
    case ProfileEvent::ProcessEnergy:
      return "Time spent applying per-step energy drain.";
    case ProfileEvent::ProcessFood:
      return "Time spent handling prey food pickup.";
    case ProfileEvent::ProcessAttacks:
      return "Time spent handling predator attack checks.";
    case ProfileEvent::BoundaryApply:
      return "Time spent applying world boundary rules.";
    case ProfileEvent::DeathCheck:
      return "Time spent marking dead agents after step processing.";
    case ProfileEvent::CountAlive:
      return "Time spent recounting living predators and prey.";
    case ProfileEvent::ComputeFitness:
      return "Time spent computing genome fitness values.";
    case ProfileEvent::Speciate:
      return "Time spent assigning genomes to species.";
    case ProfileEvent::Reproduce:
      return "Time spent producing offspring during the report window.";
    case ProfileEvent::Logging:
      return "Time spent writing report-window logs.";
    case ProfileEvent::Count:
      return "";
  }
  return "";
}

const char *profile_counter_description(ProfileCounter counter) {
  switch (counter) {
    case ProfileCounter::StepsExecuted:
      return "Number of simulation steps completed.";
    case ProfileCounter::GridQueryCalls:
      return "Number of spatial grid radius queries executed.";
    case ProfileCounter::GridCandidatesScanned:
      return "Number of candidate entries scanned during radius queries.";
    case ProfileCounter::FoodEatAttempts:
      return "Number of prey food pickup attempts.";
    case ProfileCounter::FoodEaten:
      return "Number of successful food pickups.";
    case ProfileCounter::AttackChecks:
      return "Number of potential attack targets considered.";
    case ProfileCounter::Kills:
      return "Number of prey killed by predators.";
    case ProfileCounter::CompatibilityChecks:
      return "Number of genome compatibility comparisons.";
    case ProfileCounter::Count:
      return "";
  }
  return "";
}

// DEPRECATED: GpuStageTiming is no longer used. Function kept for ABI
// compatibility.
const char *gpu_stage_timing_description(GpuStageTiming stage) {
  (void)stage;
  return "";
}

} // namespace

Profiler &Profiler::instance() {
  static Profiler profiler;
  return profiler;
}

void Profiler::set_enabled(bool enabled) {
  enabled_.store(enabled, std::memory_order_relaxed);
}

void Profiler::start_run(const ProfileRunSpec &spec) {
  if (!enabled()) {
    return;
  }

  const auto now = std::chrono::system_clock::now();
  experiment_name_ = spec.experiment_name;
  generated_at_utc_ = utc_timestamp_iso(now);
  seed_ = spec.seed;
  predator_count_ = spec.predator_count;
  prey_count_ = spec.prey_count;
  food_count_ = spec.food_count;
  total_steps_ = spec.total_steps;
  report_interval_steps_ = spec.report_interval_steps;
  gpu_allowed_ = spec.gpu_allowed;
  cuda_compiled_ = spec.cuda_compiled;
  openmp_compiled_ = spec.openmp_compiled;
  suite_name_ = spec.suite_name;
  base_experiment_name_ = spec.base_experiment_name;
  config_fingerprint_ = spec.config_fingerprint;
  profiler_entry_point_ = spec.profiler_entry_point;
  window_cpu_used_ = false;
  window_gpu_used_ = false;
  window_active_ = false;
  window_records_.clear();

  try {
    const std::filesystem::path base_path(spec.output_root_dir);
    std::string run_name;
    if (!experiment_name_.empty()) {
      // Named experiment: use timestamp + sanitized name (name already contains
      // seed from config.lua)
      run_name = utc_timestamp_for_path(now) + "_" +
                 sanitize_path_component(experiment_name_);
    } else {
      // Anonymous run: add seed suffix
      run_name = utc_timestamp_for_path(now) + "_seed" + std::to_string(seed_);
    }
    std::filesystem::path candidate = base_path / run_name;
    for (int suffix = 2; std::filesystem::exists(candidate); ++suffix) {
      candidate = base_path / (run_name + "_" + std::to_string(suffix));
    }
    std::filesystem::create_directories(candidate);
    output_dir_ = candidate.string();
  } catch (const std::exception &e) {
    output_dir_.clear();
    set_enabled(false);
    spdlog::error("Failed to initialize profiler output under '{}': {}",
                  spec.output_root_dir, e.what());
    return;
  }

  for (auto &duration : current_durations_ns_) {
    duration.store(0, std::memory_order_relaxed);
  }
  for (auto &counter : current_counters_) {
    counter.store(0, std::memory_order_relaxed);
  }
  for (auto &stage : current_gpu_stage_durations_ns_) {
    stage.store(0, std::memory_order_relaxed);
  }
}

void Profiler::start_window(int window_index) {
  if (!enabled()) {
    return;
  }
  (void)window_index;
  window_cpu_used_ = false;
  window_gpu_used_ = false;
  window_active_ = true;
  window_start_ = std::chrono::steady_clock::now();
  for (auto &duration : current_durations_ns_) {
    duration.store(0, std::memory_order_relaxed);
  }
  for (auto &counter : current_counters_) {
    counter.store(0, std::memory_order_relaxed);
  }
  for (auto &stage : current_gpu_stage_durations_ns_) {
    stage.store(0, std::memory_order_relaxed);
  }
}

void Profiler::mark_cpu_used(bool used) {
  if (!enabled() || !used) {
    return;
  }
  window_cpu_used_ = true;
}

void Profiler::mark_gpu_used(bool used) {
  if (!enabled() || !used) {
    return;
  }
  window_gpu_used_ = true;
}

void Profiler::add_duration(ProfileEvent event, std::int64_t nanoseconds) {
  if (!enabled()) {
    return;
  }
  current_durations_ns_[enum_index(event)].fetch_add(nanoseconds,
                                                     std::memory_order_relaxed);
}

void Profiler::set_duration(ProfileEvent event, std::int64_t nanoseconds) {
  if (!enabled()) {
    return;
  }
  current_durations_ns_[enum_index(event)].store(nanoseconds,
                                                 std::memory_order_relaxed);
}

void Profiler::increment(ProfileCounter counter, std::int64_t value) {
  if (!enabled()) {
    return;
  }
  current_counters_[enum_index(counter)].fetch_add(value,
                                                   std::memory_order_relaxed);
}

void Profiler::add_gpu_stage_duration(GpuStageTiming stage,
                                      std::int64_t nanoseconds) {
  if (!enabled()) {
    return;
  }
  current_gpu_stage_durations_ns_[enum_index(stage)].fetch_add(
      nanoseconds, std::memory_order_relaxed);
}

void Profiler::finish_window(const ReportWindowProfileMeta &meta) {
  if (!enabled()) {
    return;
  }

  WindowRecord record;
  record.meta = meta;
  record.cpu_used = window_cpu_used_;
  record.gpu_used = window_gpu_used_;

  if (window_active_) {
    const auto window_end = std::chrono::steady_clock::now();
    current_durations_ns_[enum_index(ProfileEvent::WindowTotal)].store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(window_end -
                                                             window_start_)
            .count(),
        std::memory_order_relaxed);
    window_active_ = false;
  }

  for (std::size_t i = 0; i < record.durations_ns.size(); ++i) {
    record.durations_ns[i] =
        current_durations_ns_[i].load(std::memory_order_relaxed);
  }
  for (std::size_t i = 0; i < record.counters.size(); ++i) {
    record.counters[i] = current_counters_[i].load(std::memory_order_relaxed);
  }
  for (std::size_t i = 0; i < record.gpu_stage_durations_ns.size(); ++i) {
    record.gpu_stage_durations_ns[i] =
        current_gpu_stage_durations_ns_[i].load(std::memory_order_relaxed);
  }

  window_records_.push_back(std::move(record));
}

void Profiler::finish_run(std::int64_t run_total_ns) {
  if (!enabled() || output_dir_.empty()) {
    return;
  }

  try {
    std::filesystem::create_directories(output_dir_);
  } catch (const std::exception &e) {
    set_enabled(false);
    spdlog::error("Failed to create profiler output directory '{}': {}",
                  output_dir_, e.what());
    return;
  }

  nlohmann::json profile;
  profile["schema_version"] = 4;
  profile["generated_at_utc"] = generated_at_utc_;
  profile["run"] = {{"experiment_name", experiment_name_},
                    {"base_experiment_name", base_experiment_name_},
                    {"suite_name", suite_name_},
                    {"output_dir", output_dir_},
                    {"seed", seed_},
                    {"config_fingerprint", config_fingerprint_},
                    {"profiler_entry_point", profiler_entry_point_},
                    {"predator_count", predator_count_},
                    {"prey_count", prey_count_},
                    {"food_count", food_count_},
                    {"total_steps", total_steps_},
                    {"report_interval_steps", report_interval_steps_},
                    {"gpu_allowed", gpu_allowed_},
                    {"headless_only", true},
#ifdef _WIN32
                    {"platform", "windows"},
#else
                    {"platform", "linux"},
#endif
                    {"cuda_compiled", cuda_compiled_},
                    {"openmp_compiled", openmp_compiled_}};
  profile["notes"] = {
      "Profiling is only supported in headless mode so window_total is not "
      "polluted by GUI rendering or pause time.",
      "Event durations are stored in milliseconds in both per-window and "
      "summary sections.",
      "Events with measurement='accumulated' sum repeated scoped timings "
      "within a report window.",
      "Path-specific events can remain zero for windows that never execute "
      "that path.",
      "cpu_window_count and gpu_window_count are non-exclusive; a fallback "
      "window can increment both counts.",
      "Fields ending with nonzero_window_count count windows where the "
      "recorded value was greater than zero.",
      "Use nonzero_window_count and avg_ms_per_nonzero_window when "
      "judging optional events.",
      "gpu_stage_timings use device elapsed timestamps collected inside the "
      "resident GPU step sequence.",
      "gpu_stage_timings decompose gpu_resident_step and are diagnostic "
      "timings, not top-level report-window wall-clock events."};

  nlohmann::json event_defs = nlohmann::json::array();
  for (std::size_t i = 0; i < enum_index(ProfileEvent::Count); ++i) {
    const auto event = static_cast<ProfileEvent>(i);
    event_defs.push_back({{"name", profile_event_name(event)},
                          {"unit", "ms"},
                          {"measurement", profile_event_measurement(event)},
                          {"description", profile_event_description(event)}});
  }
  profile["event_definitions"] = std::move(event_defs);

  nlohmann::json counter_defs = nlohmann::json::array();
  for (std::size_t i = 0; i < enum_index(ProfileCounter::Count); ++i) {
    const auto counter = static_cast<ProfileCounter>(i);
    counter_defs.push_back(
        {{"name", profile_counter_name(counter)},
         {"description", profile_counter_description(counter)}});
  }
  profile["counter_definitions"] = std::move(counter_defs);

  nlohmann::json gpu_stage_defs = nlohmann::json::array();
  for (std::size_t i = 0; i < enum_index(GpuStageTiming::Count); ++i) {
    const auto stage = static_cast<GpuStageTiming>(i);
    gpu_stage_defs.push_back(
        {{"name", gpu_stage_timing_name(stage)},
         {"unit", "ms"},
         {"measurement", "device_elapsed"},
         {"scope", "resident_step"},
         {"parent_event", "gpu_resident_step"},
         {"collection_mode", "device_globaltimer"},
         {"description", gpu_stage_timing_description(stage)}});
  }
  profile["gpu_stage_definitions"] = std::move(gpu_stage_defs);

  std::array<std::int64_t, enum_index(ProfileEvent::Count)> total_durations{};
  std::array<std::int64_t, enum_index(ProfileCounter::Count)> total_counters{};
  std::array<std::int64_t, enum_index(GpuStageTiming::Count)>
      total_gpu_stage_durations{};
  std::array<int, enum_index(ProfileEvent::Count)> active_duration_windows{};
  std::array<int, enum_index(ProfileCounter::Count)> active_counter_windows{};
  std::array<int, enum_index(GpuStageTiming::Count)> active_gpu_stage_windows{};
  int cpu_windows = 0;
  int gpu_windows = 0;
  int min_predators = std::numeric_limits<int>::max();
  int max_predators = 0;
  int min_prey = std::numeric_limits<int>::max();
  int max_prey = 0;
  nlohmann::json window_rows = nlohmann::json::array();

  for (const auto &record : window_records_) {
    if (record.cpu_used) {
      ++cpu_windows;
    }
    if (record.gpu_used) {
      ++gpu_windows;
    }
    min_predators = std::min(min_predators, record.meta.predator_count);
    max_predators = std::max(max_predators, record.meta.predator_count);
    min_prey = std::min(min_prey, record.meta.prey_count);
    max_prey = std::max(max_prey, record.meta.prey_count);
    nlohmann::json duration_row;
    nlohmann::json counter_row;
    for (std::size_t i = 0; i < total_durations.size(); ++i) {
      total_durations[i] += record.durations_ns[i];
      if (record.durations_ns[i] > 0) {
        ++active_duration_windows[i];
      }
      duration_row[profile_event_name(static_cast<ProfileEvent>(i))] =
          static_cast<double>(record.durations_ns[i]) / 1'000'000.0;
    }
    for (std::size_t i = 0; i < total_counters.size(); ++i) {
      total_counters[i] += record.counters[i];
      if (record.counters[i] > 0) {
        ++active_counter_windows[i];
      }
      counter_row[profile_counter_name(static_cast<ProfileCounter>(i))] =
          record.counters[i];
    }
    nlohmann::json gpu_stage_row;
    for (std::size_t i = 0; i < total_gpu_stage_durations.size(); ++i) {
      total_gpu_stage_durations[i] += record.gpu_stage_durations_ns[i];
      if (record.gpu_stage_durations_ns[i] > 0) {
        ++active_gpu_stage_windows[i];
      }
      gpu_stage_row[gpu_stage_timing_name(static_cast<GpuStageTiming>(i))] =
          static_cast<double>(record.gpu_stage_durations_ns[i]) / 1'000'000.0;
    }
    window_rows.push_back({{"window_index", record.meta.window_index},
                           {"cpu_used", record.cpu_used},
                           {"gpu_used", record.gpu_used},
                           {"predator_count", record.meta.predator_count},
                           {"prey_count", record.meta.prey_count},
                           {"species_count", record.meta.species_count},
                           {"best_fitness", record.meta.best_fitness},
                           {"avg_fitness", record.meta.avg_fitness},
                           {"avg_complexity", record.meta.avg_complexity},
                           {"events_ms", std::move(duration_row)},
                           {"counters", std::move(counter_row)},
                           {"gpu_stage_timings_ms", std::move(gpu_stage_row)}});
  }
  profile["windows"] = std::move(window_rows);

  nlohmann::json summary;
  summary["window_count"] = window_records_.size();
  summary["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;
  summary["cpu_window_count"] = cpu_windows;
  summary["gpu_window_count"] = gpu_windows;
  summary["population_ranges"] = {
      {"predator_min", window_records_.empty() ? 0 : min_predators},
      {"predator_max", window_records_.empty() ? 0 : max_predators},
      {"prey_min", window_records_.empty() ? 0 : min_prey},
      {"prey_max", window_records_.empty() ? 0 : max_prey}};
  summary["path_count_note"] =
      "cpu_window_count and gpu_window_count are non-exclusive; fallback "
      "windows can count toward both.";

  nlohmann::json durations_json;
  for (std::size_t i = 0; i < total_durations.size(); ++i) {
    const double total_ms =
        static_cast<double>(total_durations[i]) / 1'000'000.0;
    const int active_count = active_duration_windows[i];
    durations_json[profile_event_name(static_cast<ProfileEvent>(i))] = {
        {"total_ms", total_ms},
        {"avg_ms_per_window",
         window_records_.empty()
             ? 0.0
             : total_ms / static_cast<double>(window_records_.size())},
        {"nonzero_window_count", active_count},
        {"avg_ms_per_nonzero_window",
         active_count == 0 ? 0.0
                           : total_ms / static_cast<double>(active_count)}};
  }
  summary["events"] = durations_json;

  nlohmann::json counters_json;
  for (std::size_t i = 0; i < total_counters.size(); ++i) {
    const int active_count = active_counter_windows[i];
    counters_json[profile_counter_name(static_cast<ProfileCounter>(i))] = {
        {"total", total_counters[i]},
        {"avg_per_window",
         window_records_.empty()
             ? 0.0
             : static_cast<double>(total_counters[i]) /
                   static_cast<double>(window_records_.size())},
        {"nonzero_window_count", active_count},
        {"avg_per_nonzero_window",
         active_count == 0 ? 0.0
                           : static_cast<double>(total_counters[i]) /
                                 static_cast<double>(active_count)}};
  }
  summary["counters"] = counters_json;

  nlohmann::json gpu_stage_json;
  for (std::size_t i = 0; i < total_gpu_stage_durations.size(); ++i) {
    const double total_ms =
        static_cast<double>(total_gpu_stage_durations[i]) / 1'000'000.0;
    const int active_count = active_gpu_stage_windows[i];
    gpu_stage_json[gpu_stage_timing_name(static_cast<GpuStageTiming>(i))] = {
        {"total_ms", total_ms},
        {"avg_ms_per_window",
         window_records_.empty()
             ? 0.0
             : total_ms / static_cast<double>(window_records_.size())},
        {"nonzero_window_count", active_count},
        {"avg_ms_per_nonzero_window",
         active_count == 0 ? 0.0
                           : total_ms / static_cast<double>(active_count)}};
  }
  summary["gpu_stage_timings"] = gpu_stage_json;
  profile["summary"] = std::move(summary);

  const auto json_path = std::filesystem::path(output_dir_) / "profile.json";
  std::ofstream json(json_path);
  if (!json.is_open()) {
    set_enabled(false);
    spdlog::error("Failed to open profiler output file '{}' for writing",
                  json_path.string());
    return;
  }
  json << profile.dump(2) << "\n";
  json.flush();
  if (!json) {
    set_enabled(false);
    spdlog::error("Failed to write profiler output file '{}'",
                  json_path.string());
  }
}

const char *profile_event_name(ProfileEvent event) {
  switch (event) {
    case ProfileEvent::WindowTotal:
      return "window_total";
    case ProfileEvent::PrepareGpuWindow:
      return "prepare_gpu_window";
    case ProfileEvent::GpuSensorFlatten:
      return "gpu_sensor_flatten";
    case ProfileEvent::GpuPackInputs:
      return "gpu_pack_inputs";
    case ProfileEvent::GpuLaunch:
      return "gpu_launch";
    case ProfileEvent::GpuStartUnpack:
      return "gpu_start_unpack";
    case ProfileEvent::GpuFinishUnpack:
      return "gpu_finish_unpack";
    case ProfileEvent::GpuOutputConvert:
      return "gpu_output_convert";
    case ProfileEvent::CpuEvalTotal:
      return "cpu_eval_total";
    case ProfileEvent::CpuSensorBuild:
      return "cpu_sensor_build";
    case ProfileEvent::CpuNnActivate:
      return "cpu_nn_activate";
    case ProfileEvent::ApplyActions:
      return "apply_actions";
    case ProfileEvent::SimulationStep:
      return "simulation_step";
    case ProfileEvent::RebuildSpatialGrid:
      return "rebuild_spatial_grid";
    case ProfileEvent::RebuildFoodGrid:
      return "rebuild_food_grid";
    case ProfileEvent::AgentUpdate:
      return "agent_update";
    case ProfileEvent::ProcessEnergy:
      return "process_energy";
    case ProfileEvent::ProcessFood:
      return "process_food";
    case ProfileEvent::ProcessAttacks:
      return "process_attacks";
    case ProfileEvent::BoundaryApply:
      return "boundary_apply";
    case ProfileEvent::DeathCheck:
      return "death_check";
    case ProfileEvent::CountAlive:
      return "count_alive";
    case ProfileEvent::ComputeFitness:
      return "compute_fitness";
    case ProfileEvent::Speciate:
      return "speciate";
    case ProfileEvent::Reproduce:
      return "reproduce";
    case ProfileEvent::Logging:
      return "logging";
    case ProfileEvent::Count:
      return "count";
  }
  return "unknown";
}

const char *profile_counter_name(ProfileCounter counter) {
  switch (counter) {
    case ProfileCounter::StepsExecuted:
      return "steps_executed";
    case ProfileCounter::GridQueryCalls:
      return "grid_query_calls";
    case ProfileCounter::GridCandidatesScanned:
      return "grid_candidates_scanned";
    case ProfileCounter::FoodEatAttempts:
      return "food_eat_attempts";
    case ProfileCounter::FoodEaten:
      return "food_eaten";
    case ProfileCounter::AttackChecks:
      return "attack_checks";
    case ProfileCounter::Kills:
      return "kills";
    case ProfileCounter::CompatibilityChecks:
      return "compatibility_checks";
    case ProfileCounter::Count:
      return "count";
  }
  return "unknown";
}

const char *gpu_stage_timing_name(GpuStageTiming stage) {
  switch (stage) {
    case GpuStageTiming::ResidentStepBinRebuildPre:
      return "bin_rebuild_pre";
    case GpuStageTiming::ResidentStepSensorBuild:
      return "sensor_build";
    case GpuStageTiming::ResidentStepInference:
      return "inference";
    case GpuStageTiming::ResidentStepMovement:
      return "movement";
    case GpuStageTiming::ResidentStepBinRebuildPost:
      return "bin_rebuild_post";
    case GpuStageTiming::ResidentStepPreyFood:
      return "prey_food";
    case GpuStageTiming::ResidentStepPredatorAttack:
      return "predator_attack";
    case GpuStageTiming::ResidentStepRespawn:
      return "respawn";
    case GpuStageTiming::Count:
      return "count";
  }
  return "unknown";
}

} // namespace moonai
