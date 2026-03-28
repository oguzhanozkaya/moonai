#include "app.hpp"
#include "core/config.hpp"
#include "core/profiler_macros.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc;
#ifdef _WIN32
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &utc);
  return std::string(buf);
}

namespace moonai {
namespace profiler {

struct FrameRecord {
  int index = 0;
  std::unordered_map<std::string, std::int64_t> events_ns;
};

class Profiler {
public:
  static Profiler &instance() {
    static Profiler profiler;
    return profiler;
  }

  void start_run(const AppConfig &cfg);
  void add_duration(const char *event_name, std::int64_t ns);
  nlohmann::json finish_run(std::int64_t run_total_ns);

private:
  Profiler() = default;

  std::optional<AppConfig> cfg_;

  std::vector<FrameRecord> records_;
  std::unordered_map<std::string, std::int64_t> current_durations_;
};

// ScopedTimer implementation (declared in profiler_macros.hpp)
ScopedTimer::ScopedTimer(const char *event_name)
    : event_name_(event_name), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  const auto end = std::chrono::steady_clock::now();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
          .count();
  Profiler::instance().add_duration(event_name_, ns);
}

void Profiler::start_run(const AppConfig &cfg) {
  cfg_ = cfg;
  records_.clear();
  current_durations_.clear();
}

void Profiler::add_duration(const char *event_name, std::int64_t ns) {
  current_durations_[event_name] += ns;

  if (std::strcmp(event_name, "frame_total") == 0) {
    FrameRecord record;
    record.index = static_cast<int>(records_.size());
    record.events_ns = current_durations_;
    records_.push_back(std::move(record));
    current_durations_.clear();
  }
}

nlohmann::json Profiler::finish_run(std::int64_t run_total_ns) {
  nlohmann::json profile;
  profile["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;

  // Per-frame raw data
  nlohmann::json frame_rows = nlohmann::json::array();
  for (const auto &r : records_) {
    nlohmann::json frame_data;
    for (const auto &[name, ns] : r.events_ns) {
      frame_data[name] = static_cast<double>(ns) / 1'000'000.0;
    }
    frame_rows.push_back(std::move(frame_data));
  }
  profile["frames"] = std::move(frame_rows);

  return profile;
}

} // namespace profiler
} // namespace moonai

namespace {

struct Args {
  int frames = 600;
  std::vector<std::uint64_t> seeds = {61, 62, 63, 64, 65, 66};
  std::string output_dir = "output/profiles";
  std::string experiment_name = "profile";
  bool no_gpu = false;
};

Args parse_args(int argc, const char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--frames" && i + 1 < argc) {
      args.frames = std::stoi(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (arg == "--name" && i + 1 < argc) {
      args.experiment_name = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    }
  }
  return args;
}

struct RunResult {
  std::uint64_t seed = 0;
  nlohmann::json profile;
  bool completed = false;
};

RunResult run_profiler(const moonai::AppConfig &cfg) {
  using namespace moonai;

  auto &profiler = profiler::Profiler::instance();

  profiler.start_run(cfg);

  const auto run_start = std::chrono::steady_clock::now();

  App app(cfg);
  bool completed = app.run();

  RunResult result;
  result.seed = cfg.sim_config.seed;

  if (!completed) {
    spdlog::info("Run stopped early, skipping profile generation");
    result.completed = false;
    return result;
  }

  const auto run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - run_start)
                          .count();
  result.profile = profiler.finish_run(run_ns);
  result.completed = true;

  return result;
}

void write_manifest(std::vector<RunResult> runs,
                    const std::filesystem::path &output_path,
                    const moonai::AppConfig &cfg) {
  nlohmann::json manifest;
  manifest["schema_version"] = 1;

  // Metadata - all static info that is same for all runs
  nlohmann::json metadata;
  metadata["suite_name"] = cfg.experiment_name;
  metadata["frame_count"] = cfg.sim_config.max_steps;
  metadata["report_interval_steps"] = cfg.sim_config.report_interval_steps;
  metadata["gpu_allowed"] = cfg.enable_gpu;
  metadata["platform"] = moonai::AppConfig::platform;
  metadata["cuda_compiled"] = moonai::AppConfig::cuda_compiled;
  metadata["openmp_compiled"] = moonai::AppConfig::openmp_compiled;
  metadata["generated_at_utc"] = utc_timestamp();

  manifest["metadata"] = metadata;

  // Runs
  nlohmann::json run_rows = nlohmann::json::array();
  for (const auto &run : runs) {
    if (run.completed) {
      nlohmann::json run_obj;
      run_obj["seed"] = run.seed;
      run_obj["run_total_ms"] = run.profile["run_total_ms"];
      run_obj["frames"] = run.profile["frames"];
      run_rows.push_back(run_obj);
    }
  }
  manifest["runs"] = std::move(run_rows);

  std::filesystem::create_directories(output_path.parent_path());
  std::ofstream file(output_path);
  if (!file.is_open()) {
    spdlog::error("Failed to open profiler output '{}'", output_path.string());
    return;
  }
  file << manifest.dump(2) << '\n';
  if (!file)
    spdlog::error("Failed to write profiler output '{}'", output_path.string());
  else
    spdlog::info("Profiler output written to: {}", output_path.string());
}

} // namespace

int main(int argc, const char *argv[]) {
  const Args args = parse_args(argc, argv);

  moonai::AppConfig base_cfg;
  base_cfg.sim_config = moonai::SimulationConfig();
  base_cfg.sim_config.max_steps = args.frames;
  base_cfg.experiment_name = args.experiment_name;
  base_cfg.headless = false;
  base_cfg.enable_gpu = !args.no_gpu;
  base_cfg.interactive = false;
  base_cfg.speed_multiplier = 1;
  const auto output_path =
      std::filesystem::path(args.output_dir) /
      (utc_timestamp() + "_" + args.experiment_name + ".json");

  std::vector<RunResult> runs;
  runs.reserve(args.seeds.size());
  for (std::uint64_t seed : args.seeds) {
    base_cfg.sim_config.seed = seed;
    runs.push_back(run_profiler(base_cfg));
  }

  write_manifest(std::move(runs), output_path, base_cfg);
  return 0;
}
