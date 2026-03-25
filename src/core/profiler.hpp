#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace moonai {

enum class ProfileEvent : std::size_t {
  WindowTotal = 0,
  PrepareGpuWindow,
  GpuSensorFlatten,
  GpuPackInputs,
  GpuLaunch,
  GpuStartUnpack,
  GpuFinishUnpack,
  GpuOutputConvert,
  CpuEvalTotal,
  CpuSensorBuild,
  CpuNnActivate,
  ApplyActions,
  SimulationStep,
  RebuildSpatialGrid,
  RebuildFoodGrid,
  AgentUpdate,
  ProcessEnergy,
  ProcessFood,
  ProcessAttacks,
  BoundaryApply,
  DeathCheck,
  CountAlive,
  ComputeFitness,
  Speciate,
  Reproduce,
  Logging,
  Count
};

enum class ProfileCounter : std::size_t {
  StepsExecuted = 0,
  GridQueryCalls,
  GridCandidatesScanned,
  FoodEatAttempts,
  FoodEaten,
  AttackChecks,
  Kills,
  CompatibilityChecks,
  Count
};

// NOTE: GpuStageTiming enum and related infrastructure is deprecated.
// The resident GPU path is no longer used in the continuous ecology model.
// These types are kept for ABI compatibility but will be removed in a future
// cleanup.
enum class GpuStageTiming : std::size_t {
  ResidentStepBinRebuildPre = 0,
  ResidentStepSensorBuild,
  ResidentStepInference,
  ResidentStepMovement,
  ResidentStepBinRebuildPost,
  ResidentStepPreyFood,
  ResidentStepPredatorAttack,
  ResidentStepRespawn,
  Count
};

struct ReportWindowProfileMeta {
  int window_index = 0;
  int predator_count = 0;
  int prey_count = 0;
  int species_count = 0;
  float best_fitness = 0.0f;
  float avg_fitness = 0.0f;
  float avg_complexity = 0.0f;
};

struct ProfileRunSpec {
  std::string experiment_name;
  std::string output_root_dir;
  std::uint64_t seed = 0;
  int predator_count = 0;
  int prey_count = 0;
  int food_count = 0;
  int total_steps = 0;
  int report_interval_steps = 0;
  bool gpu_allowed = false;
  bool cuda_compiled = false;
  bool openmp_compiled = false;
  std::string suite_name;
  std::string base_experiment_name;
  std::string config_fingerprint;
  std::string profiler_entry_point;
};

class Profiler {
public:
  static Profiler &instance();

  void set_enabled(bool enabled);
  bool enabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  void start_run(const ProfileRunSpec &spec);
  void start_window(int window_index);
  void mark_cpu_used(bool used);
  void mark_gpu_used(bool used);
  void add_duration(ProfileEvent event, std::int64_t nanoseconds);
  void set_duration(ProfileEvent event, std::int64_t nanoseconds);
  void increment(ProfileCounter counter, std::int64_t value = 1);
  // DEPRECATED: Kept for ABI compatibility. GpuStageTiming is no longer used.
  void add_gpu_stage_duration(GpuStageTiming stage, std::int64_t nanoseconds);
  void finish_window(const ReportWindowProfileMeta &meta);
  void finish_run(std::int64_t run_total_ns);

  const std::string &output_dir() const {
    return output_dir_;
  }

private:
  struct WindowRecord {
    ReportWindowProfileMeta meta;
    bool cpu_used = false;
    bool gpu_used = false;
    std::array<std::int64_t, static_cast<std::size_t>(ProfileEvent::Count)>
        durations_ns{};
    std::array<std::int64_t, static_cast<std::size_t>(ProfileCounter::Count)>
        counters{};
    // DEPRECATED: Kept for ABI compatibility. GpuStageTiming is no longer used.
    std::array<std::int64_t, static_cast<std::size_t>(GpuStageTiming::Count)>
        gpu_stage_durations_ns{};
  };

  Profiler() = default;

  std::atomic<bool> enabled_{false};
  std::string experiment_name_;
  std::string output_dir_;
  std::string generated_at_utc_;
  std::uint64_t seed_ = 0;
  int predator_count_ = 0;
  int prey_count_ = 0;
  int food_count_ = 0;
  int total_steps_ = 0;
  int report_interval_steps_ = 0;
  bool gpu_allowed_ = false;
  bool cuda_compiled_ = false;
  bool openmp_compiled_ = false;
  std::string suite_name_;
  std::string base_experiment_name_;
  std::string config_fingerprint_;
  std::string profiler_entry_point_;
  bool window_cpu_used_ = false;
  bool window_gpu_used_ = false;
  bool window_active_ = false;
  std::chrono::steady_clock::time_point window_start_{};
  std::vector<WindowRecord> window_records_;
  std::array<std::atomic<std::int64_t>,
             static_cast<std::size_t>(ProfileEvent::Count)>
      current_durations_ns_{};
  std::array<std::atomic<std::int64_t>,
             static_cast<std::size_t>(ProfileCounter::Count)>
      current_counters_{};
  // DEPRECATED: Kept for ABI compatibility. GpuStageTiming is no longer used.
  std::array<std::atomic<std::int64_t>,
             static_cast<std::size_t>(GpuStageTiming::Count)>
      current_gpu_stage_durations_ns_{};
};

class ScopedTimer {
public:
  explicit ScopedTimer(ProfileEvent event)
      : event_(event), active_(Profiler::instance().enabled()) {
    if (active_) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedTimer() {
    if (!active_) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
            .count();
    Profiler::instance().add_duration(event_, ns);
  }

private:
  ProfileEvent event_;
  bool active_ = false;
  std::chrono::steady_clock::time_point start_{};
};

const char *profile_event_name(ProfileEvent event);
const char *profile_counter_name(ProfileCounter counter);
// DEPRECATED: Kept for ABI compatibility.
const char *gpu_stage_timing_name(GpuStageTiming stage);

#define MOONAI_PROFILE_CONCAT_INNER(lhs, rhs) lhs##rhs
#define MOONAI_PROFILE_CONCAT(lhs, rhs) MOONAI_PROFILE_CONCAT_INNER(lhs, rhs)

#ifdef MOONAI_BUILD_PROFILER
#define MOONAI_PROFILE_SCOPE(event)                                            \
  ::moonai::ScopedTimer MOONAI_PROFILE_CONCAT(moonai_scoped_timer_,            \
                                              __COUNTER__)(event)
#define MOONAI_PROFILE_INC(counter, ...)                                       \
  ::moonai::Profiler::instance().increment(counter, ##__VA_ARGS__)
#define MOONAI_PROFILE_MARK_CPU_USED(value)                                    \
  ::moonai::Profiler::instance().mark_cpu_used(value)
#define MOONAI_PROFILE_MARK_GPU_USED(value)                                    \
  ::moonai::Profiler::instance().mark_gpu_used(value)
#define MOONAI_PROFILE_ADD_DURATION(event, value)                              \
  ::moonai::Profiler::instance().add_duration(event, value)
#define MOONAI_PROFILE_SET_DURATION(event, value)                              \
  ::moonai::Profiler::instance().set_duration(event, value)
#else
#define MOONAI_PROFILE_SCOPE(event) ((void)0)
#define MOONAI_PROFILE_INC(counter, ...) ((void)0)
#define MOONAI_PROFILE_MARK_CPU_USED(value) ((void)0)
#define MOONAI_PROFILE_MARK_GPU_USED(value) ((void)0)
#define MOONAI_PROFILE_ADD_DURATION(event, value) ((void)0)
#define MOONAI_PROFILE_SET_DURATION(event, value) ((void)0)
#endif

} // namespace moonai
