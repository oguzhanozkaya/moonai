#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <cstring>
#include <functional>
#include <memory>
#include <optional>

namespace moonai {

struct SessionConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  std::string output_dir;
  std::uint64_t seed = 0;
  bool headless = false;
  bool enable_gpu = true;
  bool enable_logger = true;
  std::optional<std::string> run_name_override;
  int max_steps_override = 0;

  // Event loop configuration
  bool enable_interactions = true; // Allow pause, step, selection
  bool auto_run = false;           // Ignore pause, run continuously
  int speed_multiplier = 1;        // Steps per frame (1 = normal)
};

enum class StopReason { Completed, UserQuit, Signal };

class Session {
public:
  explicit Session(const SessionConfig &cfg);
  ~Session();

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  // Component access
  Registry &registry();
  SimulationManager &simulation();
  EvolutionManager &evolution();
  MetricsCollector &metrics();
  Logger *logger();
  VisualizationManager *visualization();

  const Registry &registry() const;
  const SimulationManager &simulation() const;
  const EvolutionManager &evolution() const;
  const MetricsCollector &metrics() const;

  // Core operations
  void step(float dt);
  StepMetrics record_and_log(int births_in_window, int deaths_in_window);

  // Event loop delegation
  // Runs the full simulation loop with visualization
  // should_stop: callback to check for external stop signal (e.g., Ctrl+C)
  // Returns reason why loop stopped
  StopReason
  run_event_loop(std::function<bool()> should_stop = nullptr,
                 std::function<void(const StepMetrics &)> on_report = nullptr);

  // State
  int steps_executed() const;
  int births_in_window() const;
  int deaths_in_window() const;

  // Increment counters (called by caller when births/deaths occur)
  void record_birth();
  void record_death();
  void reset_window_counters();

private:
  SessionConfig cfg_;
  Random rng_;
  Registry registry_;
  SimulationManager simulation_;
  EvolutionManager evolution_;
  MetricsCollector metrics_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<VisualizationManager> visualization_;

  int steps_executed_ = 0;
  int births_in_window_ = 0;
  int deaths_in_window_ = 0;
  std::vector<Vec2> actions_buffer_;

  // Internal helpers
  void update_selected_visualization();
  bool should_continue(std::function<bool()> should_stop) const;
};

} // namespace moonai
