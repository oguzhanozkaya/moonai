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
};

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
};

} // namespace moonai
