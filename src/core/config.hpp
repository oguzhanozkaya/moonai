#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace moonai {

struct SimulationConfig {
  int grid_size = 6000;

  int predator_count = 1200;
  int prey_count = 28800;
  int food_count = 60000;

  float predator_speed = 1.0f;
  float prey_speed = 1.0f;
  float vision_range = 36.0f;
  float interaction_range = 1.0f;
  float mate_range = 12.0f;
  float birth_spawn_radius = 12.0f;

  float food_respawn_rate = 0.00030f;
  float energy_drain_per_step = 0.00015f;
  float energy_gain_from_kill = 0.30f;
  float energy_gain_from_food = 0.25f;
  float initial_energy = 0.5f;
  float reproduction_energy_threshold = 1.0f;
  float reproduction_energy_cost = 0.25f;
  float offspring_initial_energy = 1.0f;

  float mutation_rate = 0.3f;
  float crossover_rate = 0.75f;
  float weight_mutation_power = 0.3f;
  float add_node_rate = 0.06f;
  float add_connection_rate = 0.12f;
  float delete_connection_rate = 0.01f;
  int max_hidden_nodes = 1200;
  int max_steps = 0;

  float compatibility_threshold = 9.0f;
  float compatibility_min_normalization = 36.0f;
  float c1_excess = 1.0f;
  float c2_disjoint = 1.0f;
  float c3_weight = 0.4f;
  int species_update_interval_steps = 1000;

  int seed = 67;

  std::string output_dir = "output";
  int report_interval_steps = 1000;
};

struct ConfigError {
  std::string field;
  std::string message;
};

struct AppConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  bool headless = false;
  int speed_multiplier = 1;
  std::optional<std::string> run_name_override;

  static constexpr const char *platform =
#ifdef _WIN32
      "windows";
#else
      "linux";
#endif
};

nlohmann::json config_to_json(const SimulationConfig &config);
void save_config(const SimulationConfig &config, const std::string &filepath);

std::vector<ConfigError> validate_config(const SimulationConfig &config);

} // namespace moonai
