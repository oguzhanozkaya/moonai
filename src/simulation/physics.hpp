#pragma once

#include <cstddef>
#include <vector>

namespace moonai {

// Sensor input vector layout for the neural network.
// All values normalized to roughly [0, 1] or [-1, 1].
// Note: This struct is used by ECS systems, not the legacy Agent-based Physics.
struct SensorInput {
  // Nearest predator (relative)
  float nearest_predator_dist =
      -1.0f; // -1=none in range, 0=touching, 1=at vision limit
  float nearest_predator_angle = 0.0f; // [-1, 1] mapped from [-pi, pi]

  // Nearest prey (relative)
  float nearest_prey_dist = -1.0f; // -1=none in range
  float nearest_prey_angle = 0.0f;

  // Nearest food (relative, for prey)
  float nearest_food_dist = -1.0f; // -1=none in range
  float nearest_food_angle = 0.0f;

  // Self state
  float energy_level = 1.0f; // [0, 1]
  float speed_x = 0.0f;      // [-1, 1]
  float speed_y = 0.0f;      // [-1, 1]

  // Density around agent
  float local_predator_density = 0.0f; // [0, 1]
  float local_prey_density = 0.0f;     // [0, 1]

  // Wall proximity (for clamp mode)
  float wall_left = 1.0f;
  float wall_right = 1.0f;
  float wall_top = 1.0f;
  float wall_bottom = 1.0f;

  static constexpr int SIZE = 15;
  std::vector<float> to_vector() const;
  void write_to(float *buffer) const;
};

// Legacy Physics class removed - all physics is now handled by ECS systems:
// - SensorSystem: builds sensor inputs from ECS components
// - MovementSystem: applies movement and boundaries
// - CombatSystem: processes predator attacks
// These are implemented in simulation_manager.cpp using Registry queries.

} // namespace moonai
