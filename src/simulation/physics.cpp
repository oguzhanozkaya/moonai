#include "simulation/physics.hpp"

namespace moonai {

std::vector<float> SensorInput::to_vector() const {
  return {nearest_predator_dist,
          nearest_predator_angle,
          nearest_prey_dist,
          nearest_prey_angle,
          nearest_food_dist,
          nearest_food_angle,
          energy_level,
          speed_x,
          speed_y,
          local_predator_density,
          local_prey_density,
          wall_left,
          wall_right,
          wall_top,
          wall_bottom};
}

void SensorInput::write_to(float *buffer) const {
  buffer[0] = nearest_predator_dist;
  buffer[1] = nearest_predator_angle;
  buffer[2] = nearest_prey_dist;
  buffer[3] = nearest_prey_angle;
  buffer[4] = nearest_food_dist;
  buffer[5] = nearest_food_angle;
  buffer[6] = energy_level;
  buffer[7] = speed_x;
  buffer[8] = speed_y;
  buffer[9] = local_predator_density;
  buffer[10] = local_prey_density;
  buffer[11] = wall_left;
  buffer[12] = wall_right;
  buffer[13] = wall_top;
  buffer[14] = wall_bottom;
}

} // namespace moonai
