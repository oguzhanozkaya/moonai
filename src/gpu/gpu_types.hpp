#pragma once

namespace moonai::gpu {

struct GpuSensorAgentEntry {
  unsigned int id;
  unsigned int type;
  float pos_x;
  float pos_y;
};

struct GpuSensorFoodEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
};

} // namespace moonai::gpu
