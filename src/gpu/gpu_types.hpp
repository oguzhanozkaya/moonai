#pragma once

namespace moonai::gpu {

struct alignas(16) GpuSensorAgentEntry {
  unsigned int id;
  unsigned int type;
  float pos_x;
  float pos_y;
};

struct alignas(16) GpuSensorFoodEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
  float padding;
};

} // namespace moonai::gpu
