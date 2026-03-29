#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_types.hpp"
#include <cstddef>
#include <cstdint>

// CUDA forward declarations for non-CUDA compilation
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {
namespace gpu {

struct GpuStepParams {
  float world_width = 0.0f;
  float world_height = 0.0f;
  float energy_drain_per_step = 0.0f;
  float vision_range = 120.0f;
  float max_energy = 150.0f;
  int max_age = 0;
  float food_pickup_range = 12.0f;
  float attack_range = 20.0f;
  float energy_gain_from_food = 40.0f;
  float energy_gain_from_kill = 60.0f;
};

class GpuBatch {
public:
  GpuBatch(std::size_t max_agents, std::size_t max_food);
  ~GpuBatch();

  GpuBatch(const GpuBatch &) = delete;
  GpuBatch &operator=(const GpuBatch &) = delete;
  GpuBatch(GpuBatch &&) = delete;
  GpuBatch &operator=(GpuBatch &&) = delete;

  [[nodiscard]] GpuDataBuffer &buffer() {
    return buffer_;
  }
  [[nodiscard]] const GpuDataBuffer &buffer() const {
    return buffer_;
  }

  void launch_build_sensors_async(const GpuStepParams &params,
                                  std::size_t agent_count,
                                  std::size_t food_count);
  void launch_post_inference_async(const GpuStepParams &params,
                                   std::size_t agent_count,
                                   std::size_t food_count);

  void upload_async(std::size_t agent_count, std::size_t food_count);

  void download_async(std::size_t agent_count, std::size_t food_count);

  void synchronize();

  void mark_error();

  [[nodiscard]] bool ok() const noexcept {
    return !had_error_;
  }

  [[nodiscard]] cudaStream_t stream() const {
    return static_cast<cudaStream_t>(stream_);
  }

  [[nodiscard]] std::size_t agent_capacity() const noexcept {
    return buffer_.agent_capacity();
  }

  [[nodiscard]] std::size_t food_capacity() const noexcept {
    return buffer_.food_capacity();
  }

private:
  GpuDataBuffer buffer_;

  int *d_agent_cell_counts_ = nullptr;
  int *d_agent_cell_offsets_ = nullptr;
  int *d_agent_cell_write_offsets_ = nullptr;
  GpuSensorAgentEntry *d_agent_grid_entries_ = nullptr;
  int *d_food_cell_counts_ = nullptr;
  int *d_food_cell_offsets_ = nullptr;
  int *d_food_cell_write_offsets_ = nullptr;
  GpuSensorFoodEntry *d_food_grid_entries_ = nullptr;
  std::size_t grid_cell_capacity_ = 0;
  int grid_cols_ = 0;
  int grid_rows_ = 0;
  float grid_cell_size_ = 0.0f;

  void *stream_ = nullptr;
  bool had_error_ = false;

  void init_cuda_resources();
  void cleanup_cuda_resources();
  void ensure_spatial_grid_capacity(std::size_t cell_count);
  void free_spatial_grid_buffers();
  void check_launch_error();
};

void launch_build_sensors_kernel(
    const float *d_pos_x, const float *d_pos_y, const float *d_vel_x,
    const float *d_vel_y, const float *d_speed, const uint8_t *d_agent_types,
    const uint32_t *d_agent_alive, const float *d_energy,
    const float *d_food_pos_x, const float *d_food_pos_y,
    const uint32_t *d_food_active, const int *d_agent_cell_offsets,
    const GpuSensorAgentEntry *d_agent_entries, const int *d_food_cell_offsets,
    const GpuSensorFoodEntry *d_food_entries, float *d_sensor_inputs,
    std::size_t agent_count, std::size_t food_count, int grid_cols,
    int grid_rows, float grid_cell_size, float world_width, float world_height,
    float vision_range, float max_energy, cudaStream_t stream);

void launch_post_inference_kernel(
    float *d_agent_pos_x, float *d_agent_pos_y, float *d_agent_vel_x,
    float *d_agent_vel_y, const float *d_agent_speed, float *d_agent_energy,
    int *d_agent_age, int *d_agent_reproduction_cooldown,
    uint32_t *d_agent_alive, const uint8_t *d_agent_types,
    float *d_agent_distance_traveled, uint32_t *d_agent_kill_counts,
    int *d_agent_killed_by, const float *d_agent_brain_outputs,
    float *d_food_pos_x, float *d_food_pos_y, uint32_t *d_food_active,
    int *d_food_consumed_by, const int *d_agent_cell_offsets,
    const GpuSensorAgentEntry *d_agent_entries, const int *d_food_cell_offsets,
    const GpuSensorFoodEntry *d_food_entries, int grid_cols, int grid_rows,
    float grid_cell_size, std::size_t agent_count, std::size_t food_count,
    float world_width, float world_height, float energy_drain, int max_age,
    float food_pickup_range, float attack_range, float energy_gain_from_food,
    float energy_gain_from_kill, cudaStream_t stream);

} // namespace gpu
} // namespace moonai
