#include "core/profiler_macros.hpp"
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch.hpp"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>

namespace moonai {
namespace gpu {

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr float kMaxDensity = 10.0f;
constexpr float kMissingTargetSentinel = 2.0f;
constexpr int kUnclaimed = 0x7f7f7f7f;

__device__ float clampf(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

__device__ void apply_wrap(float &dx, float &dy, float world_width,
                           float world_height) {
  const float half_width = world_width * 0.5f;
  const float half_height = world_height * 0.5f;
  if (fabsf(dx) > half_width) {
    dx = dx > 0.0f ? dx - world_width : dx + world_width;
  }
  if (fabsf(dy) > half_height) {
    dy = dy > 0.0f ? dy - world_height : dy + world_height;
  }
}

__device__ int wrap_cell_coord(int coord, int limit) {
  if (limit <= 0) {
    return 0;
  }
  coord %= limit;
  if (coord < 0) {
    coord += limit;
  }
  return coord;
}

__device__ int cell_coord(float pos, float cell_size, int limit) {
  int coord = static_cast<int>(pos / cell_size);
  if (coord < 0) {
    return 0;
  }
  if (coord >= limit) {
    return limit - 1;
  }
  return coord;
}

__device__ bool cell_may_intersect_radius(int cx, int cy, float cell_size,
                                          float origin_x, float origin_y,
                                          float radius, float world_width,
                                          float world_height) {
  const float center_x = (static_cast<float>(cx) + 0.5f) * cell_size;
  const float center_y = (static_cast<float>(cy) + 0.5f) * cell_size;
  float dx = center_x - origin_x;
  float dy = center_y - origin_y;
  apply_wrap(dx, dy, world_width, world_height);

  const float half_size = cell_size * 0.5f;
  const float nearest_x = fmaxf(fabsf(dx) - half_size, 0.0f);
  const float nearest_y = fmaxf(fabsf(dy) - half_size, 0.0f);
  return nearest_x * nearest_x + nearest_y * nearest_y <= radius * radius;
}

__global__ void
kernel_count_agent_cells(const float *__restrict__ agent_pos_x,
                         const float *__restrict__ agent_pos_y,
                         const uint32_t *__restrict__ agent_alive,
                         int *__restrict__ cell_counts, int agent_count,
                         int grid_cols, int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  const int cx = cell_coord(agent_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(agent_pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[cy * grid_cols + cx], 1);
}

__global__ void kernel_scatter_agent_cells(
    const float *__restrict__ agent_pos_x,
    const float *__restrict__ agent_pos_y,
    const uint8_t *__restrict__ agent_types,
    const uint32_t *__restrict__ agent_alive, int *__restrict__ cell_offsets,
    GpuSensorAgentEntry *__restrict__ entries, int agent_count, int grid_cols,
    int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  const int cx = cell_coord(agent_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(agent_pos_y[idx], cell_size, grid_rows);
  const int slot = atomicAdd(&cell_offsets[cy * grid_cols + cx], 1);
  entries[slot] =
      GpuSensorAgentEntry{static_cast<unsigned int>(idx),
                          static_cast<unsigned int>(agent_types[idx]),
                          agent_pos_x[idx], agent_pos_y[idx]};
}

__global__ void kernel_count_food_cells(
    const float *__restrict__ food_pos_x, const float *__restrict__ food_pos_y,
    const uint32_t *__restrict__ food_active, int *__restrict__ cell_counts,
    int food_count, int grid_cols, int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int cx = cell_coord(food_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(food_pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[cy * grid_cols + cx], 1);
}

__global__ void kernel_scatter_food_cells(
    const float *__restrict__ food_pos_x, const float *__restrict__ food_pos_y,
    const uint32_t *__restrict__ food_active, int *__restrict__ cell_offsets,
    GpuSensorFoodEntry *__restrict__ entries, int food_count, int grid_cols,
    int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int cx = cell_coord(food_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(food_pos_y[idx], cell_size, grid_rows);
  const int slot = atomicAdd(&cell_offsets[cy * grid_cols + cx], 1);
  entries[slot] = GpuSensorFoodEntry{static_cast<unsigned int>(idx),
                                     food_pos_x[idx], food_pos_y[idx], 0.0f};
}

__global__ void
kernel_build_sensors(const float *__restrict__ agent_pos_x,
                     const float *__restrict__ agent_pos_y,
                     const float *__restrict__ agent_vel_x,
                     const float *__restrict__ agent_vel_y,
                     const float *__restrict__ agent_speed,
                     const uint32_t *__restrict__ agent_alive,
                     const float *__restrict__ agent_energy,
                     const int *__restrict__ agent_cell_offsets,
                     const GpuSensorAgentEntry *__restrict__ agent_entries,
                     const int *__restrict__ food_cell_offsets,
                     const GpuSensorFoodEntry *__restrict__ food_entries,
                     float *__restrict__ sensor_inputs, int agent_count,
                     int grid_cols, int grid_rows, float grid_cell_size,
                     float world_width, float world_height, float vision_range,
                     float max_energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count) {
    return;
  }

  float *out = sensor_inputs + idx * 12;
  out[0] = kMissingTargetSentinel;
  out[1] = kMissingTargetSentinel;
  out[2] = kMissingTargetSentinel;
  out[3] = kMissingTargetSentinel;
  out[4] = kMissingTargetSentinel;
  out[5] = kMissingTargetSentinel;
  out[6] = 0.0f;
  out[7] = 0.0f;
  out[8] = 0.0f;
  out[9] = 0.0f;
  out[10] = 0.0f;
  out[11] = 0.0f;

  if (agent_alive[idx] == 0) {
    return;
  }

  const float self_x = agent_pos_x[idx];
  const float self_y = agent_pos_y[idx];
  const float vision_sq = vision_range * vision_range;

  float nearest_pred_dist_sq = INFINITY;
  float nearest_prey_dist_sq = INFINITY;
  float nearest_food_dist_sq = INFINITY;
  float pred_dx = 0.0f;
  float pred_dy = 0.0f;
  float prey_dx = 0.0f;
  float prey_dy = 0.0f;
  float food_dx = 0.0f;
  float food_dy = 0.0f;
  int local_predators = 0;
  int local_prey = 0;
  int local_food = 0;

  const int base_cx = cell_coord(self_x, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(self_y, grid_cell_size, grid_rows);

  for (int dy_cell = -1; dy_cell <= 1; ++dy_cell) {
    for (int dx_cell = -1; dx_cell <= 1; ++dx_cell) {
      const int cx = wrap_cell_coord(base_cx + dx_cell, grid_cols);
      const int cy = wrap_cell_coord(base_cy + dy_cell, grid_rows);
      const int cell = cy * grid_cols + cx;
      for (int slot = agent_cell_offsets[cell];
           slot < agent_cell_offsets[cell + 1]; ++slot) {
        const GpuSensorAgentEntry entry = agent_entries[slot];
        const int other = static_cast<int>(entry.id);
        if (other == idx) {
          continue;
        }

        float dx = entry.pos_x - self_x;
        float dy = entry.pos_y - self_y;
        apply_wrap(dx, dy, world_width, world_height);
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq > vision_sq || dist_sq <= 0.0f) {
          continue;
        }

        if (entry.type == 0U) {
          ++local_predators;
          if (dist_sq < nearest_pred_dist_sq) {
            nearest_pred_dist_sq = dist_sq;
            pred_dx = dx;
            pred_dy = dy;
          }
        } else if (entry.type == 1U) {
          ++local_prey;
          if (dist_sq < nearest_prey_dist_sq) {
            nearest_prey_dist_sq = dist_sq;
            prey_dx = dx;
            prey_dy = dy;
          }
        }
      }

      for (int slot = food_cell_offsets[cell];
           slot < food_cell_offsets[cell + 1]; ++slot) {
        float dx = food_entries[slot].pos_x - self_x;
        float dy = food_entries[slot].pos_y - self_y;
        apply_wrap(dx, dy, world_width, world_height);
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq > vision_sq) {
          continue;
        }

        ++local_food;
        if (dist_sq < nearest_food_dist_sq) {
          nearest_food_dist_sq = dist_sq;
          food_dx = dx;
          food_dy = dy;
        }
      }
    }
  }

  if (nearest_pred_dist_sq < INFINITY) {
    out[0] = clampf(pred_dx / vision_range, -1.0f, 1.0f);
    out[1] = clampf(pred_dy / vision_range, -1.0f, 1.0f);
  }
  if (nearest_prey_dist_sq < INFINITY) {
    out[2] = clampf(prey_dx / vision_range, -1.0f, 1.0f);
    out[3] = clampf(prey_dy / vision_range, -1.0f, 1.0f);
  }
  if (nearest_food_dist_sq < INFINITY) {
    out[4] = clampf(food_dx / vision_range, -1.0f, 1.0f);
    out[5] = clampf(food_dy / vision_range, -1.0f, 1.0f);
  }

  out[6] = clampf(agent_energy[idx] / (max_energy * 2.0f), 0.0f, 1.0f);
  if (agent_speed[idx] > 0.0f) {
    out[7] = clampf(agent_vel_x[idx] / agent_speed[idx], -1.0f, 1.0f);
    out[8] = clampf(agent_vel_y[idx] / agent_speed[idx], -1.0f, 1.0f);
  }
  out[9] =
      clampf(static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
  out[10] = clampf(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
  out[11] = clampf(static_cast<float>(local_food) / kMaxDensity, 0.0f, 1.0f);
}

__global__ void kernel_update_vitals(float *__restrict__ agent_energy,
                                     int *__restrict__ agent_age,
                                     uint32_t *__restrict__ agent_alive,
                                     int agent_count, float energy_drain,
                                     int max_age) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  agent_age[idx] += 1;

  agent_energy[idx] -= energy_drain;
  if (agent_energy[idx] <= 0.0f || (max_age > 0 && agent_age[idx] >= max_age)) {
    agent_energy[idx] = 0.0f;
    agent_alive[idx] = 0;
  }
}

__global__ void
kernel_claim_food(const float *__restrict__ agent_pos_x,
                  const float *__restrict__ agent_pos_y,
                  const uint8_t *__restrict__ agent_types,
                  const uint32_t *__restrict__ agent_alive,
                  const uint32_t *__restrict__ food_active,
                  const int *__restrict__ food_cell_offsets,
                  const GpuSensorFoodEntry *__restrict__ food_entries,
                  int *__restrict__ food_consumed_by, int agent_count,
                  int food_count, int grid_cols, int grid_rows,
                  float grid_cell_size, float pickup_range, float world_width,
                  float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 1) {
    return;
  }

  const float px = agent_pos_x[idx];
  const float py = agent_pos_y[idx];
  const float range_sq = pickup_range * pickup_range;

  int best_food = -1;
  float best_dist_sq = range_sq;

  const int cells_to_check =
      static_cast<int>(pickup_range / grid_cell_size) + 1;
  const int base_cx = cell_coord(px, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(py, grid_cell_size, grid_rows);

  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int cx = wrap_cell_coord(base_cx + dx_cell, grid_cols);
      const int cy = wrap_cell_coord(base_cy + dy_cell, grid_rows);
      if (!cell_may_intersect_radius(cx, cy, grid_cell_size, px, py,
                                     pickup_range, world_width, world_height)) {
        continue;
      }

      const int cell = cy * grid_cols + cx;
      for (int slot = food_cell_offsets[cell];
           slot < food_cell_offsets[cell + 1]; ++slot) {
        float dx = food_entries[slot].pos_x - px;
        float dy = food_entries[slot].pos_y - py;
        apply_wrap(dx, dy, world_width, world_height);
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq <= best_dist_sq) {
          best_dist_sq = dist_sq;
          best_food = static_cast<int>(food_entries[slot].id);
        }
      }
    }
  }

  if (best_food >= 0) {
    atomicMin(&food_consumed_by[best_food], idx);
  }
}

__global__ void kernel_finalize_food(float *__restrict__ agent_energy,
                                     const uint32_t *__restrict__ agent_alive,
                                     uint32_t *__restrict__ food_active,
                                     int *__restrict__ food_consumed_by,
                                     int food_count,
                                     float energy_gain_from_food) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int prey_idx = food_consumed_by[idx];
  if (prey_idx != kUnclaimed && prey_idx >= 0 && agent_alive[prey_idx] != 0) {
    food_active[idx] = 0;
    atomicAdd(&agent_energy[prey_idx], energy_gain_from_food);
  }
}

__global__ void
kernel_claim_combat(const float *__restrict__ agent_pos_x,
                    const float *__restrict__ agent_pos_y,
                    const uint8_t *__restrict__ agent_types,
                    const uint32_t *__restrict__ agent_alive,
                    const int *__restrict__ agent_cell_offsets,
                    const GpuSensorAgentEntry *__restrict__ agent_entries,
                    int *__restrict__ agent_killed_by, int agent_count,
                    int grid_cols, int grid_rows, float grid_cell_size,
                    float interaction_range, float world_width,
                    float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 0) {
    return;
  }

  const float px = agent_pos_x[idx];
  const float py = agent_pos_y[idx];
  const float range_sq = interaction_range * interaction_range;
  int best_prey = -1;
  float best_dist_sq = range_sq;

  const int cells_to_check =
      static_cast<int>(interaction_range / grid_cell_size) + 1;
  const int base_cx = cell_coord(px, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(py, grid_cell_size, grid_rows);

  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int cx = wrap_cell_coord(base_cx + dx_cell, grid_cols);
      const int cy = wrap_cell_coord(base_cy + dy_cell, grid_rows);
      if (!cell_may_intersect_radius(cx, cy, grid_cell_size, px, py,
                                     interaction_range, world_width,
                                     world_height)) {
        continue;
      }

      const int cell = cy * grid_cols + cx;
      for (int slot = agent_cell_offsets[cell];
           slot < agent_cell_offsets[cell + 1]; ++slot) {
        const GpuSensorAgentEntry entry = agent_entries[slot];
        const int prey_idx = static_cast<int>(entry.id);
        if (agent_alive[prey_idx] == 0 || entry.type != 1U) {
          continue;
        }

        float dx = entry.pos_x - px;
        float dy = entry.pos_y - py;
        apply_wrap(dx, dy, world_width, world_height);
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq <= best_dist_sq) {
          best_dist_sq = dist_sq;
          best_prey = prey_idx;
        }
      }
    }
  }

  if (best_prey >= 0) {
    atomicMin(&agent_killed_by[best_prey], idx);
  }
}

__global__ void kernel_finalize_combat(float *__restrict__ agent_energy,
                                       uint32_t *__restrict__ agent_alive,
                                       const uint8_t *__restrict__ agent_types,
                                       uint32_t *__restrict__ agent_kill_counts,
                                       int *__restrict__ agent_killed_by,
                                       int agent_count,
                                       float energy_gain_from_kill) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 1) {
    return;
  }

  const int killer_idx = agent_killed_by[idx];
  if (killer_idx != kUnclaimed && killer_idx >= 0 &&
      agent_alive[killer_idx] != 0) {
    agent_alive[idx] = 0;
    atomicAdd(&agent_energy[killer_idx], energy_gain_from_kill);
    atomicAdd(&agent_kill_counts[killer_idx], 1U);
  }
}

__global__ void kernel_apply_movement(
    float *__restrict__ agent_pos_x, float *__restrict__ agent_pos_y,
    float *__restrict__ agent_vel_x, float *__restrict__ agent_vel_y,
    const float *__restrict__ agent_speed,
    const uint32_t *__restrict__ agent_alive,
    float *__restrict__ agent_distance_traveled,
    const float *__restrict__ agent_brain_outputs, int agent_count,
    float world_width, float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  float dx = agent_brain_outputs[idx * 2 + 0];
  float dy = agent_brain_outputs[idx * 2 + 1];
  float len = sqrtf(dx * dx + dy * dy);
  if (len > 1e-6f) {
    dx /= len;
    dy /= len;
  } else {
    dx = 0.0f;
    dy = 0.0f;
  }

  agent_vel_x[idx] = dx * agent_speed[idx];
  agent_vel_y[idx] = dy * agent_speed[idx];

  const float old_x = agent_pos_x[idx];
  const float old_y = agent_pos_y[idx];

  agent_pos_x[idx] += agent_vel_x[idx];
  agent_pos_y[idx] += agent_vel_y[idx];

  while (agent_pos_x[idx] < 0.0f)
    agent_pos_x[idx] += world_width;
  while (agent_pos_x[idx] >= world_width)
    agent_pos_x[idx] -= world_width;
  while (agent_pos_y[idx] < 0.0f)
    agent_pos_y[idx] += world_height;
  while (agent_pos_y[idx] >= world_height)
    agent_pos_y[idx] -= world_height;

  float move_dx = agent_pos_x[idx] - old_x;
  float move_dy = agent_pos_y[idx] - old_y;
  apply_wrap(move_dx, move_dy, world_width, world_height);
  agent_distance_traveled[idx] += sqrtf(move_dx * move_dx + move_dy * move_dy);
}
} // namespace

GpuBatch::GpuBatch(std::size_t max_agents, std::size_t max_food)
    : buffer_(max_agents, max_food) {
  init_cuda_resources();
}

GpuBatch::~GpuBatch() {
  cleanup_cuda_resources();
}

void GpuBatch::ensure_spatial_grid_capacity(std::size_t cell_count) {
  if (cell_count <= grid_cell_capacity_) {
    return;
  }

  free_spatial_grid_buffers();

  CUDA_CHECK(cudaMalloc(&d_agent_cell_counts_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_agent_cell_offsets_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_agent_cell_write_offsets_, cell_count * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_agent_grid_entries_,
                 buffer_.agent_capacity() * sizeof(GpuSensorAgentEntry)));

  CUDA_CHECK(cudaMalloc(&d_food_cell_counts_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_cell_offsets_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_cell_write_offsets_, cell_count * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_grid_entries_,
                        buffer_.food_capacity() * sizeof(GpuSensorFoodEntry)));
  grid_cell_capacity_ = cell_count;
}

void GpuBatch::free_spatial_grid_buffers() {
  if (d_agent_cell_counts_) {
    cudaFree(d_agent_cell_counts_);
    d_agent_cell_counts_ = nullptr;
  }
  if (d_agent_cell_offsets_) {
    cudaFree(d_agent_cell_offsets_);
    d_agent_cell_offsets_ = nullptr;
  }
  if (d_agent_cell_write_offsets_) {
    cudaFree(d_agent_cell_write_offsets_);
    d_agent_cell_write_offsets_ = nullptr;
  }
  if (d_agent_grid_entries_) {
    cudaFree(d_agent_grid_entries_);
    d_agent_grid_entries_ = nullptr;
  }
  if (d_food_cell_counts_) {
    cudaFree(d_food_cell_counts_);
    d_food_cell_counts_ = nullptr;
  }
  if (d_food_cell_offsets_) {
    cudaFree(d_food_cell_offsets_);
    d_food_cell_offsets_ = nullptr;
  }
  if (d_food_cell_write_offsets_) {
    cudaFree(d_food_cell_write_offsets_);
    d_food_cell_write_offsets_ = nullptr;
  }
  if (d_food_grid_entries_) {
    cudaFree(d_food_grid_entries_);
    d_food_grid_entries_ = nullptr;
  }
  grid_cell_capacity_ = 0;
}

void GpuBatch::init_cuda_resources() {
  const cudaError_t err =
      cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_));
  if (err != cudaSuccess) {
    had_error_ = true;
  }
}

void GpuBatch::cleanup_cuda_resources() {
  free_spatial_grid_buffers();
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

void GpuBatch::upload_async(std::size_t agent_count, std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("gpu_upload", static_cast<cudaStream_t>(stream_));
  buffer_.upload_async(agent_count, food_count,
                       static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatch::download_async(std::size_t agent_count, std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("gpu_download", static_cast<cudaStream_t>(stream_));
  buffer_.download_async(agent_count, food_count,
                         static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatch::launch_build_sensors_async(const GpuStepParams &params,
                                          std::size_t agent_count,
                                          std::size_t food_count) {
  if (agent_count == 0 || had_error_) {
    return;
  }

  MOONAI_PROFILE_SCOPE("gpu_sensing", static_cast<cudaStream_t>(stream_));

  grid_cell_size_ = params.vision_range;
  grid_cols_ = std::max(
      1, static_cast<int>(std::ceil(params.world_width / grid_cell_size_)));
  grid_rows_ = std::max(
      1, static_cast<int>(std::ceil(params.world_height / grid_cell_size_)));
  const std::size_t cell_count = static_cast<std::size_t>(grid_cols_) *
                                 static_cast<std::size_t>(grid_rows_);
  ensure_spatial_grid_capacity(cell_count);

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int food_blocks =
      (static_cast<int>(food_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  {
    MOONAI_PROFILE_SCOPE("gpu_grid_build", stream);
    CUDA_CHECK(cudaMemsetAsync(d_agent_cell_counts_, 0,
                               (cell_count + 1) * sizeof(int), stream));
    kernel_count_agent_cells<<<blocks, kThreadsPerBlock, 0, stream>>>(
        buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
        buffer_.device_agent_alive(), d_agent_cell_counts_,
        static_cast<int>(agent_count), grid_cols_, grid_rows_, grid_cell_size_);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_agent_cell_counts_,
                           d_agent_cell_counts_ + cell_count + 1,
                           d_agent_cell_offsets_);
    CUDA_CHECK(cudaMemcpyAsync(d_agent_cell_write_offsets_,
                               d_agent_cell_offsets_, cell_count * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
    kernel_scatter_agent_cells<<<blocks, kThreadsPerBlock, 0, stream>>>(
        buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
        buffer_.device_agent_types(), buffer_.device_agent_alive(),
        d_agent_cell_write_offsets_, d_agent_grid_entries_,
        static_cast<int>(agent_count), grid_cols_, grid_rows_, grid_cell_size_);

    CUDA_CHECK(cudaMemsetAsync(d_food_cell_counts_, 0,
                               (cell_count + 1) * sizeof(int), stream));
    if (food_count > 0) {
      kernel_count_food_cells<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
          buffer_.device_food_positions_x(), buffer_.device_food_positions_y(),
          buffer_.device_food_active(), d_food_cell_counts_,
          static_cast<int>(food_count), grid_cols_, grid_rows_,
          grid_cell_size_);
    }
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_food_cell_counts_,
                           d_food_cell_counts_ + cell_count + 1,
                           d_food_cell_offsets_);
    CUDA_CHECK(cudaMemcpyAsync(d_food_cell_write_offsets_, d_food_cell_offsets_,
                               cell_count * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
    if (food_count > 0) {
      kernel_scatter_food_cells<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
          buffer_.device_food_positions_x(), buffer_.device_food_positions_y(),
          buffer_.device_food_active(), d_food_cell_write_offsets_,
          d_food_grid_entries_, static_cast<int>(food_count), grid_cols_,
          grid_rows_, grid_cell_size_);
    }
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_sensor_kernel", stream);
    kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
        buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
        buffer_.device_agent_velocities_x(),
        buffer_.device_agent_velocities_y(), buffer_.device_agent_speed(),
        buffer_.device_agent_alive(), buffer_.device_agent_energy(),
        d_agent_cell_offsets_, d_agent_grid_entries_, d_food_cell_offsets_,
        d_food_grid_entries_, buffer_.device_agent_sensor_inputs(),
        static_cast<int>(agent_count), grid_cols_, grid_rows_, grid_cell_size_,
        params.world_width, params.world_height, params.vision_range,
        params.max_energy);
  }
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_agent_brain_outputs(), 0,
                             agent_count * 2 * sizeof(float), stream));
  check_launch_error();
}

void GpuBatch::launch_post_inference_async(const GpuStepParams &params,
                                           std::size_t agent_count,
                                           std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("gpu_step", static_cast<cudaStream_t>(stream_));
  launch_post_inference_kernel(
      buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
      buffer_.device_agent_velocities_x(), buffer_.device_agent_velocities_y(),
      buffer_.device_agent_speed(), buffer_.device_agent_energy(),
      buffer_.device_agent_age(), buffer_.device_agent_alive(),
      buffer_.device_agent_types(), buffer_.device_agent_distance_traveled(),
      buffer_.device_agent_kill_counts(), buffer_.device_agent_killed_by(),
      buffer_.device_agent_brain_outputs(), buffer_.device_food_positions_x(),
      buffer_.device_food_positions_y(), buffer_.device_food_active(),
      buffer_.device_food_consumed_by(), d_agent_cell_offsets_,
      d_agent_grid_entries_, d_food_cell_offsets_, d_food_grid_entries_,
      grid_cols_, grid_rows_, grid_cell_size_, agent_count, food_count,
      params.world_width, params.world_height, params.energy_drain_per_step,
      params.max_age, params.interaction_range, params.energy_gain_from_food,
      params.energy_gain_from_kill, static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatch::synchronize() {
  const cudaError_t err =
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  if (err != cudaSuccess) {
    spdlog::error("GPU synchronize failed: {}", cudaGetErrorString(err));
    had_error_ = true;
  }
}

void GpuBatch::mark_error() {
  had_error_ = true;
}

void GpuBatch::check_launch_error() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("GPU kernel launch failed: {}", cudaGetErrorString(err));
    mark_error();
  }
}

void launch_build_sensors_kernel(
    const float *d_pos_x, const float *d_pos_y, const float *d_vel_x,
    const float *d_vel_y, const float *d_speed, const uint32_t *d_agent_alive,
    const float *d_energy, const int *d_agent_cell_offsets,
    const GpuSensorAgentEntry *d_agent_entries, const int *d_food_cell_offsets,
    const GpuSensorFoodEntry *d_food_entries, float *d_sensor_inputs,
    std::size_t agent_count, int grid_cols, int grid_rows, float grid_cell_size,
    float world_width, float world_height, float vision_range, float max_energy,
    cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_speed, d_agent_alive, d_energy,
      d_agent_cell_offsets, d_agent_entries, d_food_cell_offsets,
      d_food_entries, d_sensor_inputs, static_cast<int>(agent_count), grid_cols,
      grid_rows, grid_cell_size, world_width, world_height, vision_range,
      max_energy);
}

void launch_post_inference_kernel(
    float *d_agent_pos_x, float *d_agent_pos_y, float *d_agent_vel_x,
    float *d_agent_vel_y, const float *d_agent_speed, float *d_agent_energy,
    int *d_agent_age, uint32_t *d_agent_alive, const uint8_t *d_agent_types,
    float *d_agent_distance_traveled, uint32_t *d_agent_kill_counts,
    int *d_agent_killed_by, const float *d_agent_brain_outputs,
    float *d_food_pos_x, float *d_food_pos_y, uint32_t *d_food_active,
    int *d_food_consumed_by, const int *d_agent_cell_offsets,
    const GpuSensorAgentEntry *d_agent_entries, const int *d_food_cell_offsets,
    const GpuSensorFoodEntry *d_food_entries, int grid_cols, int grid_rows,
    float grid_cell_size, std::size_t agent_count, std::size_t food_count,
    float world_width, float world_height, float energy_drain, int max_age,
    float interaction_range, float energy_gain_from_food,
    float energy_gain_from_kill, cudaStream_t stream) {
  if (agent_count == 0) {
    return;
  }

  const int agent_blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int food_blocks =
      (static_cast<int>(food_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;

  kernel_update_vitals<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_energy, d_agent_age, d_agent_alive, static_cast<int>(agent_count),
      energy_drain, max_age);

  CUDA_CHECK(cudaMemsetAsync(d_agent_kill_counts, 0,
                             agent_count * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaMemsetAsync(d_agent_killed_by, 0x7F, agent_count * sizeof(int),
                             stream));

  if (food_count > 0) {
    CUDA_CHECK(cudaMemsetAsync(d_food_consumed_by, 0x7F,
                               food_count * sizeof(int), stream));
    kernel_claim_food<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
        d_agent_pos_x, d_agent_pos_y, d_agent_types, d_agent_alive,
        d_food_active, d_food_cell_offsets, d_food_entries, d_food_consumed_by,
        static_cast<int>(agent_count), static_cast<int>(food_count), grid_cols,
        grid_rows, grid_cell_size, interaction_range, world_width,
        world_height);
    kernel_finalize_food<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
        d_agent_energy, d_agent_alive, d_food_active, d_food_consumed_by,
        static_cast<int>(food_count), energy_gain_from_food);
  }

  kernel_claim_combat<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_pos_x, d_agent_pos_y, d_agent_types, d_agent_alive,
      d_agent_cell_offsets, d_agent_entries, d_agent_killed_by,
      static_cast<int>(agent_count), grid_cols, grid_rows, grid_cell_size,
      interaction_range, world_width, world_height);
  kernel_finalize_combat<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_energy, d_agent_alive, d_agent_types, d_agent_kill_counts,
      d_agent_killed_by, static_cast<int>(agent_count), energy_gain_from_kill);

  kernel_apply_movement<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_pos_x, d_agent_pos_y, d_agent_vel_x, d_agent_vel_y, d_agent_speed,
      d_agent_alive, d_agent_distance_traveled, d_agent_brain_outputs,
      static_cast<int>(agent_count), world_width, world_height);
}

} // namespace gpu
} // namespace moonai
