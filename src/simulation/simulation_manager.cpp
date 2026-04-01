#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "core/metrics.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/simulation_step_systems.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config) : config_(config) {}

SimulationManager::~SimulationManager() = default;

void SimulationManager::initialize(AppState &state) {
  state.food.initialize(config_, state.runtime.rng);
}

void SimulationManager::step(AppState &state, EvolutionManager &evolution) {
  metrics::begin_step(state);

  if (state.runtime.gpu_enabled) {
    this->step_gpu(state, evolution);
  } else {
    this->step_cpu(state, evolution);
  }

  state.predator.compact();
  state.prey.compact();
  state.food.respawn_step(config_, state.runtime.step, state.runtime.rng.seed());

  reproduction(state, evolution, state.predator);
  reproduction(state, evolution, state.prey);
  if (config_.species_update_interval_steps > 0 && (state.runtime.step % config_.species_update_interval_steps) == 0) {
    evolution.refresh_species(state);
  }
}

void SimulationManager::reset(AppState &state) {
  initialize(state);
}

namespace {
// Round up to the next power of 2 for GPU buffer allocation
inline std::size_t next_power_of_2(std::size_t n) {
  if (n == 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}
} // namespace

void SimulationManager::collect_gpu_step_events(AppState &state, const std::vector<uint8_t> &was_food_active) {
  MOONAI_PROFILE_SCOPE("collect_gpu_step_events");

  auto &predator_buffer = gpu_batch_->predator_buffer();
  auto &prey_buffer = gpu_batch_->prey_buffer();
  auto &food_buffer = gpu_batch_->food_buffer();

  for (std::size_t food_idx = 0; food_idx < state.food.size(); ++food_idx) {
    const int prey_idx = food_buffer.host_consumed_by()[food_idx];
    if (was_food_active[food_idx] && !state.food.active[food_idx] && prey_idx >= 0 &&
        static_cast<uint32_t>(prey_idx) < state.prey.size()) {
      ++state.prey.consumption[prey_idx];
      ++state.metrics.step_delta.food_eaten;
    }
  }

  const uint32_t predator_count = static_cast<uint32_t>(state.predator.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (predator_buffer.host_kill_counts()[predator_idx] > 0) {
      state.predator.consumption[predator_idx] += static_cast<int>(predator_buffer.host_kill_counts()[predator_idx]);
    }
    if (state.predator.alive[predator_idx] == 0) {
      ++state.metrics.step_delta.deaths;
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(state.prey.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = prey_buffer.host_claimed_by()[prey_idx];
    if (killer_idx >= 0 && static_cast<uint32_t>(killer_idx) < state.predator.size()) {
      ++state.metrics.step_delta.kills;
    }
    if (state.prey.alive[prey_idx] == 0) {
      ++state.metrics.step_delta.deaths;
    }
  }
}

void SimulationManager::ensure_gpu_capacity(std::size_t predator_count, std::size_t prey_count,
                                            std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");

  const bool needs_batch = !gpu_batch_;
  const bool predators_exceeded = gpu_batch_ && predator_count > gpu_batch_->predator_capacity();
  const bool prey_exceeded = gpu_batch_ && prey_count > gpu_batch_->prey_capacity();
  const bool food_exceeded = gpu_batch_ && food_count > gpu_batch_->food_capacity();
  const bool needs_resize = predators_exceeded || prey_exceeded || food_exceeded;

  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_predator_capacity = gpu_batch_ ? gpu_batch_->predator_capacity() : 0;
  const std::size_t current_prey_capacity = gpu_batch_ ? gpu_batch_->prey_capacity() : 0;
  const std::size_t current_food_capacity = gpu_batch_ ? gpu_batch_->food_capacity() : 0;

  // Only resize buffers that exceeded their capacity
  const std::size_t new_predator_capacity =
      needs_batch || predators_exceeded
          ? next_power_of_2(current_predator_capacity == 0 ? predator_count
                                                           : std::max(predator_count, current_predator_capacity * 2))
          : current_predator_capacity;
  const std::size_t new_prey_capacity =
      needs_batch || prey_exceeded
          ? next_power_of_2(current_prey_capacity == 0 ? prey_count : std::max(prey_count, current_prey_capacity * 2))
          : current_prey_capacity;
  const std::size_t new_food_capacity =
      needs_batch || food_exceeded
          ? next_power_of_2(current_food_capacity == 0 ? food_count : std::max(food_count, current_food_capacity * 2))
          : current_food_capacity;

  gpu_batch_ = std::make_unique<gpu::GpuBatch>(new_predator_capacity, new_prey_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(AppState &state, bool enable) {
  if (enable) {
    ensure_gpu_capacity(static_cast<std::size_t>(config_.predator_count), static_cast<std::size_t>(config_.prey_count),
                        static_cast<std::size_t>(config_.food_count));
    state.runtime.gpu_enabled = true;
  } else {
    disable_gpu(state);
  }
}

void SimulationManager::disable_gpu(AppState &state) {
  gpu_batch_.reset();
  state.runtime.gpu_enabled = false;
  spdlog::info("GPU batch processing disabled");
}

void SimulationManager::step_gpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");

  std::vector<uint8_t> was_food_active = state.food.active;

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error("GPU batch not initialized or in error state, falling back to CPU");
    return step_cpu(state, evolution);
  }

  ensure_gpu_capacity(predator_count, prey_count, food_count);

  {
    MOONAI_PROFILE_SCOPE("gpu_pack_state");

    auto &predator_buffer = gpu_batch_->predator_buffer();
    auto &prey_buffer = gpu_batch_->prey_buffer();
    auto &food_buffer = gpu_batch_->food_buffer();

    if (predator_count > 0) {
      std::memcpy(predator_buffer.host_positions_x(), state.predator.pos_x.data(), predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_positions_y(), state.predator.pos_y.data(), predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_velocities_x(), state.predator.vel_x.data(), predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_velocities_y(), state.predator.vel_y.data(), predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_energy(), state.predator.energy.data(), predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_age(), state.predator.age.data(), predator_count * sizeof(int));
      for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
        predator_buffer.host_alive()[i] = state.predator.alive[i];
      }
    }

    if (prey_count > 0) {
      std::memcpy(prey_buffer.host_positions_x(), state.prey.pos_x.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_positions_y(), state.prey.pos_y.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_velocities_x(), state.prey.vel_x.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_velocities_y(), state.prey.vel_y.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_energy(), state.prey.energy.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_age(), state.prey.age.data(), prey_count * sizeof(int));
      for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
        prey_buffer.host_alive()[i] = state.prey.alive[i];
      }
    }

    if (food_count > 0) {
      std::memcpy(food_buffer.host_positions_x(), state.food.pos_x.data(), food_count * sizeof(float));
      std::memcpy(food_buffer.host_positions_y(), state.food.pos_y.data(), food_count * sizeof(float));
      for (std::size_t i = 0; i < food_count; ++i) {
        food_buffer.host_active()[i] = state.food.active[i];
        food_buffer.host_consumed_by()[i] = -1;
      }
    }
  }

  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.vision_range = config_.vision_range;
  params.max_energy = static_cast<float>(config_.initial_energy);
  params.max_age = config_.max_steps;
  params.interaction_range = config_.interaction_range;
  params.energy_gain_from_food = static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill = static_cast<float>(config_.energy_gain_from_kill);
  params.predator_speed = config_.predator_speed;
  params.prey_speed = config_.prey_speed;

  gpu_batch_->upload_async(predator_count, prey_count, food_count);
  gpu_batch_->launch_build_sensors_async(params, predator_count, prey_count, food_count);

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_neural");
    if (!evolution.launch_gpu_neural(state, *gpu_batch_)) {
      MOONAI_PROFILE_SCOPE("cpu_fallback");
      spdlog::error("GPU neural inference failed, disabling GPU path and "
                    "retrying on CPU");
      disable_gpu(state);
      return step_cpu(state, evolution);
    }
  }

  gpu_batch_->launch_post_inference_async(params, predator_count, prey_count, food_count);
  gpu_batch_->download_async(predator_count, prey_count, food_count);
  gpu_batch_->synchronize();

  if (!gpu_batch_->ok()) {
    MOONAI_PROFILE_SCOPE("cpu_fallback");
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    disable_gpu(state);
    return step_cpu(state, evolution);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_apply_results");
    auto &predator_buffer = gpu_batch_->predator_buffer();
    auto &prey_buffer = gpu_batch_->prey_buffer();
    auto &food_buffer = gpu_batch_->food_buffer();

    if (predator_count > 0) {
      std::memcpy(state.predator.pos_x.data(), predator_buffer.host_positions_x(), predator_count * sizeof(float));
      std::memcpy(state.predator.pos_y.data(), predator_buffer.host_positions_y(), predator_count * sizeof(float));
      std::memcpy(state.predator.vel_x.data(), predator_buffer.host_velocities_x(), predator_count * sizeof(float));
      std::memcpy(state.predator.vel_y.data(), predator_buffer.host_velocities_y(), predator_count * sizeof(float));
      std::memcpy(state.predator.energy.data(), predator_buffer.host_energy(), predator_count * sizeof(float));
      std::memcpy(state.predator.age.data(), predator_buffer.host_age(), predator_count * sizeof(int));
      // Sensor inputs and brain outputs are device-only, not copied back to CPU
      for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
        state.predator.alive[i] = static_cast<uint8_t>(predator_buffer.host_alive()[i]);
      }
    }

    if (prey_count > 0) {
      std::memcpy(state.prey.pos_x.data(), prey_buffer.host_positions_x(), prey_count * sizeof(float));
      std::memcpy(state.prey.pos_y.data(), prey_buffer.host_positions_y(), prey_count * sizeof(float));
      std::memcpy(state.prey.vel_x.data(), prey_buffer.host_velocities_x(), prey_count * sizeof(float));
      std::memcpy(state.prey.vel_y.data(), prey_buffer.host_velocities_y(), prey_count * sizeof(float));
      std::memcpy(state.prey.energy.data(), prey_buffer.host_energy(), prey_count * sizeof(float));
      std::memcpy(state.prey.age.data(), prey_buffer.host_age(), prey_count * sizeof(int));
      // Sensor inputs and brain outputs are device-only, not copied back to CPU
      for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
        state.prey.alive[i] = static_cast<uint8_t>(prey_buffer.host_alive()[i]);
      }
    }

    if (food_count > 0) {
      std::memcpy(state.food.pos_x.data(), food_buffer.host_positions_x(), food_count * sizeof(float));
      std::memcpy(state.food.pos_y.data(), food_buffer.host_positions_y(), food_count * sizeof(float));
      for (std::size_t i = 0; i < food_count; ++i) {
        state.food.active[i] = static_cast<uint8_t>(food_buffer.host_active()[i]);
      }
    }
  }

  collect_gpu_step_events(state, was_food_active);
}

namespace {

class DenseReproductionGrid {
public:
  DenseReproductionGrid(float world_width, float world_height, float cell_size, std::size_t entity_count)
      : cell_size_(std::max(cell_size, 1.0f)),
        cols_(std::max(1, static_cast<int>(std::ceil(world_width / cell_size_)))),
        rows_(std::max(1, static_cast<int>(std::ceil(world_height / cell_size_)))),
        counts_(static_cast<std::size_t>(cols_ * rows_), 0), offsets_(static_cast<std::size_t>(cols_ * rows_) + 1, 0),
        write_offsets_(static_cast<std::size_t>(cols_ * rows_), 0), entries_(entity_count, INVALID_ENTITY) {}

  void build(const AgentRegistry &registry, std::size_t entity_count) {
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(offsets_.begin(), offsets_.end(), 0);

    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(registry.pos_x[idx], registry.pos_y[idx]);
      counts_[static_cast<std::size_t>(cell)] += 1;
    }

    for (std::size_t cell = 0; cell < counts_.size(); ++cell) {
      offsets_[cell + 1] = offsets_[cell] + counts_[cell];
    }

    std::copy(offsets_.begin(), offsets_.end() - 1, write_offsets_.begin());
    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(registry.pos_x[idx], registry.pos_y[idx]);
      const std::size_t slot = static_cast<std::size_t>(write_offsets_[static_cast<std::size_t>(cell)]++);
      entries_[slot] = idx;
    }
  }

  template <typename Callback> void for_each_candidate(Vec2 center, float radius, Callback &&callback) const {
    const int cells_to_check = std::max(1, static_cast<int>(std::ceil(radius / cell_size_)));
    const int base_x = cell_coord(center.x, cols_);
    const int base_y = cell_coord(center.y, rows_);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
      for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
        const int cell = flat_index(clamp_cell(base_x + dx, cols_), clamp_cell(base_y + dy, rows_));
        for (int slot = offsets_[static_cast<std::size_t>(cell)]; slot < offsets_[static_cast<std::size_t>(cell) + 1];
             ++slot) {
          callback(entries_[static_cast<std::size_t>(slot)]);
        }
      }
    }
  }

private:
  int cell_coord(float value, int limit) const {
    int coord = static_cast<int>(value / cell_size_);
    if (coord < 0) {
      return 0;
    }
    if (coord >= limit) {
      return limit - 1;
    }
    return coord;
  }

  int clamp_cell(int coord, int limit) const {
    if (coord < 0) {
      return 0;
    }
    if (coord >= limit) {
      return limit - 1;
    }
    return coord;
  }

  int flat_index(int x, int y) const {
    return y * cols_ + x;
  }

  int cell_index(float x, float y) const {
    return flat_index(cell_coord(x, cols_), cell_coord(y, rows_));
  }

  float cell_size_;
  int cols_;
  int rows_;
  std::vector<int> counts_;
  std::vector<int> offsets_;
  std::vector<int> write_offsets_;
  std::vector<uint32_t> entries_;
};

} // namespace

void SimulationManager::step_cpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");

  const std::vector<uint8_t> was_food_active = state.food.active;

  std::vector<int> food_consumed_by(state.food.size(), -1);
  std::vector<int> killed_by(state.prey.size(), -1);
  std::vector<uint32_t> kill_counts(state.predator.size(), 0U);

  std::vector<float> predator_sensors;
  std::vector<float> prey_sensors;
  simulation_detail::build_sensors(state.predator, state.predator, state.prey, state.food, config_,
                                   config_.predator_speed, predator_sensors);
  simulation_detail::build_sensors(state.prey, state.predator, state.prey, state.food, config_, config_.prey_speed,
                                   prey_sensors);

  std::vector<float> predator_decisions;
  std::vector<float> prey_decisions;
  evolution.compute_actions(state, predator_sensors, prey_sensors, predator_decisions, prey_decisions);

  simulation_detail::update_vitals(state.predator, config_);
  simulation_detail::update_vitals(state.prey, config_);
  simulation_detail::process_food(state.prey, state.food, config_, food_consumed_by);
  simulation_detail::process_combat(state.predator, state.prey, config_, killed_by, kill_counts);
  simulation_detail::apply_movement(state.predator, config_, config_.predator_speed, predator_decisions);
  simulation_detail::apply_movement(state.prey, config_, config_.prey_speed, prey_decisions);

  simulation_detail::collect_food_events(state.prey, state.food, was_food_active, food_consumed_by,
                                         state.metrics.step_delta);
  simulation_detail::collect_combat_events(state.predator, state.prey, killed_by, kill_counts,
                                           state.metrics.step_delta);
  simulation_detail::collect_death_events(state.predator, state.metrics.step_delta);
  simulation_detail::collect_death_events(state.prey, state.metrics.step_delta);
}

void SimulationManager::reproduction(AppState &state, EvolutionManager &evolution, AgentRegistry &registry) {
  std::vector<uint8_t> used(registry.size(), 0);

  DenseReproductionGrid grid(static_cast<float>(config_.grid_size), static_cast<float>(config_.grid_size),
                             config_.mate_range, registry.size());
  grid.build(registry, registry.size());
  const float world_size = static_cast<float>(config_.grid_size);
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (registry.energy[idx] < config_.reproduction_energy_threshold || used[idx] != 0) {
      continue;
    }

    const Vec2 pos{registry.pos_x[idx], registry.pos_y[idx]};
    uint32_t best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    grid.for_each_candidate(pos, config_.mate_range, [&](uint32_t mate_id) {
      if (mate_id == idx || used[mate_id] != 0 || registry.energy[mate_id] < config_.reproduction_energy_threshold) {
        return;
      }

      const Vec2 mate_pos{registry.pos_x[mate_id], registry.pos_y[mate_id]};
      const Vec2 diff{mate_pos.x - pos.x, mate_pos.y - pos.y};
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    });

    if (best_mate != INVALID_ENTITY) {
      const Vec2 mate_pos{registry.pos_x[best_mate], registry.pos_y[best_mate]};
      const Vec2 mid_pos{(pos.x + mate_pos.x) * 0.5f, (pos.y + mate_pos.y) * 0.5f};
      const float clamped_x = std::clamp(mid_pos.x, 0.0f, world_size);
      const float clamped_y = std::clamp(mid_pos.y, 0.0f, world_size);
      const uint32_t child = evolution.create_offspring(state, registry, idx, best_mate, {clamped_x, clamped_y});
      if (child != INVALID_ENTITY) {
        ++state.metrics.step_delta.births;
      }
      used[idx] = 1;
      used[best_mate] = 1;
    }
  }
}

} // namespace moonai
