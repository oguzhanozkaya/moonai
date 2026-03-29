#include "simulation/simulation_manager.hpp"

#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace moonai {

namespace {
constexpr float kMaxDensity = 10.0f;
constexpr float kMissingTargetSentinel = 2.0f;

Vec2 wrap_diff(Vec2 diff, float world_width, float world_height) {
  if (std::abs(diff.x) > world_width * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_width : diff.x + world_width;
  }
  if (std::abs(diff.y) > world_height * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_height : diff.y + world_height;
  }
  return diff;
}

void build_sensors(Registry &registry, const FoodStore &food_store,
                   const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;
  const auto agent_count = registry.size();
  const auto &positions = registry.positions();
  const auto &motion = registry.motion();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();
  auto &sensors = registry.sensors();

  for (std::size_t i = 0; i < agent_count; ++i) {
    float *sensor_ptr = sensors.input_ptr(i);
    sensor_ptr[0] = kMissingTargetSentinel;
    sensor_ptr[1] = kMissingTargetSentinel;
    sensor_ptr[2] = kMissingTargetSentinel;
    sensor_ptr[3] = kMissingTargetSentinel;
    sensor_ptr[4] = kMissingTargetSentinel;
    sensor_ptr[5] = kMissingTargetSentinel;
    sensor_ptr[6] = 0.0f;
    sensor_ptr[7] = 0.0f;
    sensor_ptr[8] = 0.0f;
    sensor_ptr[9] = 0.0f;
    sensor_ptr[10] = 0.0f;
    sensor_ptr[11] = 0.0f;

    if (!vitals.alive[i]) {
      continue;
    }

    const Vec2 pos{positions.x[i], positions.y[i]};
    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir{0.0f, 0.0f};
    Vec2 nearest_prey_dir{0.0f, 0.0f};
    Vec2 nearest_food_dir{0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;
    int local_food = 0;

    for (std::size_t other = 0; other < agent_count; ++other) {
      if (other == i || !vitals.alive[other]) {
        continue;
      }

      Vec2 diff =
          wrap_diff({positions.x[other] - pos.x, positions.y[other] - pos.y},
                    world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq || dist_sq <= 0.0f) {
        continue;
      }

      if (identity.type[other] == IdentitySoA::TYPE_PREDATOR) {
        ++local_predators;
        if (dist_sq < nearest_pred_dist_sq) {
          nearest_pred_dist_sq = dist_sq;
          nearest_pred_dir = diff;
        }
      } else {
        ++local_prey;
        if (dist_sq < nearest_prey_dist_sq) {
          nearest_prey_dist_sq = dist_sq;
          nearest_prey_dir = diff;
        }
      }
    }

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active()[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x()[food_idx] - pos.x,
                             food_store.pos_y()[food_idx] - pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq) {
        continue;
      }

      ++local_food;
      if (dist_sq < nearest_food_dist_sq) {
        nearest_food_dist_sq = dist_sq;
        nearest_food_dir = diff;
      }
    }

    if (nearest_pred_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[0] = std::clamp(nearest_pred_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[1] = std::clamp(nearest_pred_dir.y / vision, -1.0f, 1.0f);
    }
    if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[2] = std::clamp(nearest_prey_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[3] = std::clamp(nearest_prey_dir.y / vision, -1.0f, 1.0f);
    }
    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[4] = std::clamp(nearest_food_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[5] = std::clamp(nearest_food_dir.y / vision, -1.0f, 1.0f);
    }

    sensor_ptr[6] = std::clamp(
        vitals.energy[i] / (static_cast<float>(config.initial_energy) * 2.0f),
        0.0f, 1.0f);
    if (motion.speed[i] > 0.0f) {
      sensor_ptr[7] =
          std::clamp(motion.vel_x[i] / motion.speed[i], -1.0f, 1.0f);
      sensor_ptr[8] =
          std::clamp(motion.vel_y[i] / motion.speed[i], -1.0f, 1.0f);
    }
    sensor_ptr[9] = std::clamp(
        static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[10] =
        std::clamp(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[11] =
        std::clamp(static_cast<float>(local_food) / kMaxDensity, 0.0f, 1.0f);
  }
}

void update_vitals(Registry &registry, const SimulationConfig &config) {
  auto &vitals = registry.vitals();

  for (std::size_t i = 0; i < registry.size(); ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    vitals.age[i] += 1;
    if (vitals.reproduction_cooldown[i] > 0) {
      vitals.reproduction_cooldown[i] -= 1;
    }

    vitals.energy[i] -= config.energy_drain_per_step;
    const bool died_of_starvation = vitals.energy[i] <= 0.0f;
    const bool died_of_age =
        config.max_steps > 0 && vitals.age[i] >= config.max_steps;
    if (died_of_starvation || died_of_age) {
      vitals.energy[i] = 0.0f;
      vitals.alive[i] = 0;
    }
  }
}

void process_food(Registry &registry, FoodStore &food_store,
                  const SimulationConfig &config,
                  std::vector<int> &food_consumed_by) {
  std::fill(food_consumed_by.begin(), food_consumed_by.end(), -1);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.food_pickup_range * config.food_pickup_range;
  const auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{positions.x[prey_idx], positions.y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active()[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x()[food_idx] - prey_pos.x,
                             food_store.pos_y()[food_idx] - prey_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_food = static_cast<int>(food_idx);
      }
    }

    if (best_food >= 0) {
      int &owner = food_consumed_by[static_cast<std::size_t>(best_food)];
      if (owner < 0 || static_cast<int>(prey_idx) < owner) {
        owner = static_cast<int>(prey_idx);
      }
    }
  }

  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!food_store.active()[food_idx] || prey_idx < 0 ||
        !vitals.alive[static_cast<std::size_t>(prey_idx)]) {
      continue;
    }

    food_store.active()[food_idx] = 0;
    vitals.energy[static_cast<std::size_t>(prey_idx)] +=
        static_cast<float>(config.energy_gain_from_food);
  }
}

void process_combat(Registry &registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts) {
  std::fill(killed_by.begin(), killed_by.end(), -1);
  std::fill(kill_counts.begin(), kill_counts.end(), 0U);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.attack_range * config.attack_range;
  const auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (std::size_t predator_idx = 0; predator_idx < registry.size();
       ++predator_idx) {
    if (!vitals.alive[predator_idx] ||
        identity.type[predator_idx] != IdentitySoA::TYPE_PREDATOR) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{positions.x[predator_idx],
                            positions.y[predator_idx]};

    for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
      if (!vitals.alive[prey_idx] ||
          identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
        continue;
      }

      Vec2 diff = wrap_diff({positions.x[prey_idx] - predator_pos.x,
                             positions.y[prey_idx] - predator_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_prey = static_cast<int>(prey_idx);
      }
    }

    if (best_prey >= 0) {
      int &killer = killed_by[static_cast<std::size_t>(best_prey)];
      if (killer < 0 || static_cast<int>(predator_idx) < killer) {
        killer = static_cast<int>(predator_idx);
      }
    }
  }

  for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY || killer_idx < 0 ||
        !vitals.alive[static_cast<std::size_t>(killer_idx)]) {
      continue;
    }

    vitals.alive[prey_idx] = 0;
    vitals.energy[static_cast<std::size_t>(killer_idx)] +=
        static_cast<float>(config.energy_gain_from_kill);
    kill_counts[static_cast<std::size_t>(killer_idx)] += 1;
  }
}

void apply_movement(Registry &registry, const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  auto &positions = registry.positions();
  auto &motion = registry.motion();
  const auto &vitals = registry.vitals();
  auto &stats = registry.stats();
  const auto &brain = registry.brain();

  for (std::size_t i = 0; i < registry.size(); ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    float dx = brain.decision_x[i];
    float dy = brain.decision_y[i];
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    motion.vel_x[i] = dx * motion.speed[i];
    motion.vel_y[i] = dy * motion.speed[i];

    const float old_x = positions.x[i];
    const float old_y = positions.y[i];

    positions.x[i] += motion.vel_x[i];
    positions.y[i] += motion.vel_y[i];

    while (positions.x[i] < 0.0f)
      positions.x[i] += world_size;
    while (positions.x[i] >= world_size)
      positions.x[i] -= world_size;
    while (positions.y[i] < 0.0f)
      positions.y[i] += world_size;
    while (positions.y[i] >= world_size)
      positions.y[i] -= world_size;

    Vec2 move = wrap_diff({positions.x[i] - old_x, positions.y[i] - old_y},
                          world_size, world_size);
    stats.distance_traveled[i] += std::sqrt(move.x * move.x + move.y * move.y);
  }
}

void collect_cpu_step_events(Registry &registry, const FoodStore &food_store,
                             const std::vector<uint8_t> &was_alive,
                             const std::vector<uint8_t> &was_food_active,
                             const std::vector<int> &food_consumed_by,
                             const std::vector<int> &killed_by,
                             const std::vector<uint32_t> &kill_counts,
                             std::vector<SimEvent> &events) {
  auto &stats = registry.stats();
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();

  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!was_food_active[food_idx] || food_store.active()[food_idx] ||
        prey_idx < 0 || static_cast<std::size_t>(prey_idx) >= registry.size()) {
      continue;
    }

    stats.food_eaten[static_cast<std::size_t>(prey_idx)] += 1;
    events.push_back(SimEvent{
        SimEvent::Food, Entity{static_cast<uint32_t>(prey_idx)}, INVALID_ENTITY,
        Vec2{positions.x[static_cast<std::size_t>(prey_idx)],
             positions.y[static_cast<std::size_t>(prey_idx)]}});
  }

  for (std::size_t agent_idx = 0; agent_idx < registry.size(); ++agent_idx) {
    if (kill_counts[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(kill_counts[agent_idx]);
    }

    if (killed_by[agent_idx] >= 0 &&
        static_cast<std::size_t>(killed_by[agent_idx]) < registry.size()) {
      events.push_back(SimEvent{
          SimEvent::Kill, Entity{static_cast<uint32_t>(killed_by[agent_idx])},
          Entity{static_cast<uint32_t>(agent_idx)},
          Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
      continue;
    }

    if (was_alive[agent_idx] && !vitals.alive[agent_idx]) {
      const Entity entity{static_cast<uint32_t>(agent_idx)};
      events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}
} // namespace

SimulationManager::SimulationManager(const SimulationConfig &config)
    : config_(config),
      rng_(config.seed != 0
               ? config.seed
               : static_cast<std::uint64_t>(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count())),
      grid_(config.grid_size, config.grid_size,
            std::max(config.mate_range, 1.0f)) {}

void SimulationManager::initialize() {
  initialize(true);
}

void SimulationManager::initialize(bool log_initialization) {
  current_step_ = 0;
  food_store_.initialize(config_, rng_);

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})",
                 config_.food_count, rng_.seed());
  }
}

void SimulationManager::collect_gpu_step_events(
    Registry &registry, const std::vector<uint8_t> &was_alive,
    const std::vector<uint8_t> &was_food_active,
    std::vector<SimEvent> &events) {
  auto &stats = registry.stats();
  const auto &positions = registry.positions();

  for (std::size_t food_idx = 0; food_idx < food_store_.size(); ++food_idx) {
    const int prey_idx = gpu_batch_->buffer().host_food_consumed_by()[food_idx];
    if (!was_food_active[food_idx] || food_store_.active()[food_idx] ||
        prey_idx < 0 || static_cast<std::size_t>(prey_idx) >= registry.size()) {
      continue;
    }

    const Entity prey{static_cast<uint32_t>(prey_idx)};
    stats.food_eaten[prey_idx] += 1;
    events.push_back(
        SimEvent{SimEvent::Food, prey, INVALID_ENTITY,
                 Vec2{positions.x[prey_idx], positions.y[prey_idx]}});
  }

  for (std::size_t agent_idx = 0; agent_idx < registry.size(); ++agent_idx) {
    if (gpu_batch_->buffer().host_agent_kill_counts()[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(
          gpu_batch_->buffer().host_agent_kill_counts()[agent_idx]);
    }

    const int killer_idx =
        gpu_batch_->buffer().host_agent_killed_by()[agent_idx];
    if (killer_idx >= 0 &&
        static_cast<std::size_t>(killer_idx) < registry.size()) {
      events.push_back(
          SimEvent{SimEvent::Kill, Entity{static_cast<uint32_t>(killer_idx)},
                   Entity{static_cast<uint32_t>(agent_idx)},
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
      continue;
    }

    if (was_alive[agent_idx] && registry.vitals().alive[agent_idx] == 0) {
      const Entity entity{static_cast<uint32_t>(agent_idx)};
      events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}

void SimulationManager::compact_registry(Registry &registry,
                                         EvolutionManager &evolution) {
  const auto result = registry.compact_dead();
  for (const auto &[from, to] : result.moved) {
    evolution.on_entity_moved(from, to);
  }
  for (Entity removed : result.removed) {
    evolution.on_entity_destroyed(removed);
  }
}

void SimulationManager::refresh_world_state_after_step(Registry &registry) {
  {
    MOONAI_PROFILE_SCOPE("food_respawn");
    food_store_.respawn_step(config_, current_step_, rng_.seed());
  }
  rebuild_spatial_grid(registry);
  count_alive(registry);
  ++current_step_;
}

SimulationManager::SimulationStepResult
SimulationManager::step(Registry &registry, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");
  SimulationStepResult result;

  const std::vector<uint8_t> was_alive = registry.vitals().alive;
  const std::vector<uint8_t> was_food_active = food_store_.active();
  std::vector<int> food_consumed_by(food_store_.size(), -1);
  std::vector<int> killed_by(registry.size(), -1);
  std::vector<uint32_t> kill_counts(registry.size(), 0U);

  build_sensors(registry, food_store_, config_);
  evolution.compute_actions(registry);
  update_vitals(registry, config_);
  process_food(registry, food_store_, config_, food_consumed_by);
  process_combat(registry, config_, killed_by, kill_counts);
  apply_movement(registry, config_);

  collect_cpu_step_events(registry, food_store_, was_alive, was_food_active,
                          food_consumed_by, killed_by, kill_counts,
                          result.events);
  compact_registry(registry, evolution);
  refresh_world_state_after_step(registry);
  result.reproduction_pairs = find_reproduction_pairs(registry);
  return result;
}

void SimulationManager::reset() {
  initialize(false);
}

void SimulationManager::rebuild_spatial_grid(const Registry &registry) {
  MOONAI_PROFILE_SCOPE("rebuild_spatial_grid");
  grid_.clear();

  const auto &positions = registry.positions();

  for (std::size_t idx = 0; idx < registry.size(); ++idx) {
    const Entity entity{static_cast<uint32_t>(idx)};
    grid_.insert(entity, Vec2{positions.x[idx], positions.y[idx]});
  }
}

std::vector<SimulationManager::ReproductionPair>
SimulationManager::find_reproduction_pairs(const Registry &registry) const {
  std::vector<ReproductionPair> pairs;
  std::unordered_set<Entity, EntityHash> used;

  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (std::size_t idx = 0; idx < registry.size(); ++idx) {
    const Entity entity{static_cast<uint32_t>(idx)};
    if (vitals.energy[idx] < config_.reproduction_energy_threshold ||
        vitals.age[idx] < config_.min_reproductive_age_steps ||
        vitals.reproduction_cooldown[idx] > 0 ||
        used.find(entity) != used.end()) {
      continue;
    }

    Vec2 pos{positions.x[idx], positions.y[idx]};
    std::vector<Entity> nearby = grid_.query_radius(pos, config_.mate_range);

    Entity best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    for (Entity mate_id : nearby) {
      if (mate_id == entity || used.find(mate_id) != used.end()) {
        continue;
      }

      const std::size_t mate_idx = registry.index_of(mate_id);
      if (identity.type[mate_idx] != identity.type[idx] ||
          vitals.energy[mate_idx] < config_.reproduction_energy_threshold ||
          vitals.age[mate_idx] < config_.min_reproductive_age_steps ||
          vitals.reproduction_cooldown[mate_idx] > 0) {
        continue;
      }

      Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      Vec2 diff = mate_pos - pos;
      float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    }

    if (best_mate != INVALID_ENTITY) {
      const std::size_t mate_idx = registry.index_of(best_mate);
      Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      Vec2 spawn_pos{(pos.x + mate_pos.x) * 0.5f, (pos.y + mate_pos.y) * 0.5f};
      pairs.push_back({entity, best_mate, spawn_pos});
      used.insert(entity);
      used.insert(best_mate);
    }
  }

  return pairs;
}

void SimulationManager::count_alive(const Registry &registry) {
  MOONAI_PROFILE_SCOPE("count_alive");
  alive_predators_ = 0;
  alive_prey_ = 0;

  const auto &identity = registry.identity();
  for (std::size_t idx = 0; idx < registry.size(); ++idx) {
    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      ++alive_predators_;
    } else {
      ++alive_prey_;
    }
  }
}

void SimulationManager::refresh_state(Registry &registry) {
  rebuild_spatial_grid(registry);
  count_alive(registry);
}

SimulationManager::~SimulationManager() = default;

void SimulationManager::ensure_gpu_capacity(std::size_t agent_count,
                                            std::size_t food_count) {
  if (!gpu_enabled_) {
    return;
  }

  const bool needs_batch = !gpu_batch_;
  const bool needs_resize =
      gpu_batch_ && (agent_count > gpu_batch_->agent_capacity() ||
                     food_count > gpu_batch_->food_capacity());
  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_agent_capacity =
      gpu_batch_ ? gpu_batch_->agent_capacity() : 0;
  const std::size_t current_food_capacity =
      gpu_batch_ ? gpu_batch_->food_capacity() : 0;
  const std::size_t new_agent_capacity = std::max(
      agent_count,
      current_agent_capacity == 0 ? agent_count : current_agent_capacity * 2);
  const std::size_t new_food_capacity =
      std::max(food_count,
               current_food_capacity == 0 ? food_count : current_food_capacity);

  gpu_batch_ =
      std::make_unique<gpu::GpuBatch>(new_agent_capacity, new_food_capacity);
  spdlog::info(
      "GPU batch processing enabled with capacities {} agents / {} food",
      new_agent_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable) {
    ensure_gpu_capacity(
        static_cast<std::size_t>(config_.predator_count + config_.prey_count),
        static_cast<std::size_t>(config_.food_count));
  } else if (!enable) {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

SimulationManager::SimulationStepResult
SimulationManager::step_gpu(Registry &registry, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");
  SimulationStepResult result;

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    return step(registry, evolution);
  }

  const std::size_t agent_count = registry.size();
  const std::size_t food_count = food_store_.size();
  if (agent_count == 0) {
    return result;
  }

  std::vector<uint8_t> was_alive;
  std::vector<uint8_t> was_food_active;
  {
    MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");
    ensure_gpu_capacity(agent_count, food_count);
  }
  {
    MOONAI_PROFILE_SCOPE("gpu_pack_state");
    was_alive = registry.vitals().alive;
    was_food_active = food_store_.active();

    auto &buffer = gpu_batch_->buffer();
    std::memcpy(buffer.host_agent_positions_x(), registry.positions().x.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_positions_y(), registry.positions().y.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_x(),
                registry.motion().vel_x.data(), agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_y(),
                registry.motion().vel_y.data(), agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_speed(), registry.motion().speed.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_energy(), registry.vitals().energy.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_age(), registry.vitals().age.data(),
                agent_count * sizeof(int));
    for (std::size_t i = 0; i < agent_count; ++i) {
      buffer.host_agent_alive()[i] = registry.vitals().alive[i];
      buffer.host_agent_types()[i] = registry.identity().type[i];
    }
    std::memcpy(buffer.host_agent_reproduction_cooldown(),
                registry.vitals().reproduction_cooldown.data(),
                agent_count * sizeof(int));
    std::memcpy(buffer.host_agent_distance_traveled(),
                registry.stats().distance_traveled.data(),
                agent_count * sizeof(float));

    std::memcpy(buffer.host_food_positions_x(), food_store_.pos_x().data(),
                food_count * sizeof(float));
    std::memcpy(buffer.host_food_positions_y(), food_store_.pos_y().data(),
                food_count * sizeof(float));
    for (std::size_t i = 0; i < food_count; ++i) {
      buffer.host_food_active()[i] = food_store_.active()[i];
      buffer.host_food_consumed_by()[i] = -1;
    }
  }

  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.vision_range = config_.vision_range;
  params.max_energy = static_cast<float>(config_.initial_energy);
  params.max_age = config_.max_steps;
  params.food_pickup_range = config_.food_pickup_range;
  params.attack_range = config_.attack_range;
  params.energy_gain_from_food =
      static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill =
      static_cast<float>(config_.energy_gain_from_kill);

  gpu_batch_->upload_async(agent_count, food_count);

  gpu_batch_->launch_build_sensors_async(params, agent_count, food_count);

  if (!evolution.launch_gpu_neural(*gpu_batch_, agent_count)) {
    MOONAI_PROFILE_SCOPE("cpu_fallback");
    spdlog::error(
        "GPU neural inference failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    return step(registry, evolution);
  }

  gpu_batch_->launch_post_inference_async(params, agent_count, food_count);

  gpu_batch_->download_async(agent_count, food_count);

  {
    MOONAI_PROFILE_SCOPE("gpu_synchronize");
    gpu_batch_->synchronize();
  }

  if (!gpu_batch_->ok()) {
    MOONAI_PROFILE_SCOPE("cpu_fallback");
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    return step(registry, evolution);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_apply_results");
    {
      MOONAI_PROFILE_SCOPE("apply_step_state");
      auto &buffer = gpu_batch_->buffer();
      std::memcpy(registry.positions().x.data(),
                  buffer.host_agent_positions_x(), agent_count * sizeof(float));
      std::memcpy(registry.positions().y.data(),
                  buffer.host_agent_positions_y(), agent_count * sizeof(float));
      std::memcpy(registry.motion().vel_x.data(),
                  buffer.host_agent_velocities_x(),
                  agent_count * sizeof(float));
      std::memcpy(registry.motion().vel_y.data(),
                  buffer.host_agent_velocities_y(),
                  agent_count * sizeof(float));
      std::memcpy(registry.vitals().energy.data(), buffer.host_agent_energy(),
                  agent_count * sizeof(float));
      std::memcpy(registry.vitals().age.data(), buffer.host_agent_age(),
                  agent_count * sizeof(int));
      std::memcpy(registry.vitals().reproduction_cooldown.data(),
                  buffer.host_agent_reproduction_cooldown(),
                  agent_count * sizeof(int));
      std::memcpy(registry.stats().distance_traveled.data(),
                  buffer.host_agent_distance_traveled(),
                  agent_count * sizeof(float));
      std::memcpy(registry.sensors().inputs.data(),
                  buffer.host_agent_sensor_inputs(),
                  agent_count * SensorSoA::INPUT_COUNT * sizeof(float));
      for (std::size_t i = 0; i < agent_count; ++i) {
        registry.vitals().alive[i] =
            static_cast<uint8_t>(buffer.host_agent_alive()[i]);
        registry.brain().decision_x[i] =
            buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT];
        registry.brain().decision_y[i] =
            buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT + 1];
      }
      for (std::size_t i = 0; i < food_count; ++i) {
        food_store_.pos_x()[i] = buffer.host_food_positions_x()[i];
        food_store_.pos_y()[i] = buffer.host_food_positions_y()[i];
        food_store_.active()[i] =
            static_cast<uint8_t>(buffer.host_food_active()[i]);
      }
    }
    {
      MOONAI_PROFILE_SCOPE("collect_step_events");
      collect_gpu_step_events(registry, was_alive, was_food_active,
                              result.events);
    }
  }
  {
    MOONAI_PROFILE_SCOPE("compact_registry");
    compact_registry(registry, evolution);
  }
  {
    MOONAI_PROFILE_SCOPE("refresh_world_state");
    refresh_world_state_after_step(registry);
  }
  {
    MOONAI_PROFILE_SCOPE("find_reproduction_pairs");
    result.reproduction_pairs = find_reproduction_pairs(registry);
  }
  return result;
}

} // namespace moonai
