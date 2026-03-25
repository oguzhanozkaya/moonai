#include "simulation/environment.hpp"

#include "core/deterministic_respawn.hpp"

#include <algorithm>
#include <cmath>

namespace moonai {

Environment::Environment(const SimulationConfig &config)
    : width_(config.grid_size), height_(config.grid_size),
      boundary_mode_(config.boundary_mode), max_food_(config.food_count) {}

Vec2 Environment::apply_boundary(Vec2 pos) const {
  if (boundary_mode_ == BoundaryMode::Wrap) {
    float w = static_cast<float>(width_);
    float h = static_cast<float>(height_);
    auto wrap = [](float v, float size) -> float {
      v = std::fmod(v, size);
      if (v < 0.0f)
        v += size;
      return v;
    };
    pos.x = wrap(pos.x, w);
    pos.y = wrap(pos.y, h);
  } else {
    pos.x = std::clamp(pos.x, 0.0f, static_cast<float>(width_));
    pos.y = std::clamp(pos.y, 0.0f, static_cast<float>(height_));
  }
  return pos;
}

void Environment::initialize_food(Random &rng, int count) {
  food_.clear();
  food_.reserve(count);
  for (int i = 0; i < count; ++i) {
    Food f;
    f.position = {rng.next_float(0, static_cast<float>(width_)),
                  rng.next_float(0, static_cast<float>(height_))};
    f.active = true;
    food_.push_back(f);
  }
}

void Environment::step_food(Random &rng, float respawn_rate) {
  std::vector<AgentId> ignored;
  step_food(rng, respawn_rate, ignored);
}

void Environment::step_food(Random &rng, float respawn_rate,
                            std::vector<AgentId> &respawned_ids) {
  respawned_ids.clear();
  for (auto &f : food_) {
    if (!f.active && rng.next_bool(respawn_rate)) {
      f.position = {rng.next_float(0, static_cast<float>(width_)),
                    rng.next_float(0, static_cast<float>(height_))};
      f.active = true;
      respawned_ids.push_back(static_cast<AgentId>(&f - food_.data()));
    }
  }
}

void Environment::step_food_deterministic(std::uint64_t seed, int step_index,
                                          float respawn_rate,
                                          std::vector<AgentId> &respawned_ids) {
  respawned_ids.clear();
  for (std::uint32_t i = 0; i < food_.size(); ++i) {
    auto &f = food_[i];
    if (f.active ||
        !respawn::should_respawn(seed, step_index, i, respawn_rate)) {
      continue;
    }
    f.position = {
        respawn::respawn_x(seed, step_index, i, static_cast<float>(width_)),
        respawn::respawn_y(seed, step_index, i, static_cast<float>(height_))};
    f.active = true;
    respawned_ids.push_back(i);
  }
}

bool Environment::try_eat_food(Vec2 position, float range) {
  const float range_sq = range * range;
  for (auto &f : food_) {
    if (!f.active) {
      continue;
    }
    const float dx = position.x - f.position.x;
    const float dy = position.y - f.position.y;
    if ((dx * dx + dy * dy) <= range_sq) {
      f.active = false;
      return true;
    }
  }
  return false;
}

bool Environment::try_eat_food(Vec2 position, float range,
                               const std::vector<AgentId> &candidate_ids,
                               AgentId *eaten_id) {
  const float range_sq = range * range;
  for (AgentId id : candidate_ids) {
    if (id >= food_.size()) {
      continue;
    }
    auto &f = food_[id];
    if (!f.active) {
      continue;
    }
    const float dx = position.x - f.position.x;
    const float dy = position.y - f.position.y;
    if ((dx * dx + dy * dy) <= range_sq) {
      f.active = false;
      if (eaten_id != nullptr) {
        *eaten_id = id;
      }
      return true;
    }
  }
  return false;
}

} // namespace moonai
