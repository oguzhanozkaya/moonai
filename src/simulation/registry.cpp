#include "simulation/registry.hpp"

#include <algorithm>

namespace moonai {

Entity Registry::create() {
  const Entity entity{static_cast<uint32_t>(size())};
  resize(size() + 1);
  return entity;
}

void Registry::destroy(Entity e) {
  if (valid(e)) {
    vitals_.alive[e.index] = 0;
  }
}

bool Registry::valid(Entity e) const {
  return e != INVALID_ENTITY && e.index < size();
}

void Registry::clear() {
  resize(0);
  next_agent_id_ = 1;
}

Registry::CompactionResult Registry::compact_dead() {
  CompactionResult result;

  std::size_t i = 0;
  while (i < size()) {
    if (vitals_.alive[i] != 0) {
      ++i;
      continue;
    }

    const Entity removed{static_cast<uint32_t>(i)};
    const std::size_t last = size() - 1;
    if (i != last) {
      const Entity moved_from{static_cast<uint32_t>(last)};
      const Entity moved_to{static_cast<uint32_t>(i)};
      swap_entities(i, last);
      result.moved.push_back({moved_from, moved_to});
    }

    result.removed.push_back(Entity{static_cast<uint32_t>(last)});
    pop_back();
  }

  return result;
}

Entity Registry::find_by_agent_id(uint32_t agent_id) const {
  const auto it = std::find(identity_.entity_id.begin(),
                            identity_.entity_id.end(), agent_id);
  if (it == identity_.entity_id.end()) {
    return INVALID_ENTITY;
  }
  return Entity{
      static_cast<uint32_t>(std::distance(identity_.entity_id.begin(), it))};
}

void Registry::resize(std::size_t new_size) {
  positions_.resize(new_size);
  motion_.resize(new_size);
  vitals_.resize(new_size);
  identity_.resize(new_size);
  sensors_.resize(new_size);
  stats_.resize(new_size);
  brain_.resize(new_size);
}

void Registry::swap_entities(std::size_t a, std::size_t b) {
  using std::swap;

  swap(positions_.x[a], positions_.x[b]);
  swap(positions_.y[a], positions_.y[b]);
  swap(motion_.vel_x[a], motion_.vel_x[b]);
  swap(motion_.vel_y[a], motion_.vel_y[b]);
  swap(motion_.speed[a], motion_.speed[b]);
  swap(vitals_.energy[a], vitals_.energy[b]);
  swap(vitals_.age[a], vitals_.age[b]);
  swap(vitals_.alive[a], vitals_.alive[b]);
  swap(vitals_.reproduction_cooldown[a], vitals_.reproduction_cooldown[b]);
  swap(identity_.type[a], identity_.type[b]);
  swap(identity_.species_id[a], identity_.species_id[b]);
  swap(identity_.entity_id[a], identity_.entity_id[b]);
  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    swap(sensors_.inputs[a * SensorSoA::INPUT_COUNT + i],
         sensors_.inputs[b * SensorSoA::INPUT_COUNT + i]);
  }
  swap(stats_.kills[a], stats_.kills[b]);
  swap(stats_.food_eaten[a], stats_.food_eaten[b]);
  swap(stats_.distance_traveled[a], stats_.distance_traveled[b]);
  swap(stats_.offspring_count[a], stats_.offspring_count[b]);
  swap(brain_.decision_x[a], brain_.decision_x[b]);
  swap(brain_.decision_y[a], brain_.decision_y[b]);
}

void Registry::pop_back() {
  const std::size_t new_size = size() - 1;
  positions_.x.pop_back();
  positions_.y.pop_back();
  motion_.vel_x.pop_back();
  motion_.vel_y.pop_back();
  motion_.speed.pop_back();
  vitals_.energy.pop_back();
  vitals_.age.pop_back();
  vitals_.alive.pop_back();
  vitals_.reproduction_cooldown.pop_back();
  identity_.type.pop_back();
  identity_.species_id.pop_back();
  identity_.entity_id.pop_back();
  sensors_.inputs.resize(new_size * SensorSoA::INPUT_COUNT);
  stats_.kills.pop_back();
  stats_.food_eaten.pop_back();
  stats_.distance_traveled.pop_back();
  stats_.offspring_count.pop_back();
  brain_.decision_x.pop_back();
  brain_.decision_y.pop_back();
}

} // namespace moonai
