#pragma once

#include "simulation/components.hpp"
#include "simulation/entity.hpp"

#include <cstdint>
#include <vector>

namespace moonai {

class Registry {
public:
  struct CompactionResult {
    std::vector<std::pair<Entity, Entity>> moved;
    std::vector<Entity> removed;
  };

  Entity create();
  void destroy(Entity e);
  bool valid(Entity e) const;

  size_t index_of(Entity e) const {
    return e.index;
  }

  Entity entity_at(std::size_t index) const {
    return Entity{static_cast<uint32_t>(index)};
  }

  size_t size() const {
    return positions_.size();
  }

  bool empty() const {
    return size() == 0;
  }

  PositionSoA &positions() {
    return positions_;
  }
  const PositionSoA &positions() const {
    return positions_;
  }
  MotionSoA &motion() {
    return motion_;
  }
  const MotionSoA &motion() const {
    return motion_;
  }
  VitalsSoA &vitals() {
    return vitals_;
  }
  const VitalsSoA &vitals() const {
    return vitals_;
  }
  IdentitySoA &identity() {
    return identity_;
  }
  const IdentitySoA &identity() const {
    return identity_;
  }
  SensorSoA &sensors() {
    return sensors_;
  }
  const SensorSoA &sensors() const {
    return sensors_;
  }
  StatsSoA &stats() {
    return stats_;
  }
  const StatsSoA &stats() const {
    return stats_;
  }
  BrainSoA &brain() {
    return brain_;
  }
  const BrainSoA &brain() const {
    return brain_;
  }

  float &pos_x(Entity e) {
    return positions_.x[index_of(e)];
  }
  float &pos_y(Entity e) {
    return positions_.y[index_of(e)];
  }
  float &energy(Entity e) {
    return vitals_.energy[index_of(e)];
  }

  void clear();
  CompactionResult compact_dead();
  Entity find_by_agent_id(uint32_t agent_id) const;
  uint32_t next_agent_id() {
    return next_agent_id_++;
  }

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();

  PositionSoA positions_;
  MotionSoA motion_;
  VitalsSoA vitals_;
  IdentitySoA identity_;
  SensorSoA sensors_;
  StatsSoA stats_;
  BrainSoA brain_;
  uint32_t next_agent_id_ = 1;
};

} // namespace moonai
