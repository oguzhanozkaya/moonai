#pragma once

#include "simulation/agent.hpp"

namespace moonai {

class Predator : public Agent {
public:
  Predator(AgentId id, Vec2 position, float speed, float vision_range,
           float energy, float attack_range);

  void update(float dt) override;

  float attack_range() const { return attack_range_; }

private:
  float attack_range_;
};

} // namespace moonai
