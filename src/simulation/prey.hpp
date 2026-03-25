#pragma once

#include "simulation/agent.hpp"

namespace moonai {

class Prey : public Agent {
public:
  Prey(AgentId id, Vec2 position, float speed, float vision_range,
       float energy);

  void update(float dt) override;
};

} // namespace moonai
