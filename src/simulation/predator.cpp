#include "simulation/predator.hpp"

namespace moonai {

Predator::Predator(AgentId id, Vec2 position, float speed, float vision_range,
                   float energy, float attack_range)
    : Agent(id, AgentType::Predator, position, speed, vision_range, energy),
      attack_range_(attack_range) {}

void Predator::update(float /*dt*/) {
  // Movement is applied externally via apply_movement() after NN decision
  increment_age();
}

} // namespace moonai
