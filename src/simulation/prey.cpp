#include "simulation/prey.hpp"

namespace moonai {

Prey::Prey(AgentId id, Vec2 position, float speed, float vision_range, float energy)
    : Agent(id, AgentType::Prey, position, speed, vision_range, energy) {
}

void Prey::update(float /*dt*/) {
    // Movement is applied externally via apply_movement() after NN decision
    increment_age();
}

} // namespace moonai
