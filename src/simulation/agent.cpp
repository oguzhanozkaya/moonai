#include "simulation/agent.hpp"

namespace moonai {

Agent::Agent(AgentId id, AgentType type, Vec2 position, float speed,
             float vision_range, float energy)
    : id_(id), type_(type), position_(position), speed_(speed),
      vision_range_(vision_range), energy_(energy) {}

void Agent::apply_movement(Vec2 direction, float dt) {
  Vec2 norm = direction.normalized();
  velocity_ = norm * speed_;
  position_ = position_ + velocity_ * dt;
  distance_traveled_ += speed_ * dt;
}

} // namespace moonai
