#include "simulation/system.hpp"

namespace moonai {

void SystemScheduler::add_system(std::unique_ptr<System> system) {
  systems_.push_back(std::move(system));
}

void SystemScheduler::update(Registry &registry, float dt) {
  for (auto &system : systems_) {
    system->update(registry, dt);
  }
}

} // namespace moonai