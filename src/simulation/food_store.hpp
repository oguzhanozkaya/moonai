#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "simulation/components.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct FoodStore {
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index,
                    std::uint64_t seed);

  void resize(std::size_t n) {
    positions.resize(n);
    active.resize(n);
  }

  PositionSoA positions;
  std::vector<uint8_t> active;

  std::size_t size() const {
    return positions.size();
  }
};

} // namespace moonai
