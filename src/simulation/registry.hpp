#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace moonai {

struct RegistryCompactionResult {
  std::vector<std::pair<uint32_t, uint32_t>> moved;
  std::vector<uint32_t> removed;
};

struct Food {
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index,
                    std::uint64_t seed);

  std::size_t size() const {
    return pos_x.size();
  }

  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<uint8_t> active;
};

struct AgentRegistry {
  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<float> vel_x;
  std::vector<float> vel_y;
  std::vector<float> energy;
  std::vector<int> age;
  std::vector<uint8_t> alive;
  std::vector<uint32_t> species_id;
  std::vector<uint32_t> entity_id;
  std::vector<int> consumption;

  uint32_t create();
  bool valid(uint32_t entity) const;
  std::size_t size() const;
  void clear();
  RegistryCompactionResult compact_dead();
  uint32_t find_by_agent_id(uint32_t agent_id) const;

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();
};

} // namespace moonai