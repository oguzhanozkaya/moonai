#pragma once

#include <cstdint>
#include <functional>

namespace moonai {

struct Entity {
  uint32_t index = 0;

  constexpr Entity() = default;
  constexpr Entity(uint32_t value) : index(value) {}
  constexpr Entity(uint32_t value, uint32_t) : index(value) {}

  bool operator==(const Entity &other) const {
    return index == other.index;
  }

  bool operator!=(const Entity &other) const {
    return index != other.index;
  }

  bool operator<(const Entity &other) const {
    return index < other.index;
  }

  bool valid() const {
    return index != UINT32_MAX;
  }
};

constexpr Entity INVALID_ENTITY{UINT32_MAX};

struct EntityHash {
  size_t operator()(const Entity &e) const {
    return std::hash<uint32_t>{}(e.index);
  }
};

} // namespace moonai
