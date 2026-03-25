#pragma once
#include "simulation/entity.hpp"
#include <cstdint>
#include <limits>
#include <vector>

namespace moonai {

// Sparse set: maps Entity index -> dense component index
// Allows O(1) entity lookup with stable handles
class SparseSet {
public:
  // Insert entity, return its dense index
  size_t insert(Entity e) {
    // Ensure sparse array is large enough
    if (e.index >= sparse_.size()) {
      sparse_.resize(e.index + 1, invalid_index);
    }

    // Check if already exists
    if (sparse_[e.index] != invalid_index) {
      return sparse_[e.index];
    }

    // Add to dense array
    size_t dense_idx = dense_.size();
    dense_.push_back(e);
    sparse_[e.index] = static_cast<uint32_t>(dense_idx);

    return dense_idx;
  }

  // Remove entity (doesn't affect other entities' indices)
  void remove(Entity e) {
    if (!contains(e)) {
      return;
    }

    size_t dense_idx = sparse_[e.index];
    Entity last_entity = dense_.back();

    // Move last element to fill the gap (swap-and-pop)
    dense_[dense_idx] = last_entity;
    sparse_[last_entity.index] = static_cast<uint32_t>(dense_idx);

    dense_.pop_back();
    sparse_[e.index] = invalid_index;
  }

  // Check if entity exists
  bool contains(Entity e) const {
    if (e.index >= sparse_.size()) {
      return false;
    }
    if (sparse_[e.index] == invalid_index) {
      return false;
    }
    // Verify generation matches (handle is valid)
    size_t dense_idx = sparse_[e.index];
    return dense_[dense_idx].generation == e.generation;
  }

  // Get dense index for entity (or invalid_index if not found)
  size_t get_index(Entity e) const {
    if (!contains(e)) {
      return invalid_index;
    }
    return sparse_[e.index];
  }

  // Get entity at dense index
  Entity get_entity(size_t dense_index) const {
    if (dense_index >= dense_.size()) {
      return INVALID_ENTITY;
    }
    return dense_[dense_index];
  }

  size_t size() const {
    return dense_.size();
  }
  bool empty() const {
    return dense_.empty();
  }

  // Dense array of active entities (for iteration)
  const std::vector<Entity> &dense() const {
    return dense_;
  }

private:
  static constexpr uint32_t invalid_index =
      std::numeric_limits<uint32_t>::max();

  std::vector<uint32_t> sparse_;
  std::vector<Entity> dense_;
};

} // namespace moonai