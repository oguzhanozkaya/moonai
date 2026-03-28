#pragma once

#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "simulation/entity.hpp"

#include <deque>
#include <vector>

namespace moonai {

class Species {
public:
  explicit Species(const Genome &representative);

  struct Member {
    Entity entity = INVALID_ENTITY;
    float fitness = 0.0f;
    int complexity = 0;
  };

  bool is_compatible(const Genome &genome, float threshold, float c1, float c2,
                     float c3) const;

  void add_member(Entity entity, const Genome &genome);
  void clear_members();
  void refresh_summary();

  const Genome &representative() const {
    return representative_;
  }
  const std::vector<Member> &members() const {
    return members_;
  }
  float average_fitness() const {
    return average_fitness_;
  }
  float best_fitness_ever() const {
    return best_fitness_ever_;
  }
  float average_complexity() const {
    return average_complexity_;
  }
  int id() const {
    return id_;
  }

  static int next_species_id();

private:
  int id_;
  Genome representative_;
  std::vector<Member> members_;
  float average_fitness_ = 0.0f;
  float best_fitness_ever_ = 0.0f;
  float average_complexity_ = 0.0f;

  static int species_id_counter_;
};

} // namespace moonai
