#include "evolution/species.hpp"
#include "simulation/entity.hpp"

#include <algorithm>
#include <numeric>

namespace moonai {

int Species::species_id_counter_ = 0;

int Species::next_species_id() {
  return species_id_counter_++;
}

Species::Species(const Genome &representative)
    : id_(next_species_id()), representative_(representative) {}

bool Species::is_compatible(const Genome &genome, float threshold, float c1,
                            float c2, float c3) const {
  return Genome::compatibility_distance(representative_, genome, c1, c2, c3) <
         threshold;
}

void Species::add_member(Entity entity, const Genome &genome) {
  members_.push_back(
      {entity, genome.fitness(), static_cast<int>(genome.complexity())});
}

void Species::clear_members() {
  members_.clear();
  average_fitness_ = 0.0f;
  average_complexity_ = 0.0f;
}

void Species::refresh_summary() {
  if (members_.empty()) {
    average_fitness_ = 0.0f;
    average_complexity_ = 0.0f;
    return;
  }

  const float size = static_cast<float>(members_.size());
  const float total_fitness = std::accumulate(
      members_.begin(), members_.end(), 0.0f,
      [](float sum, const Member &member) { return sum + member.fitness; });
  const float total_complexity =
      std::accumulate(members_.begin(), members_.end(), 0.0f,
                      [](float sum, const Member &member) {
                        return sum + static_cast<float>(member.complexity);
                      });
  average_fitness_ = total_fitness / size;
  average_complexity_ = total_complexity / size;

  const auto best = std::max_element(members_.begin(), members_.end(),
                                     [](const Member &lhs, const Member &rhs) {
                                       return lhs.fitness < rhs.fitness;
                                     });
  if (best != members_.end()) {
    best_fitness_ever_ = std::max(best_fitness_ever_, best->fitness);
  }
  fitness_history_.push_back(average_fitness_);
}

void Species::update_representative(const Genome &genome) {
  representative_ = genome;
}

} // namespace moonai
