#include "evolution/species.hpp"

#include <algorithm>
#include <numeric>

namespace moonai {

int Species::species_id_counter_ = 0;

int Species::next_species_id() { return species_id_counter_++; }

Species::Species(const Genome &representative)
    : id_(next_species_id()), representative_(representative) {}

bool Species::is_compatible(const Genome &genome, float threshold, float c1,
                            float c2, float c3) const {
  return Genome::compatibility_distance(representative_, genome, c1, c2, c3) <
         threshold;
}

void Species::add_member(Genome *genome) { members_.push_back(genome); }

void Species::clear_members() { members_.clear(); }

void Species::adjust_fitness() {
  if (members_.empty()) {
    average_fitness_ = 0.0f;
    total_adjusted_fitness_ = 0.0f;
    return;
  }

  float size = static_cast<float>(members_.size());

  // Explicit fitness sharing: divide each member's fitness by species size
  for (auto *member : members_) {
    member->set_adjusted_fitness(member->fitness() / size);
  }

  // Compute total adjusted fitness for proportional reproduction
  total_adjusted_fitness_ = std::accumulate(
      members_.begin(), members_.end(), 0.0f,
      [](float sum, const Genome *g) { return sum + g->adjusted_fitness(); });

  // Average raw fitness for reporting
  float total_raw = std::accumulate(
      members_.begin(), members_.end(), 0.0f,
      [](float sum, const Genome *g) { return sum + g->fitness(); });
  average_fitness_ = total_raw / size;
}

void Species::sort_by_fitness() {
  std::sort(members_.begin(), members_.end(),
            [](const Genome *a, const Genome *b) {
              return a->fitness() > b->fitness();
            });
}

void Species::update_representative() {
  if (!members_.empty()) {
    // Use the best member as representative
    sort_by_fitness();
    representative_ = *members_[0];
  }
}

void Species::update_best_fitness() {
  if (members_.empty())
    return;

  float current_best = 0.0f;
  for (const auto *member : members_) {
    if (member->fitness() > current_best) {
      current_best = member->fitness();
    }
  }

  if (current_best > best_fitness_ever_) {
    best_fitness_ever_ = current_best;
    generations_without_improvement_ = 0;
  } else {
    ++generations_without_improvement_;
  }

  fitness_history_.push_back(average_fitness_);
}

bool Species::is_stagnant(int stagnation_limit) const {
  return generations_without_improvement_ >= stagnation_limit;
}

} // namespace moonai
