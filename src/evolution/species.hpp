#pragma once

#include "evolution/genome.hpp"

#include <vector>
#include <deque>

namespace moonai {

class Species {
public:
    explicit Species(const Genome& representative);

    bool is_compatible(const Genome& genome, float threshold,
                       float c1, float c2, float c3) const;

    void add_member(Genome* genome);
    void clear_members();

    // Compute adjusted fitness (explicit fitness sharing)
    void adjust_fitness();
    void sort_by_fitness();

    // Update representative to a random member for next generation
    void update_representative();

    // Stagnation tracking
    void update_best_fitness();
    bool is_stagnant(int stagnation_limit) const;

    const Genome& representative() const { return representative_; }
    const std::vector<Genome*>& members() const { return members_; }
    float average_fitness() const { return average_fitness_; }
    float total_adjusted_fitness() const { return total_adjusted_fitness_; }
    float best_fitness_ever() const { return best_fitness_ever_; }
    int generations_without_improvement() const { return generations_without_improvement_; }
    int id() const { return id_; }
    const std::deque<float>& fitness_history() const { return fitness_history_; }

    static int next_species_id();
    static void set_next_species_id(int id) { species_id_counter_ = id; }

    // Restore stagnation state from checkpoint
    void restore_stagnation(float best_ever, int gens_without_improvement) {
        best_fitness_ever_ = best_ever;
        generations_without_improvement_ = gens_without_improvement;
    }

private:
    int id_;
    Genome representative_;
    std::vector<Genome*> members_;
    float average_fitness_ = 0.0f;
    float total_adjusted_fitness_ = 0.0f;
    float best_fitness_ever_ = 0.0f;
    int generations_without_improvement_ = 0;
    std::deque<float> fitness_history_;

    static int species_id_counter_;
};

} // namespace moonai
