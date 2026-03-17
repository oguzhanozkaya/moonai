#pragma once

#include "evolution/genome.hpp"

#include <vector>

namespace moonai {

struct GenerationMetrics {
    int generation = 0;
    int predator_count = 0;
    int prey_count = 0;
    float best_fitness = 0.0f;
    float avg_fitness = 0.0f;
    int num_species = 0;
    float avg_genome_complexity = 0.0f;  // avg number of connections
};

class MetricsCollector {
public:
    GenerationMetrics collect(int generation,
                              const std::vector<Genome>& population,
                              int predator_count, int prey_count);

    const std::vector<GenerationMetrics>& history() const { return history_; }

private:
    std::vector<GenerationMetrics> history_;
};

} // namespace moonai
