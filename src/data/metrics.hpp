#pragma once

#include "evolution/genome.hpp"
#include "simulation/agent.hpp"

#include <memory>
#include <vector>

namespace moonai {

struct StepMetrics {
  int step = 0;
  int predator_count = 0;
  int prey_count = 0;
  int births = 0;
  int deaths = 0;
  float best_fitness = 0.0f;
  float avg_fitness = 0.0f;
  int num_species = 0;
  float avg_genome_complexity = 0.0f;
  float avg_predator_energy = 0.0f;
  float avg_prey_energy = 0.0f;
};

class MetricsCollector {
public:
  StepMetrics collect(int step,
                      const std::vector<std::unique_ptr<Agent>> &agents,
                      int births, int deaths, int num_species);

  const std::vector<StepMetrics> &history() const {
    return history_;
  }

private:
  std::vector<StepMetrics> history_;
};

} // namespace moonai
