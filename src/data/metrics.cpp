#include "data/metrics.hpp"

#include <algorithm>
#include <numeric>

namespace moonai {

StepMetrics
MetricsCollector::collect(int step,
                          const std::vector<std::unique_ptr<Agent>> &agents,
                          int births, int deaths, int num_species) {
  StepMetrics metrics;
  metrics.step = step;
  metrics.births = births;
  metrics.deaths = deaths;
  metrics.num_species = num_species;

  float predator_energy_sum = 0.0f;
  float prey_energy_sum = 0.0f;
  int predator_energy_count = 0;
  int prey_energy_count = 0;

  for (const auto &agent : agents) {
    if (!agent->alive()) {
      continue;
    }
    if (agent->type() == AgentType::Predator) {
      ++metrics.predator_count;
      predator_energy_sum += agent->energy();
      ++predator_energy_count;
    } else {
      ++metrics.prey_count;
      prey_energy_sum += agent->energy();
      ++prey_energy_count;
    }
  }

  if (predator_energy_count > 0) {
    metrics.avg_predator_energy =
        predator_energy_sum / static_cast<float>(predator_energy_count);
  }
  if (prey_energy_count > 0) {
    metrics.avg_prey_energy =
        prey_energy_sum / static_cast<float>(prey_energy_count);
  }

  std::vector<const Genome *> genomes;
  genomes.reserve(agents.size());
  for (const auto &agent : agents) {
    if (agent->alive()) {
      genomes.push_back(&agent->genome());
    }
  }

  if (!genomes.empty()) {
    metrics.best_fitness =
        (*std::max_element(genomes.begin(), genomes.end(),
                           [](const Genome *lhs, const Genome *rhs) {
                             return lhs->fitness() < rhs->fitness();
                           }))
            ->fitness();

    const float fitness_sum =
        std::accumulate(genomes.begin(), genomes.end(), 0.0f,
                        [](float sum, const Genome *genome) {
                          return sum + genome->fitness();
                        });
    metrics.avg_fitness = fitness_sum / static_cast<float>(genomes.size());

    const float complexity_sum =
        std::accumulate(genomes.begin(), genomes.end(), 0.0f,
                        [](float sum, const Genome *genome) {
                          return sum + static_cast<float>(genome->complexity());
                        });
    metrics.avg_genome_complexity =
        complexity_sum / static_cast<float>(genomes.size());
  }

  history_.push_back(metrics);
  return metrics;
}

} // namespace moonai
