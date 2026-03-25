#include "data/metrics.hpp"

#include <algorithm>
#include <numeric>

namespace moonai {

GenerationMetrics
MetricsCollector::collect(int generation, const std::vector<Genome> &population,
                          int predator_count, int prey_count) {
  GenerationMetrics m;
  m.generation = generation;
  m.predator_count = predator_count;
  m.prey_count = prey_count;

  if (!population.empty()) {
    m.best_fitness = std::max_element(population.begin(), population.end(),
                                      [](const Genome &a, const Genome &b) {
                                        return a.fitness() < b.fitness();
                                      })
                         ->fitness();

    float total_fitness = std::accumulate(
        population.begin(), population.end(), 0.0f,
        [](float sum, const Genome &g) { return sum + g.fitness(); });
    m.avg_fitness = total_fitness / static_cast<float>(population.size());

    float total_conns = std::accumulate(
        population.begin(), population.end(), 0.0f,
        [](float sum, const Genome &g) {
          return sum + static_cast<float>(g.connections().size());
        });
    m.avg_genome_complexity =
        total_conns / static_cast<float>(population.size());
  }

  history_.push_back(m);
  return m;
}

} // namespace moonai
