#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/network_cache.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/species.hpp"
#include "simulation/entity.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonai {

// Forward declaration
class Registry;

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config, Random &rng);
  ~EvolutionManager();

  void initialize(int num_inputs, int num_outputs);

  Genome create_initial_genome() const;
  Genome create_child_genome(const Genome &parent_a,
                             const Genome &parent_b) const;

  // ECS-aware methods (Phase 4)
  void seed_initial_population_ecs(Registry &registry);

  // Validates parent handles before creating offspring
  // Returns INVALID_ENTITY if parents are invalid or dead
  Entity create_offspring_ecs(Registry &registry, Entity parent_a,
                              Entity parent_b, Vec2 spawn_position);

  void refresh_fitness_ecs(const Registry &registry);
  void refresh_species_ecs(Registry &registry);

  // Compute actions: uses NetworkCache for NN inference
  // Note: Registry must be non-const to write brain outputs
  void compute_actions_ecs(Registry &registry, std::vector<Vec2> &actions);

  // Called when entities die (cleanup)
  void on_entity_destroyed(Entity e);

  // Accessors for ECS integration
  NetworkCache &network_cache() {
    return network_cache_;
  }
  const NetworkCache &network_cache() const {
    return network_cache_;
  }

  Genome *genome_for(Entity e);
  const Genome *genome_for(Entity e) const;

  const std::vector<Species> &species() const {
    return species_;
  }
  int species_count() const {
    return static_cast<int>(species_.size());
  }

  // ECS-aware fitness by type
  void get_fitness_by_type_ecs(const Registry &registry, float &best_predator,
                               float &avg_predator, float &best_prey,
                               float &avg_prey) const;

  void update_config(const SimulationConfig &cfg) {
    config_ = cfg;
  }
  void enable_gpu(bool use_gpu) {
    use_gpu_ = use_gpu;
  }
  bool gpu_enabled() const {
    return use_gpu_;
  }

private:
  SimulationConfig config_;
  Random &rng_;
  InnovationTracker tracker_;
  std::vector<Species> species_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  bool use_gpu_ = false;
  int species_refresh_step_ = -1;

  // ECS integration (Phase 4)
  // Entity -> Genome mapping (flat POD, fine for ECS)
  std::unordered_map<Entity, Genome, EntityHash> entity_genomes_;

  // Entity -> NeuralNetwork mapping (variable topology, separate cache)
  NetworkCache network_cache_;
};

} // namespace moonai
