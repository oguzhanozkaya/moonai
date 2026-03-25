#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/species.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/simulation_manager.hpp"

#include <memory>
#include <string>
#include <vector>

namespace moonai {

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config, Random &rng);
  ~EvolutionManager();

  void initialize(int num_inputs, int num_outputs);
  void seed_initial_population(SimulationManager &sim);

  // Compute agent actions using CPU (sensor building + inference on CPU)
  void compute_actions(const SimulationManager &sim,
                       std::vector<Vec2> &actions);

  // Full ecology step on GPU (sensing, inference, movement, food, attacks)
  // Returns true if GPU was used, false if should fall back to CPU
  bool step_gpu(SimulationManager &sim, int step_index);

  Genome create_initial_genome() const;
  Genome create_child_genome(const Genome &parent_a,
                             const Genome &parent_b) const;
  AgentId create_offspring(SimulationManager &sim, AgentId parent_a,
                           AgentId parent_b, Vec2 spawn_position);
  void refresh_species(SimulationManager &sim);
  void refresh_fitness(const SimulationManager &sim);

  const std::vector<Species> &species() const {
    return species_;
  }
  int species_count() const {
    return static_cast<int>(species_.size());
  }
  const Genome *genome_at(const SimulationManager &sim, int idx) const;
  NeuralNetwork *network_at(const SimulationManager &sim, int idx) const;
  void get_fitness_by_type(const SimulationManager &sim, float &best_predator,
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
  std::unique_ptr<Agent> create_agent(AgentId id, AgentType type, Vec2 position,
                                      Genome genome) const;
  float default_fitness(const Agent &agent) const;
  bool current_gpu_layout_matches(const SimulationManager &sim) const;
  bool rebuild_gpu_runtime(const SimulationManager &sim);

  SimulationConfig config_;
  Random &rng_;
  InnovationTracker tracker_;
  std::vector<Species> species_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  bool use_gpu_ = false;
  bool gpu_runtime_ready_ = false;
  bool gpu_warning_emitted_ = false;
  int species_refresh_step_ = -1;
  std::vector<AgentId> gpu_layout_agent_ids_;
  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
  std::string gpu_activation_function_;
};

} // namespace moonai
