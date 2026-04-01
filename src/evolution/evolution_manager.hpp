#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"

#include <vector>

namespace moonai {

struct AppState;
struct AgentRegistry;
namespace gpu {
class GpuBatch;
class GpuNetworkCache;
} // namespace gpu

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config);
  ~EvolutionManager();

  void initialize(AppState &state, int num_inputs, int num_outputs);

  void seed_initial_population(AppState &state);

  uint32_t create_offspring(AppState &state, AgentRegistry &registry, uint32_t parent_a, uint32_t parent_b,
                            Vec2 spawn_position);
  void refresh_species(AppState &state);

  void compute_actions(AppState &state, const std::vector<float> &predator_sensors,
                       const std::vector<float> &prey_sensors, std::vector<float> &predator_decisions,
                       std::vector<float> &prey_decisions);

  void enable_gpu(AppState &state, bool use_gpu);

  // GPU neural inference (called by SimulationManager during GPU step)
  bool launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch);

private:
  Genome create_initial_genome(AgentRegistry &registry, Random &rng) const;
  Genome create_child_genome(AgentRegistry &registry, Random &rng, const Genome &parent_a,
                             const Genome &parent_b) const;
  void initialize_population(AgentRegistry &registry) const;
  void compute_actions_for_population(AgentRegistry &registry, const std::vector<float> &sensors,
                                      std::vector<float> &decisions_out) const;
  void refresh_population_species(AgentRegistry &registry) const;

  const SimulationConfig &config_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
};

} // namespace moonai
