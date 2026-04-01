#include "evolution/evolution_manager.hpp"
#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/crossover.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/mutation.hpp"
#include "gpu/gpu_batch.hpp"
#include "gpu/gpu_network_cache.hpp"
#include "simulation/simulation_step_systems.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig &config) : config_(config) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize_population(AgentRegistry &registry) const {
  registry.innovation_tracker = InnovationTracker();
  registry.innovation_tracker.set_counters(0, static_cast<std::uint32_t>(num_inputs_ + num_outputs_ + 1));
  registry.species.clear();
  registry.genomes.clear();
  registry.network_cache.clear();
  if (registry.gpu_network_cache) {
    registry.gpu_network_cache->invalidate();
  }
}

void EvolutionManager::initialize(AppState &state, int num_inputs, int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;

  Species::reset_id_counter();
  initialize_population(state.predator);
  initialize_population(state.prey);
}

using simulation_detail::OUTPUT_COUNT;
using simulation_detail::SENSOR_COUNT;

void EvolutionManager::compute_actions_for_population(AgentRegistry &registry, const std::vector<float> &sensors,
                                                      std::vector<float> &decisions_out) const {
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());

  std::vector<float> all_outputs;
  registry.network_cache.activate_batch(entity_count, sensors, all_outputs, SENSOR_COUNT, OUTPUT_COUNT);

  decisions_out.resize(entity_count * OUTPUT_COUNT);
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    decisions_out[idx * OUTPUT_COUNT] = all_outputs[idx * OUTPUT_COUNT];
    decisions_out[idx * OUTPUT_COUNT + 1] = all_outputs[idx * OUTPUT_COUNT + 1];
  }
}

void EvolutionManager::compute_actions(AppState &state, const std::vector<float> &predator_sensors,
                                       const std::vector<float> &prey_sensors, std::vector<float> &predator_decisions,
                                       std::vector<float> &prey_decisions) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  compute_actions_for_population(state.predator, predator_sensors, predator_decisions);
  compute_actions_for_population(state.prey, prey_sensors, prey_decisions);
}

Genome EvolutionManager::create_initial_genome(AgentRegistry &registry, Random &rng) const {
  Genome genome(num_inputs_, num_outputs_);
  for (const auto &in_node : genome.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
      continue;
    }
    for (const auto &out_node : genome.nodes()) {
      if (out_node.type != NodeType::Output) {
        continue;
      }

      const std::uint32_t innovation = registry.innovation_tracker.get_innovation(in_node.id, out_node.id);
      genome.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f), true, innovation});
    }
  }
  return genome;
}

Genome EvolutionManager::create_child_genome(AgentRegistry &registry, Random &rng, const Genome &parent_a,
                                             const Genome &parent_b) const {
  Genome child = Crossover::crossover(parent_a, parent_b, rng);
  Mutation::mutate(child, rng, config_, registry.innovation_tracker);
  return child;
}

void EvolutionManager::seed_initial_population(AppState &state) {
  state.predator.clear();
  state.prey.clear();
  state.runtime.next_agent_id = 1;
  initialize_population(state.predator);
  initialize_population(state.prey);

  const float grid_size = static_cast<float>(config_.grid_size);

  auto seed_predator = [&] {
    const uint32_t idx = state.predator.create();
    Genome genome = create_initial_genome(state.predator, state.runtime.rng);
    if (idx >= state.predator.genomes.size()) {
      state.predator.genomes.resize(idx + 1);
    }
    state.predator.genomes[idx] = genome;
    state.predator.network_cache.assign(idx, state.predator.genomes[idx]);

    state.predator.pos_x[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.predator.pos_y[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.predator.vel_x[idx] = 0.0f;
    state.predator.vel_y[idx] = 0.0f;
    state.predator.energy[idx] = config_.initial_energy;
    state.predator.age[idx] = 0;
    state.predator.alive[idx] = 1;
    state.predator.species_id[idx] = 0;
    state.predator.entity_id[idx] = state.runtime.next_agent_id++;
    state.predator.consumption[idx] = 0;
  };

  auto seed_prey = [&] {
    const uint32_t idx = state.prey.create();
    Genome genome = create_initial_genome(state.prey, state.runtime.rng);
    if (idx >= state.prey.genomes.size()) {
      state.prey.genomes.resize(idx + 1);
    }
    state.prey.genomes[idx] = genome;
    state.prey.network_cache.assign(idx, state.prey.genomes[idx]);

    state.prey.pos_x[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.prey.pos_y[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.prey.vel_x[idx] = 0.0f;
    state.prey.vel_y[idx] = 0.0f;
    state.prey.energy[idx] = config_.initial_energy;
    state.prey.age[idx] = 0;
    state.prey.alive[idx] = 1;
    state.prey.species_id[idx] = 0;
    state.prey.entity_id[idx] = state.runtime.next_agent_id++;
    state.prey.consumption[idx] = 0;
  };

  for (int i = 0; i < config_.predator_count; ++i) {
    seed_predator();
  }
  for (int i = 0; i < config_.prey_count; ++i) {
    seed_prey();
  }

  if (state.predator.gpu_network_cache) {
    state.predator.gpu_network_cache->invalidate();
  }
  if (state.prey.gpu_network_cache) {
    state.prey.gpu_network_cache->invalidate();
  }
}

uint32_t EvolutionManager::create_offspring(AppState &state, AgentRegistry &registry, uint32_t parent_a,
                                            uint32_t parent_b, Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring_predator");
  if (!registry.valid(parent_a) || !registry.valid(parent_b) || parent_a >= registry.genomes.size() ||
      parent_b >= registry.genomes.size()) {
    return INVALID_ENTITY;
  }

  Genome child_genome =
      create_child_genome(registry, state.runtime.rng, registry.genomes[parent_a], registry.genomes[parent_b]);

  const uint32_t idx = registry.create();
  registry.pos_x[idx] = spawn_position.x;
  registry.pos_y[idx] = spawn_position.y;
  registry.vel_x[idx] = 0.0f;
  registry.vel_y[idx] = 0.0f;
  registry.energy[idx] = config_.offspring_initial_energy;
  registry.age[idx] = 0;
  registry.alive[idx] = 1;
  registry.species_id[idx] = registry.species_id[parent_a];
  registry.entity_id[idx] = state.runtime.next_agent_id++;
  registry.consumption[idx] = 0;

  if (idx >= registry.genomes.size()) {
    registry.genomes.resize(idx + 1);
  }
  registry.genomes[idx] = std::move(child_genome);
  registry.network_cache.assign(idx, registry.genomes[idx]);

  registry.energy[parent_a] -= config_.reproduction_energy_cost;
  registry.energy[parent_b] -= config_.reproduction_energy_cost;

  if (registry.gpu_network_cache) {
    registry.gpu_network_cache->invalidate();
  }

  return idx;
}

void EvolutionManager::refresh_population_species(AgentRegistry &registry) const {
  auto &species = registry.species;
  for (auto &entry : species) {
    entry.clear_members();
  }

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (idx >= registry.genomes.size()) {
      continue;
    }

    Genome &genome = registry.genomes[idx];
    int assigned_species_id = -1;

    for (auto &entry : species) {
      if (entry.is_compatible(genome, config_.compatibility_threshold, config_.c1_excess, config_.c2_disjoint,
                              config_.c3_weight)) {
        entry.add_member(idx, genome);
        assigned_species_id = entry.id();
        break;
      }
    }

    if (assigned_species_id < 0) {
      Species entry(genome);
      entry.add_member(idx, genome);
      assigned_species_id = entry.id();
      species.push_back(std::move(entry));
    }

    registry.species_id[idx] = static_cast<uint32_t>(assigned_species_id);
  }

  for (auto &entry : species) {
    entry.refresh_summary();
  }

  species.erase(
      std::remove_if(species.begin(), species.end(), [](const Species &entry) { return entry.members().empty(); }),
      species.end());

  for (auto &entry : species) {
    const uint32_t representative_idx = entry.members().front().entity;
    if (representative_idx < registry.genomes.size()) {
      entry.set_representative(registry.genomes[representative_idx]);
    }
  }
}

void EvolutionManager::refresh_species(AppState &state) {
  MOONAI_PROFILE_SCOPE("refresh_species");
  refresh_population_species(state.predator);
  refresh_population_species(state.prey);
}

namespace {

bool launch_population_gpu_neural(AgentRegistry &registry, gpu::GpuNetworkCache &gpu_cache,
                                  gpu::GpuPopulationBuffer &buffer, std::size_t count, cudaStream_t stream) {
  std::vector<std::pair<uint32_t, int>> network_entities_with_indices;
  network_entities_with_indices.reserve(count);

  const uint32_t entity_count = static_cast<uint32_t>(count);
  for (uint32_t entity = 0; entity < entity_count; ++entity) {
    if (registry.network_cache.has(entity)) {
      network_entities_with_indices.emplace_back(entity, static_cast<int>(entity));
    }
  }

  if (network_entities_with_indices.empty()) {
    return true;
  }

  if (gpu_cache.is_dirty() || gpu_cache.entity_mapping().size() != network_entities_with_indices.size() ||
      !std::equal(gpu_cache.entity_mapping().begin(), gpu_cache.entity_mapping().end(),
                  network_entities_with_indices.begin(),
                  [](uint32_t entity, const std::pair<uint32_t, int> &entity_with_index) {
                    return entity == entity_with_index.first;
                  })) {
    gpu_cache.build_from(registry.network_cache, network_entities_with_indices);
  }

  return gpu_cache.launch_inference_async(buffer.device_sensor_inputs(), buffer.device_brain_outputs(),
                                          network_entities_with_indices.size(), stream);
}

} // namespace

void EvolutionManager::enable_gpu(AppState &state, bool use_gpu) {
  if (use_gpu) {
    if (!state.predator.gpu_network_cache) {
      state.predator.gpu_network_cache = std::make_unique<gpu::GpuNetworkCache>();
      state.predator.gpu_network_cache->invalidate();
    }
    if (!state.prey.gpu_network_cache) {
      state.prey.gpu_network_cache = std::make_unique<gpu::GpuNetworkCache>();
      state.prey.gpu_network_cache->invalidate();
    }
    state.runtime.gpu_enabled = true;
    spdlog::info("GPU neural inference enabled");
  } else {
    state.predator.gpu_network_cache.reset();
    state.prey.gpu_network_cache.reset();
    state.runtime.gpu_enabled = false;
    spdlog::info("GPU neural inference disabled");
  }
}

bool EvolutionManager::launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!state.predator.gpu_network_cache || !state.prey.gpu_network_cache) {
    spdlog::error("GPU neural caches not initialized");
    return false;
  }

  if (!launch_population_gpu_neural(state.predator, *state.predator.gpu_network_cache, gpu_batch.predator_buffer(),
                                    state.predator.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  if (!launch_population_gpu_neural(state.prey, *state.prey.gpu_network_cache, gpu_batch.prey_buffer(),
                                    state.prey.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  return true;
}

} // namespace moonai
