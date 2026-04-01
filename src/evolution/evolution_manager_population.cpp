#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/crossover.hpp"
#include "evolution/mutation.hpp"
#include "gpu/gpu_network_cache.hpp"

#include <algorithm>

namespace moonai {

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

  if (predator_gpu_network_cache_) {
    predator_gpu_network_cache_->invalidate();
  }
  if (prey_gpu_network_cache_) {
    prey_gpu_network_cache_->invalidate();
  }
}

uint32_t EvolutionManager::create_predator_offspring(AppState &state, uint32_t parent_a, uint32_t parent_b,
                                                     Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring_predator");
  if (!state.predator.valid(parent_a) || !state.predator.valid(parent_b) || parent_a >= state.predator.genomes.size() ||
      parent_b >= state.predator.genomes.size()) {
    return INVALID_ENTITY;
  }

  Genome child_genome = create_child_genome(state.predator, state.runtime.rng, state.predator.genomes[parent_a],
                                            state.predator.genomes[parent_b]);

  const uint32_t idx = state.predator.create();
  state.predator.pos_x[idx] = spawn_position.x;
  state.predator.pos_y[idx] = spawn_position.y;
  state.predator.vel_x[idx] = 0.0f;
  state.predator.vel_y[idx] = 0.0f;
  state.predator.energy[idx] = config_.offspring_initial_energy;
  state.predator.age[idx] = 0;
  state.predator.alive[idx] = 1;
  state.predator.species_id[idx] = state.predator.species_id[parent_a];
  state.predator.entity_id[idx] = state.runtime.next_agent_id++;
  state.predator.consumption[idx] = 0;

  if (idx >= state.predator.genomes.size()) {
    state.predator.genomes.resize(idx + 1);
  }
  state.predator.genomes[idx] = std::move(child_genome);
  state.predator.network_cache.assign(idx, state.predator.genomes[idx]);

  state.predator.energy[parent_a] -= config_.reproduction_energy_cost;
  state.predator.energy[parent_b] -= config_.reproduction_energy_cost;

  if (predator_gpu_network_cache_) {
    predator_gpu_network_cache_->invalidate();
  }

  return idx;
}

uint32_t EvolutionManager::create_prey_offspring(AppState &state, uint32_t parent_a, uint32_t parent_b,
                                                 Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring_prey");
  if (!state.prey.valid(parent_a) || !state.prey.valid(parent_b) || parent_a >= state.prey.genomes.size() ||
      parent_b >= state.prey.genomes.size()) {
    return INVALID_ENTITY;
  }

  Genome child_genome =
      create_child_genome(state.prey, state.runtime.rng, state.prey.genomes[parent_a], state.prey.genomes[parent_b]);

  const uint32_t idx = state.prey.create();
  state.prey.pos_x[idx] = spawn_position.x;
  state.prey.pos_y[idx] = spawn_position.y;
  state.prey.vel_x[idx] = 0.0f;
  state.prey.vel_y[idx] = 0.0f;
  state.prey.energy[idx] = config_.offspring_initial_energy;
  state.prey.age[idx] = 0;
  state.prey.alive[idx] = 1;
  state.prey.species_id[idx] = state.prey.species_id[parent_a];
  state.prey.entity_id[idx] = state.runtime.next_agent_id++;
  state.prey.consumption[idx] = 0;

  if (idx >= state.prey.genomes.size()) {
    state.prey.genomes.resize(idx + 1);
  }
  state.prey.genomes[idx] = std::move(child_genome);
  state.prey.network_cache.assign(idx, state.prey.genomes[idx]);

  state.prey.energy[parent_a] -= config_.reproduction_energy_cost;
  state.prey.energy[parent_b] -= config_.reproduction_energy_cost;

  if (prey_gpu_network_cache_) {
    prey_gpu_network_cache_->invalidate();
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
  refresh_population_species(state.predator);
  refresh_population_species(state.prey);
}

void EvolutionManager::on_population_destroyed(AgentRegistry &registry, uint32_t entity) {
  if (entity != INVALID_ENTITY && static_cast<std::size_t>(entity) + 1 == registry.genomes.size()) {
    registry.genomes.pop_back();
  }
  registry.network_cache.remove(entity);
}

void EvolutionManager::on_population_moved(AgentRegistry &registry, uint32_t from, uint32_t to) {
  if (from == to) {
    return;
  }

  if (from < registry.genomes.size()) {
    if (to >= registry.genomes.size()) {
      registry.genomes.resize(static_cast<std::size_t>(to) + 1);
    }
    registry.genomes[to] = std::move(registry.genomes[from]);
  }

  registry.network_cache.move_entity(from, to);
}

void EvolutionManager::on_predator_destroyed(AppState &state, uint32_t entity) {
  on_population_destroyed(state.predator, entity);
  if (predator_gpu_network_cache_) {
    predator_gpu_network_cache_->invalidate();
  }
}

void EvolutionManager::on_predator_moved(AppState &state, uint32_t from, uint32_t to) {
  on_population_moved(state.predator, from, to);
  if (predator_gpu_network_cache_) {
    predator_gpu_network_cache_->invalidate();
  }
}

void EvolutionManager::on_prey_destroyed(AppState &state, uint32_t entity) {
  on_population_destroyed(state.prey, entity);
  if (prey_gpu_network_cache_) {
    prey_gpu_network_cache_->invalidate();
  }
}

void EvolutionManager::on_prey_moved(AppState &state, uint32_t from, uint32_t to) {
  on_population_moved(state.prey, from, to);
  if (prey_gpu_network_cache_) {
    prey_gpu_network_cache_->invalidate();
  }
}

} // namespace moonai
