#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

#include <vector>

namespace moonai::systems {

void build_sensors(AgentRegistry &self_agents, const AgentRegistry &predator_agents, const AgentRegistry &prey_agents,
                   const Food &food_store, const SimulationConfig &config, float agent_speed,
                   std::vector<float> &sensors_out);
void update_vitals(AgentRegistry &agents, const SimulationConfig &config);
void process_food(AgentRegistry &prey_registry, Food &food_store, const SimulationConfig &config,
                  std::vector<int> &food_consumed_by);
void process_combat(AgentRegistry &predator_registry, AgentRegistry &prey_registry, const SimulationConfig &config,
                    std::vector<int> &killed_by, std::vector<uint32_t> &kill_counts);
void apply_movement(AgentRegistry &agents, const SimulationConfig &config, float agent_speed,
                    const std::vector<float> &decisions);
void collect_food_events(AgentRegistry &prey_registry, const Food &food_store,
                         const std::vector<uint8_t> &was_food_active, const std::vector<int> &food_consumed_by,
                         MetricsSnapshot &metrics);
void collect_combat_events(AgentRegistry &predator_registry, const AgentRegistry &prey_registry,
                           const std::vector<int> &killed_by, const std::vector<uint32_t> &kill_counts,
                           MetricsSnapshot &metrics);
void collect_death_events(const AgentRegistry &registry, MetricsSnapshot &metrics);

} // namespace moonai::systems