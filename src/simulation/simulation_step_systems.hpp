#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

#include <vector>

namespace moonai::simulation_detail {

void build_sensors(Registry &registry, const FoodStore &food_store,
                   const SimulationConfig &config);
void update_vitals(Registry &registry, const SimulationConfig &config);
void process_food(Registry &registry, FoodStore &food_store,
                  const SimulationConfig &config,
                  std::vector<int> &food_consumed_by);
void process_combat(Registry &registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts);
void apply_movement(Registry &registry, const SimulationConfig &config);
void collect_cpu_step_events(Registry &registry, const FoodStore &food_store,
                             const std::vector<uint8_t> &was_alive,
                             const std::vector<uint8_t> &was_food_active,
                             const std::vector<int> &food_consumed_by,
                             const std::vector<int> &killed_by,
                             const std::vector<uint32_t> &kill_counts,
                             std::vector<SimEvent> &events);
void accumulate_events(EventCounters &counters,
                       const std::vector<SimEvent> &events);

} // namespace moonai::simulation_detail
