#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {

typedef struct SimulationContext SimulationContext;

int gpu_sim_is_available(void);

SimulationContext* sim_create(void);
void sim_destroy(SimulationContext* ctx);

int sim_init(
    SimulationContext* ctx,
    int grid_size,
    int predator_count,
    int prey_count,
    int food_count,
    float predator_speed,
    float prey_speed,
    float vision_range,
    float interaction_range,
    float mate_range,
    float food_respawn_rate,
    float energy_drain_per_step,
    float energy_gain_from_kill,
    float energy_gain_from_food,
    float initial_energy,
    float max_energy,
    float reproduction_energy_threshold,
    float reproduction_energy_cost,
    float offspring_initial_energy,
    int max_age,
    float mutation_rate,
    float weight_mutation_power,
    float add_node_rate,
    float add_connection_rate,
    float delete_connection_rate,
    int max_hidden_nodes,
    int max_steps,
    float compatibility_threshold,
    float compatibility_min_normalization,
    float c1_excess,
    float c2_disjoint,
    float c3_weight,
    int seed,
    const int8_t* output_dir,
    int report_interval_steps
);

int sim_step(SimulationContext* ctx);
int sim_prepare_step(SimulationContext* ctx);
int sim_resolve_step(SimulationContext* ctx);
int sim_run_inference(SimulationContext* ctx);
int sim_post_step(SimulationContext* ctx);

void sim_refresh_species(SimulationContext* ctx);

void sim_get_predator_state(
    SimulationContext* ctx,
    const float** out_pos_x,
    const float** out_pos_y,
    const float** out_vel_x,
    const float** out_vel_y,
    const float** out_energy,
    const int32_t** out_age,
    const uint8_t** out_alive,
    size_t* out_count
);

void sim_get_prey_state(
    SimulationContext* ctx,
    const float** out_pos_x,
    const float** out_pos_y,
    const float** out_vel_x,
    const float** out_vel_y,
    const float** out_energy,
    const int32_t** out_age,
    const uint8_t** out_alive,
    size_t* out_count
);

void sim_get_food_state(
    SimulationContext* ctx,
    const float** out_pos_x,
    const float** out_pos_y,
    const uint8_t** out_active,
    size_t* out_count
);

int sim_get_step(SimulationContext* ctx);
int sim_get_predator_count(SimulationContext* ctx);
int sim_get_prey_count(SimulationContext* ctx);

void sim_set_predator_alive(SimulationContext* ctx, size_t idx, uint8_t alive);
void sim_set_prey_alive(SimulationContext* ctx, size_t idx, uint8_t alive);

void sim_compact_predator(SimulationContext* ctx);
void sim_compact_prey(SimulationContext* ctx);

const int8_t* sim_get_error(SimulationContext* ctx);

}