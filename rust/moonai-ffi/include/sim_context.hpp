#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/simulation.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct SimContext SimContext;

SimContext* sim_create(const SimulationConfig* config);
void sim_destroy(SimContext* ctx);

int sim_init(SimContext* ctx);
int sim_step(SimContext* ctx);

int sim_prepare_step(SimContext* ctx);
int sim_resolve_step(SimContext* ctx);
int sim_post_step(SimContext* ctx);

int sim_run_inference(SimContext* ctx);

void sim_refresh_species(SimContext* ctx);

void sim_get_state(
    SimContext* ctx,

    float** out_predator_pos_x,
    float** out_predator_pos_y,
    float** out_predator_vel_x,
    float** out_predator_vel_y,
    float** out_predator_energy,
    int** out_predator_age,
    uint8_t** out_predator_alive,
    size_t* out_predator_count,

    float** out_prey_pos_x,
    float** out_prey_pos_y,
    float** out_prey_vel_x,
    float** out_prey_vel_y,
    float** out_prey_energy,
    int** out_prey_age,
    uint8_t** out_prey_alive,
    size_t* out_prey_count,

    float** out_food_pos_x,
    float** out_food_pos_y,
    uint8_t** out_food_active,
    size_t* out_food_count
);

int sim_get_step(SimContext* ctx);
int sim_get_predator_count(SimContext* ctx);
int sim_get_prey_count(SimContext* ctx);

void sim_log_report(SimContext* ctx);

const char* sim_get_error(SimContext* ctx);

}
