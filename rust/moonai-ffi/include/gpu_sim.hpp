#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct GpuSimContext GpuSimContext;

GpuSimContext* gpu_sim_create(
    std::size_t max_predators,
    std::size_t max_prey,
    std::size_t max_food,
    std::size_t grid_size
);

void gpu_sim_destroy(GpuSimContext* ctx);

int gpu_sim_is_available();

int gpu_sim_prepare_step(GpuSimContext* ctx);

int gpu_sim_build_sensors(GpuSimContext* ctx);

int gpu_sim_run_inference(
    GpuSimContext* ctx,
    const float* predator_sensors,
    const float* prey_sensors,
    std::size_t predator_count,
    std::size_t prey_count
);

int gpu_sim_resolve_step(GpuSimContext* ctx);

int gpu_sim_get_results(
    GpuSimContext* ctx,
    float* out_predator_pos_x,
    float* out_predator_pos_y,
    float* out_predator_vel_x,
    float* out_predator_vel_y,
    float* out_predator_energy,
    int* out_predator_age,
    uint8_t* out_predator_alive,
    std::size_t predator_count,
    float* out_prey_pos_x,
    float* out_prey_pos_y,
    float* out_prey_vel_x,
    float* out_prey_vel_y,
    float* out_prey_energy,
    int* out_prey_age,
    uint8_t* out_prey_alive,
    std::size_t prey_count,
    float* out_food_pos_x,
    float* out_food_pos_y,
    uint8_t* out_food_active,
    std::size_t food_count
);

const float* gpu_sim_get_predator_outputs(GpuSimContext* ctx, std::size_t* out_count);
const float* gpu_sim_get_prey_outputs(GpuSimContext* ctx, std::size_t* out_count);

const int8_t* gpu_sim_get_error(GpuSimContext* ctx);

}