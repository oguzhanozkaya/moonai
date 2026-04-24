#include "simulation.hpp"

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/simulation.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

struct SimulationContext {
    SimulationConfig config;
    AppState* state;
    EvolutionManager* evolution;

    bool initialized;
    char error_message[512];

    SimulationContext()
        : state(nullptr)
        , evolution(nullptr)
        , initialized(false)
    {
        error_message[0] = '\0';
    }

    void set_error(const char* msg) {
        snprintf(error_message, sizeof(error_message), "%s", msg);
    }
};

extern "C" {

SimulationContext* sim_create(void) {
    return new SimulationContext();
}

void sim_destroy(SimulationContext* ctx) {
    delete ctx;
}

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
) {
    if (!ctx) {
        return 0;
    }

    ctx->config.grid_size = grid_size;
    ctx->config.predator_count = predator_count;
    ctx->config.prey_count = prey_count;
    ctx->config.food_count = food_count;
    ctx->config.predator_speed = predator_speed;
    ctx->config.prey_speed = prey_speed;
    ctx->config.vision_range = vision_range;
    ctx->config.interaction_range = interaction_range;
    ctx->config.mate_range = mate_range;
    ctx->config.food_respawn_rate = food_respawn_rate;
    ctx->config.energy_drain_per_step = energy_drain_per_step;
    ctx->config.energy_gain_from_kill = energy_gain_from_kill;
    ctx->config.energy_gain_from_food = energy_gain_from_food;
    ctx->config.initial_energy = initial_energy;
    ctx->config.max_energy = max_energy;
    ctx->config.reproduction_energy_threshold = reproduction_energy_threshold;
    ctx->config.reproduction_energy_cost = reproduction_energy_cost;
    ctx->config.offspring_initial_energy = offspring_initial_energy;
    ctx->config.max_age = max_age;
    ctx->config.mutation_rate = mutation_rate;
    ctx->config.weight_mutation_power = weight_mutation_power;
    ctx->config.add_node_rate = add_node_rate;
    ctx->config.add_connection_rate = add_connection_rate;
    ctx->config.delete_connection_rate = delete_connection_rate;
    ctx->config.max_hidden_nodes = max_hidden_nodes;
    ctx->config.max_steps = max_steps;
    ctx->config.compatibility_threshold = compatibility_threshold;
    ctx->config.compatibility_min_normalization = compatibility_min_normalization;
    ctx->config.c1_excess = c1_excess;
    ctx->config.c2_disjoint = c2_disjoint;
    ctx->config.c3_weight = c3_weight;
    ctx->config.seed = seed;
    ctx->config.report_interval_steps = report_interval_steps;

    if (output_dir) {
        ctx->config.output_dir = reinterpret_cast<const char*>(output_dir);
    }

    try {
        ctx->state = new AppState(ctx->config.seed);
        ctx->evolution = new EvolutionManager(ctx->config);

        moonai::simulation::initialize(*ctx->state, ctx->config);
        ctx->evolution->initialize(*ctx->state, 35, 2);
        ctx->evolution->seed_initial_population(*ctx->state);
        ctx->evolution->refresh_species(*ctx->state);
        ctx->evolution->initialize_inference(*ctx->state);

        ctx->initialized = true;
        return 1;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error during initialization");
        return 0;
    }
}

int sim_prepare_step(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    try {
        return moonai::simulation::prepare_step(*ctx->state, ctx->config) ? 1 : 0;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error in prepare_step");
        return 0;
    }
}

int sim_run_inference(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    try {
        return ctx->evolution->run_inference(*ctx->state) ? 1 : 0;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error in run_inference");
        return 0;
    }
}

int sim_resolve_step(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    try {
        return moonai::simulation::resolve_step(*ctx->state, ctx->config) ? 1 : 0;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error in resolve_step");
        return 0;
    }
}

int sim_post_step(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    try {
        moonai::simulation::post_step(*ctx->state, ctx->config);
        ctx->evolution->post_step(*ctx->state);
        return 1;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error in post_step");
        return 0;
    }
}

int sim_step(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    try {
        if (!sim_prepare_step(ctx)) {
            return 0;
        }

        if (!sim_run_inference(ctx)) {
            return 0;
        }

        if (!sim_resolve_step(ctx)) {
            return 0;
        }

        if (!sim_post_step(ctx)) {
            return 0;
        }

        ctx->state->runtime.step++;
        return 1;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return 0;
    } catch (...) {
        ctx->set_error("Unknown error in step");
        return 0;
    }
}

void sim_refresh_species(SimulationContext* ctx) {
    if (ctx && ctx->initialized) {
        ctx->evolution->refresh_species(*ctx->state);
    }
}

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
) {
    if (!ctx || !ctx->initialized) {
        return;
    }

    if (out_pos_x) *out_pos_x = ctx->state->predator.pos_x.data();
    if (out_pos_y) *out_pos_y = ctx->state->predator.pos_y.data();
    if (out_vel_x) *out_vel_x = ctx->state->predator.vel_x.data();
    if (out_vel_y) *out_vel_y = ctx->state->predator.vel_y.data();
    if (out_energy) *out_energy = ctx->state->predator.energy.data();
    if (out_age) *out_age = reinterpret_cast<const int32_t*>(ctx->state->predator.age.data());
    if (out_alive) *out_alive = ctx->state->predator.alive.data();
    if (out_count) *out_count = ctx->state->predator.size();
}

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
) {
    if (!ctx || !ctx->initialized) {
        return;
    }

    if (out_pos_x) *out_pos_x = ctx->state->prey.pos_x.data();
    if (out_pos_y) *out_pos_y = ctx->state->prey.pos_y.data();
    if (out_vel_x) *out_vel_x = ctx->state->prey.vel_x.data();
    if (out_vel_y) *out_vel_y = ctx->state->prey.vel_y.data();
    if (out_energy) *out_energy = ctx->state->prey.energy.data();
    if (out_age) *out_age = reinterpret_cast<const int32_t*>(ctx->state->prey.age.data());
    if (out_alive) *out_alive = ctx->state->prey.alive.data();
    if (out_count) *out_count = ctx->state->prey.size();
}

void sim_get_food_state(
    SimulationContext* ctx,
    const float** out_pos_x,
    const float** out_pos_y,
    const uint8_t** out_active,
    size_t* out_count
) {
    if (!ctx || !ctx->initialized) {
        return;
    }

    if (out_pos_x) *out_pos_x = ctx->state->food.pos_x.data();
    if (out_pos_y) *out_pos_y = ctx->state->food.pos_y.data();
    if (out_active) *out_active = ctx->state->food.active.data();
    if (out_count) *out_count = ctx->state->food.size();
}

int sim_get_step(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return ctx->state->runtime.step;
}

int sim_get_predator_count(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return static_cast<int>(ctx->state->predator.size());
}

int sim_get_prey_count(SimulationContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return static_cast<int>(ctx->state->prey.size());
}

void sim_set_predator_alive(SimulationContext* ctx, size_t idx, uint8_t alive) {
    if (ctx && ctx->initialized && idx < ctx->state->predator.size()) {
        ctx->state->predator.alive[idx] = alive;
    }
}

void sim_set_prey_alive(SimulationContext* ctx, size_t idx, uint8_t alive) {
    if (ctx && ctx->initialized && idx < ctx->state->prey.size()) {
        ctx->state->prey.alive[idx] = alive;
    }
}

void sim_compact_predator(SimulationContext* ctx) {
    if (ctx && ctx->initialized) {
        ctx->state->predator.compact();
    }
}

void sim_compact_prey(SimulationContext* ctx) {
    if (ctx && ctx->initialized) {
        ctx->state->prey.compact();
    }
}

const int8_t* sim_get_error(SimulationContext* ctx) {
    if (!ctx) {
        return reinterpret_cast<const int8_t*>("Null context");
    }
    return reinterpret_cast<const int8_t*>(ctx->error_message);
}

} // extern "C"