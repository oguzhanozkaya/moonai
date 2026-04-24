#include "sim_context.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

struct SimContext {
    SimulationConfig config;
    AppState* state;
    EvolutionManager* evolution;

    char error_message[512];
    bool initialized;
    bool error;

    SimContext(const SimulationConfig* cfg)
        : config(*cfg)
        , state(nullptr)
        , evolution(nullptr)
        , initialized(false)
        , error(false)
    {
        error_message[0] = '\0';
    }

    void set_error(const char* msg) {
        snprintf(error_message, sizeof(error_message), "%s", msg);
        error = true;
    }
};

SimContext* sim_create(const SimulationConfig* config) {
    if (!config) {
        return nullptr;
    }
    return new SimContext(config);
}

void sim_destroy(SimContext* ctx) {
    delete ctx;
}

int sim_init(SimContext* ctx) {
    if (!ctx || ctx->initialized) {
        return 0;
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

int sim_prepare_step(SimContext* ctx) {
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

int sim_run_inference(SimContext* ctx) {
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

int sim_resolve_step(SimContext* ctx) {
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

int sim_post_step(SimContext* ctx) {
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

int sim_step(SimContext* ctx) {
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

void sim_refresh_species(SimContext* ctx) {
    if (ctx && ctx->initialized) {
        ctx->evolution->refresh_species(*ctx->state);
    }
}

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
) {
    if (!ctx || !ctx->initialized) {
        return;
    }

    if (out_predator_pos_x) *out_predator_pos_x = ctx->state->predator.pos_x.data();
    if (out_predator_pos_y) *out_predator_pos_y = ctx->state->predator.pos_y.data();
    if (out_predator_vel_x) *out_predator_vel_x = ctx->state->predator.vel_x.data();
    if (out_predator_vel_y) *out_predator_vel_y = ctx->state->predator.vel_y.data();
    if (out_predator_energy) *out_predator_energy = ctx->state->predator.energy.data();
    if (out_predator_age) *out_predator_age = ctx->state->predator.age.data();
    if (out_predator_alive) *out_predator_alive = ctx->state->predator.alive.data();
    if (out_predator_count) *out_predator_count = ctx->state->predator.size();

    if (out_prey_pos_x) *out_prey_pos_x = ctx->state->prey.pos_x.data();
    if (out_prey_pos_y) *out_prey_pos_y = ctx->state->prey.pos_y.data();
    if (out_prey_vel_x) *out_prey_vel_x = ctx->state->prey.vel_x.data();
    if (out_prey_vel_y) *out_prey_vel_y = ctx->state->prey.vel_y.data();
    if (out_prey_energy) *out_prey_energy = ctx->state->prey.energy.data();
    if (out_prey_age) *out_prey_age = ctx->state->prey.age.data();
    if (out_prey_alive) *out_prey_alive = ctx->state->prey.alive.data();
    if (out_prey_count) *out_prey_count = ctx->state->prey.size();

    if (out_food_pos_x) *out_food_pos_x = ctx->state->food.pos_x.data();
    if (out_food_pos_y) *out_food_pos_y = ctx->state->food.pos_y.data();
    if (out_food_active) *out_food_active = ctx->state->food.active.data();
    if (out_food_count) *out_food_count = ctx->state->food.size();
}

int sim_get_step(SimContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return ctx->state->runtime.step;
}

int sim_get_predator_count(SimContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return static_cast<int>(ctx->state->predator.size());
}

int sim_get_prey_count(SimContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    return static_cast<int>(ctx->state->prey.size());
}

void sim_log_report(SimContext* ctx) {
    (void)ctx;
}

const char* sim_get_error(SimContext* ctx) {
    if (!ctx) {
        return "Null context";
    }
    return ctx->error_message;
}
