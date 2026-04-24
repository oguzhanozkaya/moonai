#[cxx::bridge]
mod random_ffi {
    unsafe extern "C++" {
        include!("random.hpp");

        type CRandom;

        unsafe fn c_random_create(seed: i32) -> *mut CRandom;
        unsafe fn c_random_destroy(rng: *mut CRandom);
        unsafe fn c_random_next_int(rng: *mut CRandom, min: i32, max: i32) -> i32;
        unsafe fn c_random_next_float(rng: *mut CRandom, min: f32, max: f32) -> f32;
        unsafe fn c_random_next_bool(rng: *mut CRandom, probability: f32) -> bool;
        unsafe fn c_random_weighted_select(rng: *mut CRandom, weights: *const f32, len: usize) -> i32;
    }
}

#[cxx::bridge]
mod simulation_ffi {
    unsafe extern "C++" {
        include!("simulation.hpp");

        type SimulationContext;

        unsafe fn gpu_sim_is_available() -> i32;

        unsafe fn sim_create() -> *mut SimulationContext;
        unsafe fn sim_destroy(ctx: *mut SimulationContext);

        unsafe fn sim_init(
            ctx: *mut SimulationContext,
            grid_size: i32,
            predator_count: i32,
            prey_count: i32,
            food_count: i32,
            predator_speed: f32,
            prey_speed: f32,
            vision_range: f32,
            interaction_range: f32,
            mate_range: f32,
            food_respawn_rate: f32,
            energy_drain_per_step: f32,
            energy_gain_from_kill: f32,
            energy_gain_from_food: f32,
            initial_energy: f32,
            max_energy: f32,
            reproduction_energy_threshold: f32,
            reproduction_energy_cost: f32,
            offspring_initial_energy: f32,
            max_age: i32,
            mutation_rate: f32,
            weight_mutation_power: f32,
            add_node_rate: f32,
            add_connection_rate: f32,
            delete_connection_rate: f32,
            max_hidden_nodes: i32,
            max_steps: i32,
            compatibility_threshold: f32,
            compatibility_min_normalization: f32,
            c1_excess: f32,
            c2_disjoint: f32,
            c3_weight: f32,
            seed: i32,
            output_dir: *const i8,
            report_interval_steps: i32,
        ) -> i32;

        unsafe fn sim_step(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_prepare_step(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_resolve_step(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_run_inference(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_post_step(ctx: *mut SimulationContext) -> i32;

        unsafe fn sim_refresh_species(ctx: *mut SimulationContext);

        unsafe fn sim_get_predator_state(
            ctx: *mut SimulationContext,
            out_pos_x: *mut *const f32,
            out_pos_y: *mut *const f32,
            out_vel_x: *mut *const f32,
            out_vel_y: *mut *const f32,
            out_energy: *mut *const f32,
            out_age: *mut *const i32,
            out_alive: *mut *const u8,
            out_count: *mut usize,
        );

        unsafe fn sim_get_prey_state(
            ctx: *mut SimulationContext,
            out_pos_x: *mut *const f32,
            out_pos_y: *mut *const f32,
            out_vel_x: *mut *const f32,
            out_vel_y: *mut *const f32,
            out_energy: *mut *const f32,
            out_age: *mut *const i32,
            out_alive: *mut *const u8,
            out_count: *mut usize,
        );

        unsafe fn sim_get_food_state(
            ctx: *mut SimulationContext,
            out_pos_x: *mut *const f32,
            out_pos_y: *mut *const f32,
            out_active: *mut *const u8,
            out_count: *mut usize,
        );

        unsafe fn sim_get_step(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_get_predator_count(ctx: *mut SimulationContext) -> i32;
        unsafe fn sim_get_prey_count(ctx: *mut SimulationContext) -> i32;

        unsafe fn sim_set_predator_alive(ctx: *mut SimulationContext, idx: usize, alive: u8);
        unsafe fn sim_set_prey_alive(ctx: *mut SimulationContext, idx: usize, alive: u8);

        unsafe fn sim_compact_predator(ctx: *mut SimulationContext);
        unsafe fn sim_compact_prey(ctx: *mut SimulationContext);

        unsafe fn sim_get_error(ctx: *mut SimulationContext) -> *const i8;
    }
}

mod ffi;

pub use ffi::Random;
pub use ffi::SimContext;
pub use ffi::SimulationConfig;
pub use ffi::SyncBuffers;
pub use simulation_ffi::gpu_sim_is_available;
pub use simulation_ffi::sim_create;
pub use simulation_ffi::sim_destroy;
pub use simulation_ffi::sim_init;
pub use simulation_ffi::sim_step;
pub use simulation_ffi::sim_prepare_step;
pub use simulation_ffi::sim_resolve_step;
pub use simulation_ffi::sim_run_inference;
pub use simulation_ffi::sim_post_step;
pub use simulation_ffi::sim_refresh_species;
pub use simulation_ffi::sim_get_predator_state;
pub use simulation_ffi::sim_get_prey_state;
pub use simulation_ffi::sim_get_food_state;
pub use simulation_ffi::sim_get_step;
pub use simulation_ffi::sim_get_predator_count;
pub use simulation_ffi::sim_get_prey_count;
pub use simulation_ffi::sim_set_predator_alive;
pub use simulation_ffi::sim_set_prey_alive;
pub use simulation_ffi::sim_compact_predator;
pub use simulation_ffi::sim_compact_prey;
pub use simulation_ffi::sim_get_error;