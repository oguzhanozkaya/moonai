use crate::random_ffi;
use crate::simulation_ffi;

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub grid_size: i32,
    pub predator_count: i32,
    pub prey_count: i32,
    pub food_count: i32,
    pub predator_speed: f32,
    pub prey_speed: f32,
    pub vision_range: f32,
    pub interaction_range: f32,
    pub mate_range: f32,
    pub food_respawn_rate: f32,
    pub energy_drain_per_step: f32,
    pub energy_gain_from_kill: f32,
    pub energy_gain_from_food: f32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub reproduction_energy_threshold: f32,
    pub reproduction_energy_cost: f32,
    pub offspring_initial_energy: f32,
    pub max_age: i32,
    pub mutation_rate: f32,
    pub weight_mutation_power: f32,
    pub add_node_rate: f32,
    pub add_connection_rate: f32,
    pub delete_connection_rate: f32,
    pub max_hidden_nodes: i32,
    pub max_steps: i32,
    pub compatibility_threshold: f32,
    pub compatibility_min_normalization: f32,
    pub c1_excess: f32,
    pub c2_disjoint: f32,
    pub c3_weight: f32,
    pub seed: i32,
    pub report_interval_steps: i32,
}

pub struct Random {
    ptr: *mut random_ffi::CRandom,
}

impl Random {
    pub fn new(seed: i32) -> Self {
        let ptr = unsafe { random_ffi::c_random_create(seed) };
        Self { ptr }
    }

    pub fn next_int(&mut self, min: i32, max: i32) -> i32 {
        unsafe { random_ffi::c_random_next_int(self.ptr, min, max) }
    }

    pub fn next_float(&mut self, min: f32, max: f32) -> f32 {
        unsafe { random_ffi::c_random_next_float(self.ptr, min, max) }
    }

    pub fn next_bool(&mut self, probability: f32) -> bool {
        unsafe { random_ffi::c_random_next_bool(self.ptr, probability) }
    }

    pub fn weighted_select(&mut self, weights: &[f32]) -> i32 {
        let len = weights.len();
        let ptr = weights.as_ptr();
        unsafe { random_ffi::c_random_weighted_select(self.ptr, ptr, len) }
    }
}

impl Drop for Random {
    fn drop(&mut self) {
        unsafe { random_ffi::c_random_destroy(self.ptr) };
    }
}

pub struct SimContext {
    ptr: *mut simulation_ffi::SimulationContext,
}

#[derive(Default)]
pub struct SyncBuffers {
    pub predator_pos_x: Vec<f32>,
    pub predator_pos_y: Vec<f32>,
    pub predator_vel_x: Vec<f32>,
    pub predator_vel_y: Vec<f32>,
    pub predator_energy: Vec<f32>,
    pub predator_age: Vec<i32>,
    pub predator_alive: Vec<u8>,
    pub prey_pos_x: Vec<f32>,
    pub prey_pos_y: Vec<f32>,
    pub prey_vel_x: Vec<f32>,
    pub prey_vel_y: Vec<f32>,
    pub prey_energy: Vec<f32>,
    pub prey_age: Vec<i32>,
    pub prey_alive: Vec<u8>,
}

impl SimContext {
    pub fn new() -> Option<Self> {
        let ptr = unsafe { simulation_ffi::sim_create() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn init(&mut self, config: &SimulationConfig) -> bool {
        unsafe {
            simulation_ffi::sim_init(
                self.ptr,
                config.grid_size,
                config.predator_count,
                config.prey_count,
                config.food_count,
                config.predator_speed,
                config.prey_speed,
                config.vision_range,
                config.interaction_range,
                config.mate_range,
                config.food_respawn_rate,
                config.energy_drain_per_step,
                config.energy_gain_from_kill,
                config.energy_gain_from_food,
                config.initial_energy,
                config.max_energy,
                config.reproduction_energy_threshold,
                config.reproduction_energy_cost,
                config.offspring_initial_energy,
                config.max_age,
                config.mutation_rate,
                config.weight_mutation_power,
                config.add_node_rate,
                config.add_connection_rate,
                config.delete_connection_rate,
                config.max_hidden_nodes,
                config.max_steps,
                config.compatibility_threshold,
                config.compatibility_min_normalization,
                config.c1_excess,
                config.c2_disjoint,
                config.c3_weight,
                config.seed,
                std::ptr::null(),
                config.report_interval_steps,
            ) != 0
        }
    }

    pub fn step(&mut self) -> bool {
        unsafe { simulation_ffi::sim_step(self.ptr) != 0 }
    }

    pub fn prepare_step(&mut self) -> bool {
        unsafe { simulation_ffi::sim_prepare_step(self.ptr) != 0 }
    }

    pub fn run_inference(&mut self) -> bool {
        unsafe { simulation_ffi::sim_run_inference(self.ptr) != 0 }
    }

    pub fn resolve_step(&mut self) -> bool {
        unsafe { simulation_ffi::sim_resolve_step(self.ptr) != 0 }
    }

    pub fn post_step(&mut self) -> bool {
        unsafe { simulation_ffi::sim_post_step(self.ptr) != 0 }
    }

    pub fn refresh_species(&mut self) {
        unsafe { simulation_ffi::sim_refresh_species(self.ptr) }
    }

    pub fn get_step(&self) -> i32 {
        unsafe { simulation_ffi::sim_get_step(self.ptr) }
    }

    pub fn get_predator_count(&self) -> i32 {
        unsafe { simulation_ffi::sim_get_predator_count(self.ptr) }
    }

    pub fn get_prey_count(&self) -> i32 {
        unsafe { simulation_ffi::sim_get_prey_count(self.ptr) }
    }

    pub fn sync_predator_state(&self, buffers: &mut SyncBuffers) {
        unsafe {
            let mut pos_x: *const f32 = std::ptr::null();
            let mut pos_y: *const f32 = std::ptr::null();
            let mut vel_x: *const f32 = std::ptr::null();
            let mut vel_y: *const f32 = std::ptr::null();
            let mut energy: *const f32 = std::ptr::null();
            let mut age: *const i32 = std::ptr::null();
            let mut alive: *const u8 = std::ptr::null();
            let mut count: usize = 0;

            simulation_ffi::sim_get_predator_state(
                self.ptr,
                &mut pos_x,
                &mut pos_y,
                &mut vel_x,
                &mut vel_y,
                &mut energy,
                &mut age,
                &mut alive,
                &mut count,
            );

            buffers.predator_pos_x.resize(count, 0.0);
            buffers.predator_pos_y.resize(count, 0.0);
            buffers.predator_vel_x.resize(count, 0.0);
            buffers.predator_vel_y.resize(count, 0.0);
            buffers.predator_energy.resize(count, 0.0);
            buffers.predator_age.resize(count as usize, 0);
            buffers.predator_alive.resize(count, 0);

            if !pos_x.is_null() {
                std::ptr::copy_nonoverlapping(pos_x, buffers.predator_pos_x.as_mut_ptr(), count);
            }
            if !pos_y.is_null() {
                std::ptr::copy_nonoverlapping(pos_y, buffers.predator_pos_y.as_mut_ptr(), count);
            }
            if !vel_x.is_null() {
                std::ptr::copy_nonoverlapping(vel_x, buffers.predator_vel_x.as_mut_ptr(), count);
            }
            if !vel_y.is_null() {
                std::ptr::copy_nonoverlapping(vel_y, buffers.predator_vel_y.as_mut_ptr(), count);
            }
            if !energy.is_null() {
                std::ptr::copy_nonoverlapping(energy, buffers.predator_energy.as_mut_ptr(), count);
            }
            if !age.is_null() {
                std::ptr::copy_nonoverlapping(age, buffers.predator_age.as_mut_ptr(), count);
            }
            if !alive.is_null() {
                std::ptr::copy_nonoverlapping(alive, buffers.predator_alive.as_mut_ptr(), count);
            }
        }
    }

    pub fn sync_prey_state(&self, buffers: &mut SyncBuffers) {
        unsafe {
            let mut pos_x: *const f32 = std::ptr::null();
            let mut pos_y: *const f32 = std::ptr::null();
            let mut vel_x: *const f32 = std::ptr::null();
            let mut vel_y: *const f32 = std::ptr::null();
            let mut energy: *const f32 = std::ptr::null();
            let mut age: *const i32 = std::ptr::null();
            let mut alive: *const u8 = std::ptr::null();
            let mut count: usize = 0;

            simulation_ffi::sim_get_prey_state(
                self.ptr,
                &mut pos_x,
                &mut pos_y,
                &mut vel_x,
                &mut vel_y,
                &mut energy,
                &mut age,
                &mut alive,
                &mut count,
            );

            buffers.prey_pos_x.resize(count, 0.0);
            buffers.prey_pos_y.resize(count, 0.0);
            buffers.prey_vel_x.resize(count, 0.0);
            buffers.prey_vel_y.resize(count, 0.0);
            buffers.prey_energy.resize(count, 0.0);
            buffers.prey_age.resize(count, 0);
            buffers.prey_alive.resize(count, 0);

            if !pos_x.is_null() {
                std::ptr::copy_nonoverlapping(pos_x, buffers.prey_pos_x.as_mut_ptr(), count);
            }
            if !pos_y.is_null() {
                std::ptr::copy_nonoverlapping(pos_y, buffers.prey_pos_y.as_mut_ptr(), count);
            }
            if !vel_x.is_null() {
                std::ptr::copy_nonoverlapping(vel_x, buffers.prey_vel_x.as_mut_ptr(), count);
            }
            if !vel_y.is_null() {
                std::ptr::copy_nonoverlapping(vel_y, buffers.prey_vel_y.as_mut_ptr(), count);
            }
            if !energy.is_null() {
                std::ptr::copy_nonoverlapping(energy, buffers.prey_energy.as_mut_ptr(), count);
            }
            if !age.is_null() {
                std::ptr::copy_nonoverlapping(age, buffers.prey_age.as_mut_ptr(), count);
            }
            if !alive.is_null() {
                std::ptr::copy_nonoverlapping(alive, buffers.prey_alive.as_mut_ptr(), count);
            }
        }
    }

    pub fn compact_predator(&mut self) {
        unsafe { simulation_ffi::sim_compact_predator(self.ptr) }
    }

    pub fn compact_prey(&mut self) {
        unsafe { simulation_ffi::sim_compact_prey(self.ptr) }
    }

    pub fn error(&self) -> String {
        unsafe {
            let ptr = simulation_ffi::sim_get_error(self.ptr);
            if ptr.is_null() {
                String::new()
            } else {
                let c_str = std::ffi::CStr::from_ptr(ptr);
                c_str.to_string_lossy().into_owned()
            }
        }
    }
}

impl Default for SimContext {
    fn default() -> Self {
        Self::new().expect("Failed to create simulation context")
    }
}

impl Drop for SimContext {
    fn drop(&mut self) {
        unsafe { simulation_ffi::sim_destroy(self.ptr) };
    }
}

impl Default for Random {
    fn default() -> Self {
        Self::new(0)
    }
}