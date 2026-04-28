use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default = "grid_size_default")]
    pub grid_size: i32,
    #[serde(default = "predator_count_default")]
    pub predator_count: i32,
    #[serde(default = "prey_count_default")]
    pub prey_count: i32,
    #[serde(default = "food_count_default")]
    pub food_count: i32,
    #[serde(default = "predator_speed_default")]
    pub predator_speed: f32,
    #[serde(default = "prey_speed_default")]
    pub prey_speed: f32,
    #[serde(default = "vision_range_default")]
    pub vision_range: f32,
    #[serde(default = "interaction_range_default")]
    pub interaction_range: f32,
    #[serde(default = "mate_range_default")]
    pub mate_range: f32,
    #[serde(default = "food_respawn_rate_default")]
    pub food_respawn_rate: f32,
    #[serde(default = "energy_drain_per_tick_default")]
    pub energy_drain_per_tick: f32,
    #[serde(default = "energy_gain_from_kill_default")]
    pub energy_gain_from_kill: f32,
    #[serde(default = "energy_gain_from_food_default")]
    pub energy_gain_from_food: f32,
    #[serde(default = "initial_energy_default")]
    pub initial_energy: f32,
    #[serde(default = "max_energy_default")]
    pub max_energy: f32,
    #[serde(default = "reproduction_energy_threshold_default")]
    pub reproduction_energy_threshold: f32,
    #[serde(default = "reproduction_energy_cost_default")]
    pub reproduction_energy_cost: f32,
    #[serde(default = "offspring_initial_energy_default")]
    pub offspring_initial_energy: f32,
    #[serde(default = "max_age_default")]
    pub max_age: i32,
    #[serde(default = "mutation_rate_default")]
    pub mutation_rate: f32,
    #[serde(default = "weight_mutation_power_default")]
    pub weight_mutation_power: f32,
    #[serde(default = "add_node_rate_default")]
    pub add_node_rate: f32,
    #[serde(default = "add_connection_rate_default")]
    pub add_connection_rate: f32,
    #[serde(default = "delete_connection_rate_default")]
    pub delete_connection_rate: f32,
    #[serde(default = "max_hidden_nodes_default")]
    pub max_hidden_nodes: i32,
    #[serde(default = "max_ticks_default")]
    pub max_ticks: i32,
    #[serde(default = "compatibility_threshold_default")]
    pub compatibility_threshold: f32,
    #[serde(default = "compatibility_min_normalization_default")]
    pub compatibility_min_normalization: f32,
    #[serde(default = "c1_excess_default")]
    pub c1_excess: f32,
    #[serde(default = "c2_disjoint_default")]
    pub c2_disjoint: f32,
    #[serde(default = "c3_weight_default")]
    pub c3_weight: f32,
    #[serde(default = "seed_default")]
    pub seed: i32,
    #[serde(default = "output_dir_default")]
    pub output_dir: String,
    #[serde(default = "report_interval_ticks_default")]
    pub report_interval_ticks: i32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: grid_size_default(),
            predator_count: predator_count_default(),
            prey_count: prey_count_default(),
            food_count: food_count_default(),
            predator_speed: predator_speed_default(),
            prey_speed: prey_speed_default(),
            vision_range: vision_range_default(),
            interaction_range: interaction_range_default(),
            mate_range: mate_range_default(),
            food_respawn_rate: food_respawn_rate_default(),
            energy_drain_per_tick: energy_drain_per_tick_default(),
            energy_gain_from_kill: energy_gain_from_kill_default(),
            energy_gain_from_food: energy_gain_from_food_default(),
            initial_energy: initial_energy_default(),
            max_energy: max_energy_default(),
            reproduction_energy_threshold: reproduction_energy_threshold_default(),
            reproduction_energy_cost: reproduction_energy_cost_default(),
            offspring_initial_energy: offspring_initial_energy_default(),
            max_age: max_age_default(),
            mutation_rate: mutation_rate_default(),
            weight_mutation_power: weight_mutation_power_default(),
            add_node_rate: add_node_rate_default(),
            add_connection_rate: add_connection_rate_default(),
            delete_connection_rate: delete_connection_rate_default(),
            max_hidden_nodes: max_hidden_nodes_default(),
            max_ticks: max_ticks_default(),
            compatibility_threshold: compatibility_threshold_default(),
            compatibility_min_normalization: compatibility_min_normalization_default(),
            c1_excess: c1_excess_default(),
            c2_disjoint: c2_disjoint_default(),
            c3_weight: c3_weight_default(),
            seed: seed_default(),
            output_dir: output_dir_default(),
            report_interval_ticks: report_interval_ticks_default(),
        }
    }
}

fn grid_size_default() -> i32 {
    3600
}
fn predator_count_default() -> i32 {
    24000
}
fn prey_count_default() -> i32 {
    96000
}
fn food_count_default() -> i32 {
    240000
}
fn predator_speed_default() -> f32 {
    1.0
}
fn prey_speed_default() -> f32 {
    1.006
}
fn vision_range_default() -> f32 {
    12.0
}
fn interaction_range_default() -> f32 {
    1.0
}
fn mate_range_default() -> f32 {
    6.0
}
fn food_respawn_rate_default() -> f32 {
    0.006
}
fn energy_drain_per_tick_default() -> f32 {
    0.001
}
fn energy_gain_from_kill_default() -> f32 {
    0.24
}
fn energy_gain_from_food_default() -> f32 {
    0.24
}
fn initial_energy_default() -> f32 {
    0.36
}
fn max_energy_default() -> f32 {
    2.0
}
fn reproduction_energy_threshold_default() -> f32 {
    1.0
}
fn reproduction_energy_cost_default() -> f32 {
    0.18
}
fn offspring_initial_energy_default() -> f32 {
    0.36
}
fn max_age_default() -> i32 {
    10000
}
fn mutation_rate_default() -> f32 {
    0.30
}
fn weight_mutation_power_default() -> f32 {
    0.30
}
fn add_node_rate_default() -> f32 {
    0.12
}
fn add_connection_rate_default() -> f32 {
    0.60
}
fn delete_connection_rate_default() -> f32 {
    0.00000006
}
fn max_hidden_nodes_default() -> i32 {
    1200
}
fn max_ticks_default() -> i32 {
    0
}
fn compatibility_threshold_default() -> f32 {
    60.0
}
fn compatibility_min_normalization_default() -> f32 {
    240.0
}
fn c1_excess_default() -> f32 {
    1.0
}
fn c2_disjoint_default() -> f32 {
    1.0
}
fn c3_weight_default() -> f32 {
    0.4
}
fn seed_default() -> i32 {
    67
}
fn output_dir_default() -> String {
    String::from("output/experiments")
}
fn report_interval_ticks_default() -> i32 {
    1000
}
