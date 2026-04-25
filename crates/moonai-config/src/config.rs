use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default = "default_predator_count")]
    pub predator_count: u32,
    #[serde(default = "default_prey_count")]
    pub prey_count: u32,
    #[serde(default = "default_grid_size")]
    pub grid_size: u32,
    #[serde(default = "default_food_count")]
    pub food_count: u32,
    #[serde(default = "default_max_steps")]
    pub max_steps: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_report_interval")]
    pub report_interval: u32,
}

fn default_predator_count() -> u32 {
    500
}
fn default_prey_count() -> u32 {
    1500
}
fn default_grid_size() -> u32 {
    3000
}
fn default_food_count() -> u32 {
    500
}
fn default_max_steps() -> u64 {
    u64::MAX
}
fn default_seed() -> u64 {
    42
}
fn default_report_interval() -> u32 {
    1500
}
