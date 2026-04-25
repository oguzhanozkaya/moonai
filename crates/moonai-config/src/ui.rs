use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_predator_radius")]
    pub predator_radius: f32,
    #[serde(default = "default_prey_radius")]
    pub prey_radius: f32,
    #[serde(default = "default_food_radius")]
    pub food_radius: f32,
    #[serde(default = "default_predator_color")]
    pub predator_color: [f32; 3],
    #[serde(default = "default_prey_color")]
    pub prey_color: [f32; 3],
    #[serde(default = "default_food_color")]
    pub food_color: [f32; 3],
}

fn default_predator_radius() -> f32 {
    4.0
}
fn default_prey_radius() -> f32 {
    3.0
}
fn default_food_radius() -> f32 {
    2.0
}
fn default_predator_color() -> [f32; 3] {
    [0.8, 0.2, 0.2]
}
fn default_prey_color() -> [f32; 3] {
    [0.2, 0.8, 0.3]
}
fn default_food_color() -> [f32; 3] {
    [0.2, 0.6, 0.8]
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            predator_radius: default_predator_radius(),
            prey_radius: default_prey_radius(),
            food_radius: default_food_radius(),
            predator_color: default_predator_color(),
            prey_color: default_prey_color(),
            food_color: default_food_color(),
        }
    }
}
