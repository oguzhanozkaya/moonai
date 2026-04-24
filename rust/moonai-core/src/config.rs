use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default = "default_grid_size")]
    pub grid_size: i32,
    #[serde(default = "default_predator_count")]
    pub predator_count: i32,
    #[serde(default = "default_prey_count")]
    pub prey_count: i32,
    #[serde(default = "default_food_count")]
    pub food_count: i32,
    #[serde(default = "default_predator_speed")]
    pub predator_speed: f32,
    #[serde(default = "default_prey_speed")]
    pub prey_speed: f32,
    #[serde(default = "default_vision_range")]
    pub vision_range: f32,
    #[serde(default = "default_interaction_range")]
    pub interaction_range: f32,
    #[serde(default = "default_mate_range")]
    pub mate_range: f32,
    #[serde(default = "default_food_respawn_rate")]
    pub food_respawn_rate: f32,
    #[serde(default = "default_energy_drain_per_step")]
    pub energy_drain_per_step: f32,
    #[serde(default = "default_energy_gain_from_kill")]
    pub energy_gain_from_kill: f32,
    #[serde(default = "default_energy_gain_from_food")]
    pub energy_gain_from_food: f32,
    #[serde(default = "default_initial_energy")]
    pub initial_energy: f32,
    #[serde(default = "default_max_energy")]
    pub max_energy: f32,
    #[serde(default = "default_reproduction_energy_threshold")]
    pub reproduction_energy_threshold: f32,
    #[serde(default = "default_reproduction_energy_cost")]
    pub reproduction_energy_cost: f32,
    #[serde(default = "default_offspring_initial_energy")]
    pub offspring_initial_energy: f32,
    #[serde(default = "default_max_age")]
    pub max_age: i32,
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f32,
    #[serde(default = "default_weight_mutation_power")]
    pub weight_mutation_power: f32,
    #[serde(default = "default_add_node_rate")]
    pub add_node_rate: f32,
    #[serde(default = "default_add_connection_rate")]
    pub add_connection_rate: f32,
    #[serde(default = "default_delete_connection_rate")]
    pub delete_connection_rate: f32,
    #[serde(default = "default_max_hidden_nodes")]
    pub max_hidden_nodes: i32,
    #[serde(default = "default_max_steps")]
    pub max_steps: i32,
    #[serde(default = "default_compatibility_threshold")]
    pub compatibility_threshold: f32,
    #[serde(default = "default_compatibility_min_normalization")]
    pub compatibility_min_normalization: f32,
    #[serde(default = "default_c1_excess")]
    pub c1_excess: f32,
    #[serde(default = "default_c2_disjoint")]
    pub c2_disjoint: f32,
    #[serde(default = "default_c3_weight")]
    pub c3_weight: f32,
    #[serde(default = "default_seed")]
    pub seed: i32,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default = "default_report_interval_steps")]
    pub report_interval_steps: i32,
}

fn default_grid_size() -> i32 { 3600 }
fn default_predator_count() -> i32 { 24000 }
fn default_prey_count() -> i32 { 96000 }
fn default_food_count() -> i32 { 240000 }
fn default_predator_speed() -> f32 { 1.0 }
fn default_prey_speed() -> f32 { 1.006 }
fn default_vision_range() -> f32 { 12.0 }
fn default_interaction_range() -> f32 { 1.0 }
fn default_mate_range() -> f32 { 6.0 }
fn default_food_respawn_rate() -> f32 { 0.006 }
fn default_energy_drain_per_step() -> f32 { 0.001 }
fn default_energy_gain_from_kill() -> f32 { 0.24 }
fn default_energy_gain_from_food() -> f32 { 0.24 }
fn default_initial_energy() -> f32 { 0.36 }
fn default_max_energy() -> f32 { 2.0 }
fn default_reproduction_energy_threshold() -> f32 { 1.0 }
fn default_reproduction_energy_cost() -> f32 { 0.18 }
fn default_offspring_initial_energy() -> f32 { 0.36 }
fn default_max_age() -> i32 { 10000 }
fn default_mutation_rate() -> f32 { 0.30 }
fn default_weight_mutation_power() -> f32 { 0.30 }
fn default_add_node_rate() -> f32 { 0.12 }
fn default_add_connection_rate() -> f32 { 0.60 }
fn default_delete_connection_rate() -> f32 { 0.00000006 }
fn default_max_hidden_nodes() -> i32 { 1200 }
fn default_max_steps() -> i32 { 0 }
fn default_compatibility_threshold() -> f32 { 60.0 }
fn default_compatibility_min_normalization() -> f32 { 240.0 }
fn default_c1_excess() -> f32 { 1.0 }
fn default_c2_disjoint() -> f32 { 1.0 }
fn default_c3_weight() -> f32 { 0.4 }
fn default_seed() -> i32 { 67 }
fn default_output_dir() -> String { String::from("output/experiments") }
fn default_report_interval_steps() -> i32 { 1000 }

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: default_grid_size(),
            predator_count: default_predator_count(),
            prey_count: default_prey_count(),
            food_count: default_food_count(),
            predator_speed: default_predator_speed(),
            prey_speed: default_prey_speed(),
            vision_range: default_vision_range(),
            interaction_range: default_interaction_range(),
            mate_range: default_mate_range(),
            food_respawn_rate: default_food_respawn_rate(),
            energy_drain_per_step: default_energy_drain_per_step(),
            energy_gain_from_kill: default_energy_gain_from_kill(),
            energy_gain_from_food: default_energy_gain_from_food(),
            initial_energy: default_initial_energy(),
            max_energy: default_max_energy(),
            reproduction_energy_threshold: default_reproduction_energy_threshold(),
            reproduction_energy_cost: default_reproduction_energy_cost(),
            offspring_initial_energy: default_offspring_initial_energy(),
            max_age: default_max_age(),
            mutation_rate: default_mutation_rate(),
            weight_mutation_power: default_weight_mutation_power(),
            add_node_rate: default_add_node_rate(),
            add_connection_rate: default_add_connection_rate(),
            delete_connection_rate: default_delete_connection_rate(),
            max_hidden_nodes: default_max_hidden_nodes(),
            max_steps: default_max_steps(),
            compatibility_threshold: default_compatibility_threshold(),
            compatibility_min_normalization: default_compatibility_min_normalization(),
            c1_excess: default_c1_excess(),
            c2_disjoint: default_c2_disjoint(),
            c3_weight: default_c3_weight(),
            seed: default_seed(),
            output_dir: default_output_dir(),
            report_interval_steps: default_report_interval_steps(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConfigError {
    pub field: String,
    pub message: String,
}

pub fn validate_config(config: &SimulationConfig) -> Vec<ConfigError> {
    let mut errors = Vec::new();

    let check = |errors: &mut Vec<ConfigError>, cond: bool, field: &str, msg: &str| {
        if !cond {
            errors.push(ConfigError {
                field: field.to_string(),
                message: msg.to_string(),
            });
        }
    };

    check(&mut errors, config.grid_size >= 100, "grid_size", "must be >= 100");
    check(&mut errors, config.grid_size <= 20000, "grid_size", "must be <= 20000");
    check(&mut errors, config.predator_count >= 1, "predator_count", "must be >= 1");
    check(&mut errors, config.prey_count >= 1, "prey_count", "must be >= 1");
    check(&mut errors, config.predator_count + config.prey_count <= 1000000, "population", "total population must be <= 1000000");
    check(&mut errors, config.predator_speed > 0.0, "predator_speed", "must be > 0");
    check(&mut errors, config.prey_speed > 0.0, "prey_speed", "must be > 0");
    check(&mut errors, config.vision_range > 0.0, "vision_range", "must be > 0");
    check(&mut errors, config.interaction_range > 0.0, "interaction_range", "must be > 0");
    check(&mut errors, config.interaction_range < config.vision_range, "interaction_range", "must be less than vision_range");
    check(&mut errors, config.initial_energy > 0.0, "initial_energy", "must be > 0");
    check(&mut errors, config.max_energy > 0.0, "max_energy", "must be > 0");
    check(&mut errors, config.initial_energy <= config.max_energy, "initial_energy", "must be <= max_energy");
    check(&mut errors, config.energy_drain_per_step >= 0.0, "energy_drain_per_step", "must be >= 0");
    check(&mut errors, config.food_count >= 0, "food_count", "must be >= 0");
    check(&mut errors, config.food_respawn_rate >= 0.0 && config.food_respawn_rate <= 1.0, "food_respawn_rate", "must be in [0, 1]");
    check(&mut errors, config.mutation_rate >= 0.0 && config.mutation_rate <= 1.0, "mutation_rate", "must be in [0, 1]");
    check(&mut errors, config.add_node_rate >= 0.0 && config.add_node_rate <= 1.0, "add_node_rate", "must be in [0, 1]");
    check(&mut errors, config.add_connection_rate >= 0.0 && config.add_connection_rate <= 1.0, "add_connection_rate", "must be in [0, 1]");
    check(&mut errors, config.delete_connection_rate >= 0.0 && config.delete_connection_rate <= 1.0, "delete_connection_rate", "must be in [0, 1]");
    check(&mut errors, config.weight_mutation_power > 0.0, "weight_mutation_power", "must be > 0");
    check(&mut errors, config.max_age >= 0, "max_age", "must be >= 0 (0 = infinite)");
    check(&mut errors, config.max_steps >= 0, "max_steps", "must be >= 0 (0 = infinite)");
    check(&mut errors, config.compatibility_threshold > 0.0, "compatibility_threshold", "must be > 0");
    check(&mut errors, config.compatibility_min_normalization >= 1.0, "compatibility_min_normalization", "must be >= 1");
    check(&mut errors, config.report_interval_steps >= 1, "report_interval_steps", "must be >= 1");
    check(&mut errors, config.mate_range > 0.0, "mate_range", "must be > 0");
    check(&mut errors, config.reproduction_energy_threshold > 0.0, "reproduction_energy_threshold", "must be > 0");
    check(&mut errors, config.reproduction_energy_threshold <= config.max_energy, "reproduction_energy_threshold", "must be <= max_energy");
    check(&mut errors, config.reproduction_energy_cost > 0.0, "reproduction_energy_cost", "must be > 0");
    check(&mut errors, config.offspring_initial_energy > 0.0, "offspring_initial_energy", "must be > 0");
    check(&mut errors, config.offspring_initial_energy <= config.max_energy, "offspring_initial_energy", "must be <= max_energy");

    errors
}

pub fn config_to_json(config: &SimulationConfig) -> serde_json::Value {
    serde_json::json!({
        "grid_size": config.grid_size,
        "predator_count": config.predator_count,
        "prey_count": config.prey_count,
        "predator_speed": config.predator_speed,
        "prey_speed": config.prey_speed,
        "vision_range": config.vision_range,
        "interaction_range": config.interaction_range,
        "initial_energy": config.initial_energy,
        "max_energy": config.max_energy,
        "energy_drain_per_step": config.energy_drain_per_step,
        "energy_gain_from_kill": config.energy_gain_from_kill,
        "energy_gain_from_food": config.energy_gain_from_food,
        "food_count": config.food_count,
        "food_respawn_rate": config.food_respawn_rate,
        "mutation_rate": config.mutation_rate,
        "weight_mutation_power": config.weight_mutation_power,
        "add_node_rate": config.add_node_rate,
        "add_connection_rate": config.add_connection_rate,
        "delete_connection_rate": config.delete_connection_rate,
        "max_hidden_nodes": config.max_hidden_nodes,
        "max_age": config.max_age,
        "max_steps": config.max_steps,
        "compatibility_threshold": config.compatibility_threshold,
        "compatibility_min_normalization": config.compatibility_min_normalization,
        "c1_excess": config.c1_excess,
        "c2_disjoint": config.c2_disjoint,
        "c3_weight": config.c3_weight,
        "seed": config.seed,
        "output_dir": config.output_dir,
        "report_interval_steps": config.report_interval_steps,
        "mate_range": config.mate_range,
        "reproduction_energy_threshold": config.reproduction_energy_threshold,
        "reproduction_energy_cost": config.reproduction_energy_cost,
        "offspring_initial_energy": config.offspring_initial_energy,
    })
}

pub fn save_config(config: &SimulationConfig, filepath: &Path) -> std::io::Result<()> {
    let j = config_to_json(config);
    let json_str = serde_json::to_string_pretty(&j).unwrap();
    fs::write(filepath, json_str)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimulationConfig::default();
        assert_eq!(config.grid_size, 3600);
        assert_eq!(config.predator_count, 24000);
        assert_eq!(config.prey_count, 96000);
        assert_eq!(config.seed, 67);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = SimulationConfig::default();
        let errors = validate_config(&config);
        assert!(errors.is_empty(), "Expected no errors, got {:?}", errors);
    }

    #[test]
    fn test_validate_invalid_grid_size() {
        let mut config = SimulationConfig::default();
        config.grid_size = 50;
        let errors = validate_config(&config);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.field == "grid_size"));
    }
}
