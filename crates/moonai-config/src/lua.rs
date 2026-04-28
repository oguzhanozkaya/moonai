use mlua::Lua;
use std::collections::HashMap;

use crate::{ConfigError, SimulationConfig};

pub fn load_config(path: &str) -> Result<HashMap<String, SimulationConfig>, ConfigError> {
    let lua = Lua::new();
    let defaults = SimulationConfig::default();

    let globals = lua.globals();
    let moonai_defaults = lua.create_table().map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    moonai_defaults.set("grid_size", defaults.grid_size as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("predator_count", defaults.predator_count as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("prey_count", defaults.prey_count as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("food_count", defaults.food_count as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("predator_speed", defaults.predator_speed as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("prey_speed", defaults.prey_speed as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("vision_range", defaults.vision_range as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("interaction_range", defaults.interaction_range as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("mate_range", defaults.mate_range as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("food_respawn_rate", defaults.food_respawn_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_drain_per_tick", defaults.energy_drain_per_tick as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_gain_from_kill", defaults.energy_gain_from_kill as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("energy_gain_from_food", defaults.energy_gain_from_food as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("initial_energy", defaults.initial_energy as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_energy", defaults.max_energy as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("reproduction_energy_threshold", defaults.reproduction_energy_threshold as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("reproduction_energy_cost", defaults.reproduction_energy_cost as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("offspring_initial_energy", defaults.offspring_initial_energy as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_age", defaults.max_age as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("mutation_rate", defaults.mutation_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("weight_mutation_power", defaults.weight_mutation_power as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("add_node_rate", defaults.add_node_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("add_connection_rate", defaults.add_connection_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("delete_connection_rate", defaults.delete_connection_rate as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("max_hidden_nodes", defaults.max_hidden_nodes as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("max_ticks", defaults.max_ticks as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("compatibility_threshold", defaults.compatibility_threshold as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("compatibility_min_normalization", defaults.compatibility_min_normalization as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("c1_excess", defaults.c1_excess as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("c2_disjoint", defaults.c2_disjoint as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("c3_weight", defaults.c3_weight as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults.set("seed", defaults.seed as f64).map_err(|e| ConfigError::LuaParse(e.to_string()))?;
    moonai_defaults
        .set("report_interval_ticks", defaults.report_interval_ticks as f64)
        .map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    globals.set("moonai_defaults", moonai_defaults).map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
    let experiments_table: mlua::Table = lua.load(&content).eval().map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    let mut experiments = HashMap::new();
    for pair in experiments_table.pairs::<String, mlua::Table>() {
        let (name, cfg_table) = pair.map_err(|e| ConfigError::LuaParse(e.to_string()))?;
        let config = table_to_config(&cfg_table)?;
        experiments.insert(name, config);
    }

    Ok(experiments)
}

fn table_to_config(table: &mlua::Table) -> Result<SimulationConfig, ConfigError> {
    let mut config = SimulationConfig::default();

    if let Ok(v) = table.get::<_, f64>("grid_size") {
        config.grid_size = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("predator_count") {
        config.predator_count = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("prey_count") {
        config.prey_count = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("food_count") {
        config.food_count = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("predator_speed") {
        config.predator_speed = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("prey_speed") {
        config.prey_speed = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("vision_range") {
        config.vision_range = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("interaction_range") {
        config.interaction_range = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("mate_range") {
        config.mate_range = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("food_respawn_rate") {
        config.food_respawn_rate = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("energy_drain_per_tick") {
        config.energy_drain_per_tick = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("energy_gain_from_kill") {
        config.energy_gain_from_kill = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("energy_gain_from_food") {
        config.energy_gain_from_food = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("initial_energy") {
        config.initial_energy = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("max_energy") {
        config.max_energy = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("reproduction_energy_threshold") {
        config.reproduction_energy_threshold = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("reproduction_energy_cost") {
        config.reproduction_energy_cost = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("offspring_initial_energy") {
        config.offspring_initial_energy = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("max_age") {
        config.max_age = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("mutation_rate") {
        config.mutation_rate = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("weight_mutation_power") {
        config.weight_mutation_power = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("add_node_rate") {
        config.add_node_rate = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("add_connection_rate") {
        config.add_connection_rate = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("delete_connection_rate") {
        config.delete_connection_rate = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("max_hidden_nodes") {
        config.max_hidden_nodes = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("max_ticks") {
        config.max_ticks = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("compatibility_threshold") {
        config.compatibility_threshold = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("compatibility_min_normalization") {
        config.compatibility_min_normalization = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("c1_excess") {
        config.c1_excess = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("c2_disjoint") {
        config.c2_disjoint = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("c3_weight") {
        config.c3_weight = v as f32;
    }
    if let Ok(v) = table.get::<_, f64>("seed") {
        config.seed = v as i32;
    }
    if let Ok(v) = table.get::<_, f64>("report_interval_ticks") {
        config.report_interval_ticks = v as i32;
    }

    Ok(config)
}
