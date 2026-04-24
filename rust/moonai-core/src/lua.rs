use mlua::{Lua, Table, StdLib};
use std::collections::HashMap;
use std::path::Path;

use crate::config::SimulationConfig;

fn default_config() -> SimulationConfig {
    SimulationConfig::default()
}

fn inject_defaults(lua: &Lua) -> mlua::Result<()> {
    let defaults = default_config();
    let table = lua.create_table()?;

    table.set("grid_size", defaults.grid_size)?;
    table.set("predator_count", defaults.predator_count)?;
    table.set("prey_count", defaults.prey_count)?;
    table.set("predator_speed", defaults.predator_speed)?;
    table.set("prey_speed", defaults.prey_speed)?;
    table.set("vision_range", defaults.vision_range)?;
    table.set("interaction_range", defaults.interaction_range)?;
    table.set("initial_energy", defaults.initial_energy)?;
    table.set("max_energy", defaults.max_energy)?;
    table.set("energy_drain_per_step", defaults.energy_drain_per_step)?;
    table.set("energy_gain_from_kill", defaults.energy_gain_from_kill)?;
    table.set("energy_gain_from_food", defaults.energy_gain_from_food)?;
    table.set("food_count", defaults.food_count)?;
    table.set("food_respawn_rate", defaults.food_respawn_rate)?;
    table.set("mutation_rate", defaults.mutation_rate)?;
    table.set("weight_mutation_power", defaults.weight_mutation_power)?;
    table.set("add_node_rate", defaults.add_node_rate)?;
    table.set("add_connection_rate", defaults.add_connection_rate)?;
    table.set("delete_connection_rate", defaults.delete_connection_rate)?;
    table.set("max_hidden_nodes", defaults.max_hidden_nodes)?;
    table.set("max_age", defaults.max_age)?;
    table.set("max_steps", defaults.max_steps)?;
    table.set("compatibility_threshold", defaults.compatibility_threshold)?;
    table.set("compatibility_min_normalization", defaults.compatibility_min_normalization)?;
    table.set("c1_excess", defaults.c1_excess)?;
    table.set("c2_disjoint", defaults.c2_disjoint)?;
    table.set("c3_weight", defaults.c3_weight)?;
    table.set("seed", defaults.seed)?;
    table.set("output_dir", defaults.output_dir.as_str())?;
    table.set("report_interval_steps", defaults.report_interval_steps)?;
    table.set("mate_range", defaults.mate_range)?;
    table.set("reproduction_energy_threshold", defaults.reproduction_energy_threshold)?;
    table.set("reproduction_energy_cost", defaults.reproduction_energy_cost)?;
    table.set("offspring_initial_energy", defaults.offspring_initial_energy)?;

    lua.globals().set("moonai_defaults", table)?;
    Ok(())
}

fn get_i32(tbl: &Table, key: &str) -> Option<i32> {
    tbl.get(key).ok()
}

fn get_f32(tbl: &Table, key: &str) -> Option<f32> {
    tbl.get(key).ok()
}

fn get_string(tbl: &Table, key: &str) -> Option<String> {
    tbl.get(key).ok()
}

fn table_to_config(tbl: &Table) -> SimulationConfig {
    let mut config = SimulationConfig::default();

    if let Some(v) = get_i32(tbl, "grid_size") { config.grid_size = v; }
    if let Some(v) = get_i32(tbl, "predator_count") { config.predator_count = v; }
    if let Some(v) = get_i32(tbl, "prey_count") { config.prey_count = v; }
    if let Some(v) = get_i32(tbl, "food_count") { config.food_count = v; }
    if let Some(v) = get_f32(tbl, "predator_speed") { config.predator_speed = v; }
    if let Some(v) = get_f32(tbl, "prey_speed") { config.prey_speed = v; }
    if let Some(v) = get_f32(tbl, "vision_range") { config.vision_range = v; }
    if let Some(v) = get_f32(tbl, "interaction_range") { config.interaction_range = v; }
    if let Some(v) = get_f32(tbl, "mate_range") { config.mate_range = v; }
    if let Some(v) = get_f32(tbl, "food_respawn_rate") { config.food_respawn_rate = v; }
    if let Some(v) = get_f32(tbl, "energy_drain_per_step") { config.energy_drain_per_step = v; }
    if let Some(v) = get_f32(tbl, "energy_gain_from_kill") { config.energy_gain_from_kill = v; }
    if let Some(v) = get_f32(tbl, "energy_gain_from_food") { config.energy_gain_from_food = v; }
    if let Some(v) = get_f32(tbl, "initial_energy") { config.initial_energy = v; }
    if let Some(v) = get_f32(tbl, "max_energy") { config.max_energy = v; }
    if let Some(v) = get_f32(tbl, "reproduction_energy_threshold") { config.reproduction_energy_threshold = v; }
    if let Some(v) = get_f32(tbl, "reproduction_energy_cost") { config.reproduction_energy_cost = v; }
    if let Some(v) = get_f32(tbl, "offspring_initial_energy") { config.offspring_initial_energy = v; }
    if let Some(v) = get_i32(tbl, "max_age") { config.max_age = v; }
    if let Some(v) = get_f32(tbl, "mutation_rate") { config.mutation_rate = v; }
    if let Some(v) = get_f32(tbl, "weight_mutation_power") { config.weight_mutation_power = v; }
    if let Some(v) = get_f32(tbl, "add_node_rate") { config.add_node_rate = v; }
    if let Some(v) = get_f32(tbl, "add_connection_rate") { config.add_connection_rate = v; }
    if let Some(v) = get_f32(tbl, "delete_connection_rate") { config.delete_connection_rate = v; }
    if let Some(v) = get_i32(tbl, "max_hidden_nodes") { config.max_hidden_nodes = v; }
    if let Some(v) = get_i32(tbl, "max_steps") { config.max_steps = v; }
    if let Some(v) = get_f32(tbl, "compatibility_threshold") { config.compatibility_threshold = v; }
    if let Some(v) = get_f32(tbl, "compatibility_min_normalization") { config.compatibility_min_normalization = v; }
    if let Some(v) = get_f32(tbl, "c1_excess") { config.c1_excess = v; }
    if let Some(v) = get_f32(tbl, "c2_disjoint") { config.c2_disjoint = v; }
    if let Some(v) = get_f32(tbl, "c3_weight") { config.c3_weight = v; }
    if let Some(v) = get_i32(tbl, "seed") { config.seed = v; }
    if let Some(v) = get_string(tbl, "output_dir") { config.output_dir = v; }
    if let Some(v) = get_i32(tbl, "report_interval_steps") { config.report_interval_steps = v; }

    config
}

pub fn load_config_file(filepath: &Path) -> mlua::Result<HashMap<String, SimulationConfig>> {
    let libs = StdLib::MATH | StdLib::TABLE | StdLib::STRING;
    let lua = Lua::new_with(libs, mlua::LuaOptions::default())?;
    inject_defaults(&lua)?;

    let script = std::fs::read_to_string(filepath)?;
    let result: Table = lua.load(&script).eval()?;

    let mut configs = HashMap::new();
    for pair in result.pairs::<String, Table>() {
        if let Ok((name, exp_tbl)) = pair {
            if name == "moonai_defaults" {
                continue;
            }
            configs.insert(name, table_to_config(&exp_tbl));
        }
    }

    Ok(configs)
}

pub fn load_all_configs_lua(filepath: &Path) -> HashMap<String, SimulationConfig> {
    match load_config_file(filepath) {
        Ok(configs) => {
            if configs.is_empty() {
                log::error!("Lua config '{}' returned no named experiments.", filepath.display());
            } else {
                log::info!("Loaded {} experiment(s) from '{}'.", configs.len(), filepath.display());
            }
            configs
        }
        Err(e) => {
            log::error!("Failed to load Lua config '{}': {}", filepath.display(), e);
            HashMap::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_default_config() {
        let defaults = default_config();
        assert_eq!(defaults.grid_size, 3600);
        assert_eq!(defaults.predator_count, 24000);
        assert_eq!(defaults.seed, 67);
    }
}
