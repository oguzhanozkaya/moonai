use anyhow::Result;
use mlua::Lua;
use std::collections::HashMap;

use crate::{ConfigError, SimulationConfig};

pub fn load_config(path: &str) -> Result<HashMap<String, SimulationConfig>, ConfigError> {
    let lua = Lua::new();
    let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;

    lua.load(&content).exec().map_err(|e| ConfigError::LuaParse(e.to_string()))?;

    Ok(HashMap::new())
}
