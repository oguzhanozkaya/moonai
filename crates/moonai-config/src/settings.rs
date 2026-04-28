use crate::{ConfigError, UiConfig};

pub fn load_settings(path: Option<&str>) -> Result<UiConfig, ConfigError> {
    let file = match path {
        Some(p) => std::fs::File::open(p).map_err(ConfigError::Io)?,
        None => {
            let Some(binary_path) = std::env::current_exe().ok() else {
                return Ok(UiConfig::default());
            };
            let Some(binary_dir) = binary_path.parent() else {
                return Ok(UiConfig::default());
            };
            let settings_path = binary_dir.join("config").join("settings.json");
            match std::fs::File::open(&settings_path) {
                Ok(f) => f,
                Err(_) => return Ok(UiConfig::default()),
            }
        }
    };

    let settings: UiConfig = serde_json::from_reader(file)?;
    Ok(settings)
}

pub fn settings_path_from_binary() -> Option<std::path::PathBuf> {
    let binary_path = std::env::current_exe().ok()?;
    let binary_dir = binary_path.parent()?;
    let settings_path = binary_dir.join("config").join("settings.json");
    if settings_path.exists() { Some(settings_path) } else { None }
}

pub fn config_path_from_binary() -> Option<std::path::PathBuf> {
    let binary_path = std::env::current_exe().ok()?;
    let binary_dir = binary_path.parent()?;
    let config_path = binary_dir.join("config").join("config.lua");
    if config_path.exists() { Some(config_path) } else { None }
}
