use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Lua parsing error: {0}")]
    LuaParse(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
