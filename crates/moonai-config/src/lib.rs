pub mod cli;
pub mod config;
pub mod error;
pub mod lua;
pub mod ui;

pub use cli::CliArgs;
pub use config::SimulationConfig;
pub use error::ConfigError;
pub use ui::UiConfig;

pub fn validate_config(_config: &SimulationConfig) -> Result<(), ConfigError> {
    todo!()
}
