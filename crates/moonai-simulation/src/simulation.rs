use moonai_config::SimulationConfig;

pub struct SimulationState;

impl Default for SimulationState {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationState {
    pub fn new() -> Self {
        Self
    }

    pub fn init_from_config(_config: &SimulationConfig) -> anyhow::Result<Self> {
        Ok(Self)
    }
}
