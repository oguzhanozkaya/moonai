use moonai_core::MetricsSnapshot;

use crate::agent::AgentRegistry;
use crate::food::Food;

#[derive(Clone, Debug)]
pub struct UiState {
    pub paused: bool,
    pub step_requested: bool,
    pub speed_multiplier: i32,
    pub selected_agent_id: u32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            paused: false,
            step_requested: false,
            speed_multiplier: 1,
            selected_agent_id: 0,
        }
    }
}

#[derive(Clone)]
pub struct RuntimeState {
    pub rng: moonai_core::Random,
    pub next_agent_id: u32,
    pub step: i32,
}

impl RuntimeState {
    pub fn new(seed: i32) -> Self {
        Self {
            rng: moonai_core::Random::new(seed),
            next_agent_id: 1,
            step: 0,
        }
    }
}

pub struct StepBuffers {
    pub was_food_active: Vec<u8>,
    pub predator_sensors: Vec<f32>,
    pub prey_sensors: Vec<f32>,
    pub predator_decisions: Vec<f32>,
    pub prey_decisions: Vec<f32>,
    pub food_consumed_by: Vec<i32>,
    pub killed_by: Vec<i32>,
    pub kill_counts: Vec<u32>,
}

impl StepBuffers {
    pub fn new() -> Self {
        Self {
            was_food_active: Vec::new(),
            predator_sensors: Vec::new(),
            prey_sensors: Vec::new(),
            predator_decisions: Vec::new(),
            prey_decisions: Vec::new(),
            food_consumed_by: Vec::new(),
            killed_by: Vec::new(),
            kill_counts: Vec::new(),
        }
    }
}

impl Default for StepBuffers {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AppState {
    pub ui: UiState,
    pub predator: AgentRegistry,
    pub prey: AgentRegistry,
    pub food: Food,
    pub metrics: MetricsSnapshot,
    pub runtime: RuntimeState,
    pub step_buffers: StepBuffers,
}

impl AppState {
    pub fn new(seed: i32) -> Self {
        Self {
            ui: UiState::default(),
            predator: AgentRegistry::new(),
            prey: AgentRegistry::new(),
            food: Food::new(),
            metrics: MetricsSnapshot::default(),
            runtime: RuntimeState::new(seed),
            step_buffers: StepBuffers::new(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_state_default() {
        let ui = UiState::default();
        assert!(!ui.paused);
        assert!(!ui.step_requested);
        assert_eq!(ui.speed_multiplier, 1);
        assert_eq!(ui.selected_agent_id, 0);
    }

    #[test]
    fn test_runtime_state_creation() {
        let runtime = RuntimeState::new(42);
        assert_eq!(runtime.next_agent_id, 1);
        assert_eq!(runtime.step, 0);
        assert_eq!(runtime.rng.seed(), 42);
    }

    #[test]
    fn test_step_buffers_creation() {
        let buffers = StepBuffers::new();
        assert!(buffers.was_food_active.is_empty());
    }

    #[test]
    fn test_app_state_creation() {
        let state = AppState::new(42);
        assert_eq!(state.runtime.rng.seed(), 42);
        assert!(state.predator.is_empty());
        assert!(state.prey.is_empty());
        assert!(state.food.is_empty());
    }
}