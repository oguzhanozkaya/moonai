use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub step: i32,
    pub predator_count: i32,
    pub prey_count: i32,
    pub active_food: i32,
    pub kills: i32,
    pub food_eaten: i32,
    pub predator_births: i32,
    pub prey_births: i32,
    pub predator_deaths: i32,
    pub prey_deaths: i32,
    pub predator_species: i32,
    pub prey_species: i32,
    pub avg_predator_complexity: f32,
    pub avg_prey_complexity: f32,
    pub avg_predator_energy: f32,
    pub avg_prey_energy: f32,
    pub max_predator_generation: i32,
    pub avg_predator_generation: f32,
    pub max_prey_generation: i32,
    pub avg_prey_generation: f32,
}

impl MetricsSnapshot {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_snapshot_default() {
        let metrics = MetricsSnapshot::default();
        assert_eq!(metrics.step, 0);
        assert_eq!(metrics.predator_count, 0);
        assert_eq!(metrics.prey_count, 0);
    }

    #[test]
    fn test_metrics_snapshot_serialize() {
        let metrics = MetricsSnapshot::default();
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(!json.is_empty());
    }
}