use moonai_core::{Random, SimulationConfig};

use crate::respawn;

pub struct Food {
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub active: Vec<u8>,
}

impl Food {
    pub fn new() -> Self {
        Self {
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            active: Vec::new(),
        }
    }

    pub fn initialize(&mut self, config: &SimulationConfig, rng: &mut Random) {
        let count = config.food_count as usize;
        self.pos_x.resize(count, 0.0);
        self.pos_y.resize(count, 0.0);
        self.active.resize(count, 0);
        self.active.fill(1);

        let grid_size = config.grid_size as f32;
        for i in 0..config.food_count as usize {
            self.pos_x[i] = rng.next_float(0.0, grid_size);
            self.pos_y[i] = rng.next_float(0.0, grid_size);
        }
    }

    pub fn respawn_step(&mut self, config: &SimulationConfig, step_index: i32, seed: i32) {
        let world_size = config.grid_size as f32;

        for i in 0..self.active.len() {
            if self.active[i] != 0 {
                continue;
            }

            let slot = i as u32;
            if !respawn::should_respawn(seed, step_index, slot, config.food_respawn_rate) {
                continue;
            }

            self.pos_x[i] = respawn::respawn_x(seed, step_index, slot, world_size);
            self.pos_y[i] = respawn::respawn_y(seed, step_index, slot, world_size);
            self.active[i] = 1;
        }
    }

    pub fn size(&self) -> usize {
        self.pos_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }
}

impl Default for Food {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Food {
    fn clone(&self) -> Self {
        Self {
            pos_x: self.pos_x.clone(),
            pos_y: self.pos_y.clone(),
            active: self.active.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_food_creation() {
        let food = Food::new();
        assert!(food.is_empty());
        assert_eq!(food.size(), 0);
    }

    #[test]
    fn test_food_initialize() {
        let mut food = Food::new();
        let config = SimulationConfig::default();
        let mut rng = Random::new(42);
        food.initialize(&config, &mut rng);
        assert_eq!(food.size(), config.food_count as usize);
    }

    #[test]
    fn test_food_all_active_after_init() {
        let mut food = Food::new();
        let config = SimulationConfig::default();
        let mut rng = Random::new(42);
        food.initialize(&config, &mut rng);
        for &active in &food.active {
            assert_eq!(active, 1);
        }
    }
}