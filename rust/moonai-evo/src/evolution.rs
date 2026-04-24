use moonai_core::{Random, SimulationConfig, Vec2, INVALID_ENTITY};

use crate::crossover::Crossover;
use crate::genome::{ConnectionGene, Genome, NodeType};
use crate::mutation::{InnovationTracker, Mutation};
use crate::species::Species;

pub struct DenseReproductionGrid {
    cell_size: f32,
    cols: i32,
    rows: i32,
    counts: Vec<i32>,
    offsets: Vec<i32>,
    write_offsets: Vec<i32>,
    entries: Vec<u32>,
}

impl DenseReproductionGrid {
    pub fn new(world_width: f32, world_height: f32, cell_size: f32, entity_count: usize) -> Self {
        let cell_size = cell_size.max(1.0);
        let cols = ((world_width / cell_size).ceil() as i32).max(1);
        let rows = ((world_height / cell_size).ceil() as i32).max(1);
        let total_cells = (cols * rows) as usize;

        Self {
            cell_size,
            cols,
            rows,
            counts: vec![0; total_cells],
            offsets: vec![0; total_cells + 1],
            write_offsets: vec![0; total_cells],
            entries: vec![INVALID_ENTITY; entity_count],
        }
    }

    pub fn build(
        &mut self,
        positions: &[(f32, f32)],
        entity_count: usize,
    ) {
        self.counts.fill(0);
        self.offsets.fill(0);

        for idx in 0..entity_count {
            let (x, y) = positions[idx];
            let cell = self.cell_index(x, y);
            self.counts[cell as usize] += 1;
        }

        for cell in 0..self.counts.len() {
            self.offsets[cell + 1] = self.offsets[cell] + self.counts[cell];
        }

        self.write_offsets[..self.offsets.len() - 1]
            .copy_from_slice(&self.offsets[..self.offsets.len() - 1]);

        for idx in 0..entity_count {
            let (x, y) = positions[idx];
            let cell = self.cell_index(x, y) as usize;
            let slot = self.write_offsets[cell] as usize;
            self.write_offsets[cell] += 1;
            self.entries[slot] = idx as u32;
        }
    }

    pub fn for_each_candidate<F>(&self, center: Vec2, radius: f32, mut callback: F)
    where
        F: FnMut(u32),
    {
        let cells_to_check = ((radius / self.cell_size).ceil() as i32).max(1);
        let base_x = self.cell_coord(center.x, self.cols);
        let base_y = self.cell_coord(center.y, self.rows);

        for dy in -cells_to_check..=cells_to_check {
            for dx in -cells_to_check..=cells_to_check {
                let cell_x = self.clamp_cell(base_x + dx, self.cols);
                let cell_y = self.clamp_cell(base_y + dy, self.rows);
                let cell = self.flat_index(cell_x, cell_y);

                let start = self.offsets[cell as usize] as usize;
                let end = self.offsets[(cell + 1) as usize] as usize;
                for slot in start..end {
                    callback(self.entries[slot]);
                }
            }
        }
    }

    fn cell_coord(&self, value: f32, limit: i32) -> i32 {
        let coord = (value / self.cell_size) as i32;
        self.clamp_cell(coord, limit)
    }

    fn clamp_cell(&self, coord: i32, limit: i32) -> i32 {
        coord.max(0).min(limit - 1)
    }

    fn flat_index(&self, x: i32, y: i32) -> i32 {
        y * self.cols + x
    }

    fn cell_index(&self, x: f32, y: f32) -> i32 {
        self.flat_index(self.cell_coord(x, self.cols), self.cell_coord(y, self.rows))
    }
}

pub struct EvolutionManager {
    config: SimulationConfig,
    num_inputs: i32,
    num_outputs: i32,
}

impl EvolutionManager {
    pub fn new(config: &SimulationConfig) -> Self {
        Self {
            config: config.clone(),
            num_inputs: 0,
            num_outputs: 0,
        }
    }

    pub fn initialize(&mut self, num_inputs: i32, num_outputs: i32) {
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
        Species::reset_id_counter();
    }

    pub fn create_initial_genome(
        &self,
        innovation_tracker: &mut InnovationTracker,
        rng: &mut Random,
    ) -> Genome {
        let mut genome = Genome::new(self.num_inputs, self.num_outputs);

        let in_nodes: Vec<_> = genome.nodes().iter().filter(|n| n.node_type == NodeType::Input || n.node_type == NodeType::Bias).cloned().collect();
        let out_nodes: Vec<_> = genome.nodes().iter().filter(|n| n.node_type == NodeType::Output).cloned().collect();

        for in_node in &in_nodes {
            for out_node in &out_nodes {
                let innovation = innovation_tracker.get_innovation(in_node.id, out_node.id);
                genome.add_connection(ConnectionGene {
                    in_node: in_node.id,
                    out_node: out_node.id,
                    weight: rng.next_float(-1.0, 1.0),
                    enabled: true,
                    innovation,
                });
            }
        }
        genome
    }

    pub fn create_child_genome(
        &self,
        innovation_tracker: &mut InnovationTracker,
        rng: &mut Random,
        parent_a: &Genome,
        parent_b: &Genome,
    ) -> Genome {
        let mut child = Crossover::crossover(parent_a, parent_b, rng);
        Mutation::mutate(&mut child, rng, &self.config, innovation_tracker);
        child
    }

    pub fn num_inputs(&self) -> i32 {
        self.num_inputs
    }

    pub fn num_outputs(&self) -> i32 {
        self.num_outputs
    }

    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_reproduction_grid_creation() {
        let grid = DenseReproductionGrid::new(100.0, 100.0, 10.0, 50);
        assert_eq!(grid.cols, 10);
        assert_eq!(grid.rows, 10);
    }

    fn test_dense_reproduction_grid_build() {
        let mut grid = DenseReproductionGrid::new(100.0, 100.0, 10.0, 5);
        let positions = vec![(5.0, 5.0), (15.0, 15.0), (25.0, 25.0), (35.0, 35.0), (45.0, 45.0)];
        grid.build(&positions, 5);

        let mut count = 0;
        grid.for_each_candidate(Vec2::new(20.0, 20.0), 15.0, |_| count += 1);
        assert!(count > 0);
    }

    #[test]
    fn test_evolution_manager_creation() {
        let config = SimulationConfig::default();
        let manager = EvolutionManager::new(&config);
        assert_eq!(manager.num_inputs(), 0);
        assert_eq!(manager.num_outputs(), 0);
    }

    #[test]
    fn test_evolution_manager_initialize() {
        let config = SimulationConfig::default();
        let mut manager = EvolutionManager::new(&config);
        manager.initialize(35, 2);
        assert_eq!(manager.num_inputs(), 35);
        assert_eq!(manager.num_outputs(), 2);
    }

    #[test]
    fn test_create_initial_genome() {
        let config = SimulationConfig::default();
        let mut manager = EvolutionManager::new(&config);
        manager.initialize(2, 1);

        let mut rng = Random::new(42);
        let mut tracker = InnovationTracker::new();
        tracker.set_counters(0, 4);

        let genome = manager.create_initial_genome(&mut tracker, &mut rng);
        assert_eq!(genome.num_inputs(), 2);
        assert_eq!(genome.num_outputs(), 1);
        assert!(!genome.connections().is_empty());
    }

    #[test]
    fn test_create_child_genome() {
        let config = SimulationConfig::default();
        let mut manager = EvolutionManager::new(&config);
        manager.initialize(2, 1);

        let mut rng = Random::new(42);
        let mut tracker = InnovationTracker::new();
        tracker.set_counters(0, 4);

        let parent_a = manager.create_initial_genome(&mut tracker, &mut rng);
        let parent_b = manager.create_initial_genome(&mut tracker, &mut rng);

        let child = manager.create_child_genome(&mut tracker, &mut rng, &parent_a, &parent_b);
        assert_eq!(child.num_inputs(), 2);
        assert_eq!(child.num_outputs(), 1);
    }
}
