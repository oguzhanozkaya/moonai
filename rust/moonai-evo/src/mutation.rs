use moonai_core::{Random, SimulationConfig};
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::genome::{ConnectionGene, Genome, NodeGene, NodeType};

pub struct InnovationTracker {
    innovation_counter: u32,
    node_counter: u32,
    innovation_cache: HashMap<(u32, u32), u32>,
    split_node_cache: HashMap<(u32, u32), u32>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        Self {
            innovation_counter: 0,
            node_counter: 0,
            innovation_cache: HashMap::new(),
            split_node_cache: HashMap::new(),
        }
    }

    pub fn init_from_population(&mut self, population: &[Genome]) {
        self.innovation_counter = 0;
        self.node_counter = 0;
        self.innovation_cache.clear();
        self.split_node_cache.clear();

        for genome in population {
            for conn in genome.connections() {
                self.innovation_cache
                    .insert((conn.in_node, conn.out_node), conn.innovation);
                if conn.innovation >= self.innovation_counter {
                    self.innovation_counter = conn.innovation + 1;
                }
            }
            for node in genome.nodes() {
                if node.id >= self.node_counter {
                    self.node_counter = node.id + 1;
                }
            }
        }
    }

    pub fn get_innovation(&mut self, in_node: u32, out_node: u32) -> u32 {
        let key = (in_node, out_node);
        if let Some(&innov) = self.innovation_cache.get(&key) {
            return innov;
        }
        let innov = self.innovation_counter;
        self.innovation_counter += 1;
        self.innovation_cache.insert(key, innov);
        innov
    }

    pub fn next_node_id(&mut self) -> u32 {
        let id = self.node_counter;
        self.node_counter += 1;
        id
    }

    pub fn get_split_node_id(&mut self, in_node: u32, out_node: u32) -> u32 {
        let key = (in_node, out_node);
        if let Some(&node_id) = self.split_node_cache.get(&key) {
            return node_id;
        }
        let node_id = self.next_node_id();
        self.split_node_cache.insert(key, node_id);
        node_id
    }

    pub fn set_counters(&mut self, innov: u32, node: u32) {
        self.innovation_counter = innov;
        self.node_counter = node;
    }
}

impl Default for InnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}

fn would_create_cycle(genome: &Genome, from_id: u32, to_id: u32) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut stack = VecDeque::new();
    stack.push_back(to_id);

    while let Some(nid) = stack.pop_front() {
        if nid == from_id {
            return true;
        }
        if !visited.insert(nid) {
            continue;
        }
        for c in genome.connections() {
            if c.enabled && c.in_node == nid {
                stack.push_back(c.out_node);
            }
        }
    }
    false
}

pub struct Mutation;

impl Mutation {
    pub fn mutate_weights(genome: &mut Genome, rng: &mut Random, power: f32) {
        for conn in genome.connections_mut() {
            if rng.next_bool(0.9) {
                conn.weight += rng.next_gaussian(0.0, power);
                conn.weight = conn.weight.clamp(-8.0, 8.0);
            } else {
                conn.weight = rng.next_float(-2.0, 2.0);
            }
        }
    }

    pub fn add_connection(genome: &mut Genome, rng: &mut Random, tracker: &mut InnovationTracker) {
        let nodes = genome.nodes();
        if nodes.len() < 2 {
            return;
        }

        for _ in 0..30 {
            let from_idx = rng.next_int(0, nodes.len() as i32 - 1) as usize;
            let to_idx = rng.next_int(0, nodes.len() as i32 - 1) as usize;

            let from = &nodes[from_idx];
            let to = &nodes[to_idx];

            if to.node_type == NodeType::Input || to.node_type == NodeType::Bias {
                continue;
            }
            if from.node_type == NodeType::Output {
                continue;
            }
            if from.id == to.id {
                continue;
            }

            if genome.has_connection(from.id, to.id) {
                continue;
            }

            if would_create_cycle(genome, from.id, to.id) {
                continue;
            }

            let innov = tracker.get_innovation(from.id, to.id);
            genome.add_connection(ConnectionGene {
                in_node: from.id,
                out_node: to.id,
                weight: rng.next_float(-1.0, 1.0),
                enabled: true,
                innovation: innov,
            });
            return;
        }
    }

    pub fn add_node(
        genome: &mut Genome,
        rng: &mut Random,
        tracker: &mut InnovationTracker,
        max_hidden_nodes: i32,
    ) {
        if max_hidden_nodes > 0 {
            let hidden_count = genome
                .nodes()
                .iter()
                .filter(|n| n.node_type == NodeType::Hidden)
                .count() as i32;
            if hidden_count >= max_hidden_nodes {
                return;
            }
        }

        let conns = genome.connections();
        if conns.is_empty() {
            return;
        }

        let enabled_indices: Vec<usize> = conns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();

        if enabled_indices.is_empty() {
            return;
        }

        let idx = enabled_indices[rng.next_int(0, enabled_indices.len() as i32 - 1) as usize];

        let in_id = conns[idx].in_node;
        let out_id = conns[idx].out_node;
        let old_weight = conns[idx].weight;

        let conns_mut = genome.connections_mut();
        conns_mut[idx].enabled = false;

        let new_id = tracker.get_split_node_id(in_id, out_id);
        if !genome.has_node(new_id) {
            genome.add_node(NodeGene {
                id: new_id,
                node_type: NodeType::Hidden,
            });
        }

        let ensure_connection =
            |genome: &mut Genome, from_id: u32, to_id: u32, weight: f32, innovation: u32| {
                for conn in genome.connections_mut() {
                    if conn.in_node == from_id && conn.out_node == to_id {
                        conn.enabled = true;
                        return;
                    }
                }
                genome.add_connection(ConnectionGene {
                    in_node: from_id,
                    out_node: to_id,
                    weight,
                    enabled: true,
                    innovation,
                });
            };

        let innov1 = tracker.get_innovation(in_id, new_id);
        ensure_connection(genome, in_id, new_id, 1.0, innov1);

        let innov2 = tracker.get_innovation(new_id, out_id);
        ensure_connection(genome, new_id, out_id, old_weight, innov2);
    }

    pub fn delete_connection(genome: &mut Genome, rng: &mut Random) {
        let conns = genome.connections();
        if conns.len() <= 1 {
            return;
        }

        let idx = rng.next_int(0, conns.len() as i32 - 1) as usize;
        let conns_mut = genome.connections_mut();
        conns_mut.remove(idx);
    }

    pub fn mutate(
        genome: &mut Genome,
        rng: &mut Random,
        config: &SimulationConfig,
        tracker: &mut InnovationTracker,
    ) {
        if rng.next_bool(config.mutation_rate) {
            Self::mutate_weights(genome, rng, config.weight_mutation_power);
        }
        if rng.next_bool(config.add_connection_rate) {
            Self::add_connection(genome, rng, tracker);
        }
        if rng.next_bool(config.add_node_rate) {
            Self::add_node(genome, rng, tracker, config.max_hidden_nodes);
        }
        if rng.next_bool(config.delete_connection_rate) {
            Self::delete_connection(genome, rng);
        }

        let any_enabled = genome
            .connections()
            .iter()
            .any(|conn| conn.enabled);
        if !any_enabled && !genome.connections().is_empty() {
            let idx = rng.next_int(0, genome.connections().len() as i32 - 1) as usize;
            genome.connections_mut()[idx].enabled = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_innovation_tracker_new() {
        let tracker = InnovationTracker::new();
        assert_eq!(tracker.innovation_counter, 0);
        assert_eq!(tracker.node_counter, 0);
    }

    #[test]
    fn test_innovation_tracker_get_innovation() {
        let mut tracker = InnovationTracker::new();
        let innov1 = tracker.get_innovation(0, 3);
        let innov2 = tracker.get_innovation(0, 3);
        assert_eq!(innov1, innov2);
        let innov3 = tracker.get_innovation(1, 3);
        assert_ne!(innov1, innov3);
    }

    #[test]
    fn test_innovation_tracker_next_node_id() {
        let mut tracker = InnovationTracker::new();
        assert_eq!(tracker.next_node_id(), 0);
        assert_eq!(tracker.next_node_id(), 1);
        assert_eq!(tracker.next_node_id(), 2);
    }

    #[test]
    fn test_mutate_weights() {
        let mut rng = Random::new(42);
        let mut genome = Genome::new(2, 1);
        genome.add_connection(ConnectionGene {
            in_node: 0,
            out_node: 3,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        });
        let original_weight = genome.connections()[0].weight;

        Mutation::mutate_weights(&mut genome, &mut rng, 0.5);
    }

    #[test]
    fn test_add_connection() {
        let mut rng = Random::new(42);
        let mut tracker = InnovationTracker::new();
        let mut genome = Genome::new(2, 1);

        Mutation::add_connection(&mut genome, &mut rng, &mut tracker);
    }
}