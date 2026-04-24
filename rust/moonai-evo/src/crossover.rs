use moonai_core::Random;
use std::collections::{HashMap, HashSet};

use crate::genome::{ConnectionGene, Genome, NodeType};

pub struct Crossover;

impl Crossover {
    pub fn crossover(parent_a: &Genome, parent_b: &Genome, rng: &mut Random) -> Genome {
        let child = Genome::new(parent_a.num_inputs(), parent_a.num_outputs());

        let map_a: HashMap<u32, &ConnectionGene> =
            parent_a.connections().iter().map(|c| (c.innovation, c)).collect();
        let map_b: HashMap<u32, &ConnectionGene> =
            parent_b.connections().iter().map(|c| (c.innovation, c)).collect();

        let mut needed_nodes: HashSet<u32> = HashSet::new();
        let mut inherited_connections: Vec<ConnectionGene> = Vec::new();

        let mut all_innovations: HashSet<u32> = HashSet::new();
        for &innov in map_a.keys() {
            all_innovations.insert(innov);
        }
        for &innov in map_b.keys() {
            all_innovations.insert(innov);
        }

        let mut all_innovations_vec: Vec<u32> = all_innovations.into_iter().collect();
        all_innovations_vec.sort();

        let all_innovations_len = all_innovations_vec.len();

        for innov in &all_innovations_vec {
            let it_a = map_a.get(&innov);
            let it_b = map_b.get(&innov);

            if let (Some(gene_a), Some(gene_b)) = (it_a, it_b) {
                let mut gene = if rng.next_bool(0.5) {
                    (*gene_a).clone()
                } else {
                    (*gene_b).clone()
                };

                if !gene_a.enabled || !gene_b.enabled {
                    gene.enabled = !rng.next_bool(0.75);
                }
                needed_nodes.insert(gene.in_node);
                needed_nodes.insert(gene.out_node);
                inherited_connections.push(gene);
                continue;
            }

            if !rng.next_bool(0.5) {
                continue;
            }

            let gene = if let Some(g) = it_a {
                (*g).clone()
            } else if let Some(g) = it_b {
                (*g).clone()
            } else {
                continue;
            };
            needed_nodes.insert(gene.in_node);
            needed_nodes.insert(gene.out_node);
            inherited_connections.push(gene);
        }

        if inherited_connections.is_empty() && all_innovations_len > 0 {
            let innov_idx = rng.next_int(0, all_innovations_len as i32 - 1) as usize;
            let innov = all_innovations_vec[innov_idx];
            let gene = if let Some(g) = map_a.get(&innov) {
                (*g).clone()
            } else {
                (*map_b.get(&innov).unwrap()).clone()
            };
            needed_nodes.insert(gene.in_node);
            needed_nodes.insert(gene.out_node);
            inherited_connections.push(gene);
        }

        let mut result = child;
        for gene in inherited_connections {
            result.add_connection(gene);
        }

        for node in parent_a.nodes() {
            if node.node_type == NodeType::Hidden && needed_nodes.contains(&node.id) {
                result.add_node(node.clone());
            }
        }

        for node in parent_b.nodes() {
            if node.node_type == NodeType::Hidden
                && needed_nodes.contains(&node.id)
                && !result.has_node(node.id)
            {
                result.add_node(node.clone());
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover_empty_genomes() {
        let mut rng = Random::new(42);
        let a = Genome::new(2, 1);
        let b = Genome::new(2, 1);
        let child = Crossover::crossover(&a, &b, &mut rng);
        assert_eq!(child.num_inputs(), 2);
        assert_eq!(child.num_outputs(), 1);
    }

    #[test]
    fn test_crossover_with_connections() {
        let mut rng = Random::new(42);
        let mut a = Genome::new(2, 1);
        a.add_connection(ConnectionGene {
            in_node: 0,
            out_node: 3,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        });

        let mut b = Genome::new(2, 1);
        b.add_connection(ConnectionGene {
            in_node: 0,
            out_node: 3,
            weight: 2.0,
            enabled: true,
            innovation: 0,
        });

        let child = Crossover::crossover(&a, &b, &mut rng);
        assert_eq!(child.connections().len(), 1);
    }
}