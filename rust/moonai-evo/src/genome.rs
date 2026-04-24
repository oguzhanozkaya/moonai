use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NodeType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub in_node: u32,
    pub out_node: u32,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    num_inputs: i32,
    num_outputs: i32,
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

impl Genome {
    pub fn new(num_inputs: i32, num_outputs: i32) -> Self {
        let mut nodes = Vec::new();
        let mut id = 0u32;

        for _ in 0..num_inputs {
            nodes.push(NodeGene {
                id,
                node_type: NodeType::Input,
            });
            id += 1;
        }

        nodes.push(NodeGene {
            id,
            node_type: NodeType::Bias,
        });
        id += 1;

        for _ in 0..num_outputs {
            nodes.push(NodeGene {
                id,
                node_type: NodeType::Output,
            });
            id += 1;
        }

        Self {
            num_inputs,
            num_outputs,
            nodes,
            connections: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: NodeGene) {
        self.nodes.push(node);
    }

    pub fn add_connection(&mut self, conn: ConnectionGene) {
        self.connections.push(conn);
    }

    pub fn has_connection(&self, from: u32, to: u32) -> bool {
        self.connections
            .iter()
            .any(|c| c.in_node == from && c.out_node == to)
    }

    pub fn has_node(&self, id: u32) -> bool {
        self.nodes.iter().any(|n| n.id == id)
    }

    pub fn max_node_id(&self) -> u32 {
        self.nodes.iter().map(|n| n.id).max().unwrap_or(0)
    }

    pub fn nodes(&self) -> &[NodeGene] {
        &self.nodes
    }

    pub fn connections(&self) -> &[ConnectionGene] {
        &self.connections
    }

    pub fn connections_mut(&mut self) -> &mut Vec<ConnectionGene> {
        &mut self.connections
    }

    pub fn num_inputs(&self) -> i32 {
        self.num_inputs
    }

    pub fn num_outputs(&self) -> i32 {
        self.num_outputs
    }

    pub fn complexity(&self) -> i32 {
        (self.nodes.len() + self.connections.len()) as i32
    }

    pub fn compatibility_distance(
        a: &Genome,
        b: &Genome,
        c1: f32,
        c2: f32,
        c3: f32,
        min_normalization: f32,
    ) -> f32 {
        let raw_conns_a = a.connections();
        let raw_conns_b = b.connections();

        let is_sorted_by_innovation = |conns: &[ConnectionGene]| {
            for idx in 1..conns.len() {
                if conns[idx - 1].innovation > conns[idx].innovation {
                    return false;
                }
            }
            true
        };

        let sorted_a = is_sorted_by_innovation(raw_conns_a);
        let sorted_b = is_sorted_by_innovation(raw_conns_b);

        let sorted_copy_a: Vec<ConnectionGene> = if !sorted_a {
            let mut v = raw_conns_a.to_vec();
            v.sort_by_key(|c| c.innovation);
            v
        } else {
            raw_conns_a.to_vec()
        };

        let sorted_copy_b: Vec<ConnectionGene> = if !sorted_b {
            let mut v = raw_conns_b.to_vec();
            v.sort_by_key(|c| c.innovation);
            v
        } else {
            raw_conns_b.to_vec()
        };

        let conns_a: &[ConnectionGene] = if sorted_a {
            raw_conns_a
        } else {
            &sorted_copy_a
        };

        let conns_b: &[ConnectionGene] = if sorted_b {
            raw_conns_b
        } else {
            &sorted_copy_b
        };

        if conns_a.is_empty() && conns_b.is_empty() {
            return 0.0f32;
        }

        let mut excess = 0i32;
        let mut disjoint = 0i32;
        let mut matching = 0i32;
        let mut weight_diff = 0.0f32;

        let mut i = 0usize;
        let mut j = 0usize;
        while i < conns_a.len() && j < conns_b.len() {
            let innov_a = conns_a[i].innovation;
            let innov_b = conns_b[j].innovation;

            if innov_a == innov_b {
                matching += 1;
                weight_diff += (conns_a[i].weight - conns_b[j].weight).abs();
                i += 1;
                j += 1;
            } else if innov_a < innov_b {
                disjoint += 1;
                i += 1;
            } else {
                disjoint += 1;
                j += 1;
            }
        }

        let max_a = conns_a.last().map(|c| c.innovation).unwrap_or(0);
        let max_b = conns_b.last().map(|c| c.innovation).unwrap_or(0);
        let min_max = max_a.min(max_b);

        for k in i..conns_a.len() {
            if conns_a[k].innovation > min_max {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }

        for k in j..conns_b.len() {
            if conns_b[k].innovation > min_max {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }

        let avg_weight = if matching > 0 {
            weight_diff / matching as f32
        } else {
            0.0f32
        };

        let n = (conns_a.len().max(conns_b.len()) as f32).max(min_normalization);

        (c1 * excess as f32 / n) + (c2 * disjoint as f32 / n) + (c3 * avg_weight)
    }

    pub fn to_json(&self) -> String {
        let mut j = serde_json::json!({
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
        });

        let nodes_array: Vec<serde_json::Value> = self
            .nodes
            .iter()
            .map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "type": match n.node_type {
                        NodeType::Input => 0,
                        NodeType::Hidden => 1,
                        NodeType::Output => 2,
                        NodeType::Bias => 3,
                    }
                })
            })
            .collect();
        j["nodes"] = serde_json::Value::Array(nodes_array);

        let connections_array: Vec<serde_json::Value> = self
            .connections
            .iter()
            .map(|c| {
                serde_json::json!({
                    "in": c.in_node,
                    "out": c.out_node,
                    "weight": c.weight,
                    "enabled": c.enabled,
                    "innovation": c.innovation,
                })
            })
            .collect();
        j["connections"] = serde_json::Value::Array(connections_array);

        serde_json::to_string_pretty(&j).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_creation() {
        let genome = Genome::new(5, 2);
        assert_eq!(genome.num_inputs(), 5);
        assert_eq!(genome.num_outputs(), 2);

        let input_nodes: Vec<_> = genome
            .nodes()
            .iter()
            .filter(|n| n.node_type == NodeType::Input)
            .collect();
        assert_eq!(input_nodes.len(), 5);

        let bias_nodes: Vec<_> = genome
            .nodes()
            .iter()
            .filter(|n| n.node_type == NodeType::Bias)
            .collect();
        assert_eq!(bias_nodes.len(), 1);

        let output_nodes: Vec<_> = genome
            .nodes()
            .iter()
            .filter(|n| n.node_type == NodeType::Output)
            .collect();
        assert_eq!(output_nodes.len(), 2);
    }

    #[test]
    fn test_has_connection() {
        let mut genome = Genome::new(2, 1);
        assert!(!genome.has_connection(0, 3));

        genome.add_connection(ConnectionGene {
            in_node: 0,
            out_node: 3,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        });
        assert!(genome.has_connection(0, 3));
        assert!(!genome.has_connection(1, 3));
    }

    #[test]
    fn test_complexity() {
        let genome = Genome::new(2, 1);
        assert_eq!(genome.complexity(), 4);

        let mut genome2 = genome.clone();
        genome2.add_node(NodeGene {
            id: 10,
            node_type: NodeType::Hidden,
        });
        assert_eq!(genome2.complexity(), 5);
    }

    #[test]
    fn test_compatibility_distance_empty() {
        let a = Genome::new(2, 1);
        let b = Genome::new(2, 1);
        assert_eq!(
            Genome::compatibility_distance(&a, &b, 1.0, 1.0, 0.4, 1.0),
            0.0
        );
    }
}