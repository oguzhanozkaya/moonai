use std::collections::{HashMap, HashSet, VecDeque};

use crate::genome::{Genome, NodeType};

#[derive(Clone)]
pub struct Node {
    pub id: u32,
    pub node_type: NodeType,
}

#[derive(Clone)]
pub struct Connection {
    from: u32,
    to: u32,
    weight: f32,
}

pub struct IncomingConnection {
    pub from_node: usize,
    pub weight: f32,
}

#[derive(Clone)]
pub struct NeuralNetwork {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
    evaluation_order: Vec<u32>,
    evaluation_order_indices: Vec<usize>,
    node_index: HashMap<u32, usize>,
    incoming: Vec<Vec<(usize, f32)>>,
    output_indices: Vec<usize>,
    values: Vec<f32>,
    num_inputs: i32,
    num_outputs: i32,
}

impl NeuralNetwork {
    pub fn new(genome: &Genome) -> Self {
        let mut nodes = Vec::new();
        let mut node_index = HashMap::new();
        let mut connections = Vec::new();

        for ng in genome.nodes() {
            let idx = nodes.len();
            node_index.insert(ng.id, idx);
            nodes.push(Node {
                id: ng.id,
                node_type: ng.node_type.clone(),
            });
        }

        for cg in genome.connections() {
            if cg.enabled {
                connections.push(Connection {
                    from: cg.in_node,
                    to: cg.out_node,
                    weight: cg.weight,
                });
            }
        }

        let num_inputs = genome.num_inputs();
        let num_outputs = genome.num_outputs();

        let mut result = Self {
            nodes,
            connections,
            evaluation_order: Vec::new(),
            evaluation_order_indices: Vec::new(),
            node_index,
            incoming: Vec::new(),
            output_indices: Vec::new(),
            values: Vec::new(),
            num_inputs,
            num_outputs,
        };

        result.values.resize(result.nodes.len(), 0.0f32);
        result.build_evaluation_order();

        result
    }

    pub fn activate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values = vec![0.0f32; self.nodes.len()];
        let mut idx = 0usize;

        for i in 0..self.nodes.len() {
            match self.nodes[i].node_type {
                NodeType::Input => {
                    if idx < inputs.len() {
                        values[i] = inputs[idx];
                        idx += 1;
                    }
                }
                NodeType::Bias => {
                    values[i] = 1.0f32;
                }
                _ => {}
            }
        }

        for &node_id in &self.evaluation_order {
            let ni = self.node_index[&node_id];
            let mut sum = 0.0f32;
            for &(from_idx, w) in &self.incoming[ni] {
                sum += values[from_idx] * w;
            }
            values[ni] = Self::apply_activation(sum);
        }

        let mut outputs = Vec::new();
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].node_type, NodeType::Output) {
                outputs.push(values[i]);
            }
        }

        outputs
    }

    pub fn activate_into(&mut self, inputs: &[f32], outputs: &mut [f32]) {
        self.values.fill(0.0f32);
        let mut idx = 0usize;

        for i in 0..self.nodes.len() {
            match self.nodes[i].node_type {
                NodeType::Input => {
                    if idx < inputs.len() {
                        self.values[i] = inputs[idx];
                        idx += 1;
                    }
                }
                NodeType::Bias => {
                    self.values[i] = 1.0f32;
                }
                _ => {}
            }
        }

        for &node_id in &self.evaluation_order {
            let ni = self.node_index[&node_id];
            let mut sum = 0.0f32;
            for &(from_idx, w) in &self.incoming[ni] {
                sum += self.values[from_idx] * w;
            }
            self.values[ni] = Self::apply_activation(sum);
        }

        let mut out_idx = 0usize;
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].node_type, NodeType::Output) && out_idx < outputs.len() {
                outputs[out_idx] = self.values[i];
                out_idx += 1;
            }
        }
    }

    fn build_evaluation_order(&mut self) {
        self.evaluation_order.clear();

        let mut eval_nodes: HashSet<u32> = HashSet::new();
        for node in &self.nodes {
            if matches!(node.node_type, NodeType::Hidden | NodeType::Output) {
                eval_nodes.insert(node.id);
            }
        }

        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut in_degree: HashMap<u32, i32> = HashMap::new();

        for &nid in &eval_nodes {
            in_degree.insert(nid, 0);
        }

        for conn in &self.connections {
            if eval_nodes.contains(&conn.to) {
                adj.entry(conn.from).or_default().push(conn.to);
                if eval_nodes.contains(&conn.from) {
                    *in_degree.entry(conn.to).or_default() += 1;
                }
            }
        }

        let mut ready: VecDeque<u32> = VecDeque::new();
        for &nid in &eval_nodes {
            if in_degree[&nid] == 0 {
                ready.push_back(nid);
            }
        }

        while let Some(nid) = ready.pop_front() {
            self.evaluation_order.push(nid);

            if let Some(to_ids) = adj.get(&nid) {
                for &to_id in to_ids {
                    if eval_nodes.contains(&to_id) {
                        *in_degree.entry(to_id).or_default() -= 1;
                        if in_degree[&to_id] == 0 {
                            ready.push_back(to_id);
                        }
                    }
                }
            }
        }

        let ordered_set: HashSet<u32> = self.evaluation_order.iter().cloned().collect();
        let mut cycle_nodes: Vec<u32> = eval_nodes
            .iter()
            .filter(|nid| !ordered_set.contains(nid))
            .cloned()
            .collect();
        cycle_nodes.sort();
        self.evaluation_order.extend(cycle_nodes);

        self.evaluation_order_indices.clear();
        self.evaluation_order_indices.reserve(self.evaluation_order.len());
        for &node_id in &self.evaluation_order {
            self.evaluation_order_indices.push(self.node_index[&node_id]);
        }

        self.incoming
            .resize(self.nodes.len(), Vec::new());
        for conn in &self.connections {
            if let Some(&to_idx) = self.node_index.get(&conn.to) {
                if let Some(&from_idx) = self.node_index.get(&conn.from) {
                    self.incoming[to_idx].push((from_idx, conn.weight));
                }
            }
        }

        self.output_indices.clear();
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].node_type, NodeType::Output) {
                self.output_indices.push(i);
            }
        }
    }

    fn apply_activation(x: f32) -> f32 {
        x.tanh()
    }

    pub fn num_nodes(&self) -> i32 {
        self.nodes.len() as i32
    }

    pub fn num_inputs(&self) -> i32 {
        self.num_inputs
    }

    pub fn num_outputs(&self) -> i32 {
        self.num_outputs
    }

    pub fn eval_order(&self) -> &[u32] {
        &self.evaluation_order
    }

    pub fn eval_order_indices(&self) -> &[usize] {
        &self.evaluation_order_indices
    }

    pub fn incoming(&self) -> &[Vec<(usize, f32)>] {
        &self.incoming
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.output_indices
    }

    pub fn node_index_map(&self) -> &HashMap<u32, usize> {
        &self.node_index
    }

    pub fn num_input_nodes(&self) -> i32 {
        self.nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Input))
            .count() as i32
    }

    pub fn num_output_nodes(&self) -> i32 {
        self.nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Output))
            .count() as i32
    }

    pub fn get_incoming_connections(&self, node_idx: usize) -> Vec<IncomingConnection> {
        if node_idx < self.incoming.len() {
            self.incoming[node_idx]
                .iter()
                .map(|&(from, w)| IncomingConnection {
                    from_node: from,
                    weight: w,
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_output_indices(&self) -> Vec<usize> {
        self.output_indices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::ConnectionGene;

    #[test]
    fn test_neural_network_creation() {
        let genome = Genome::new(2, 1);
        let network = NeuralNetwork::new(&genome);
        assert_eq!(network.num_inputs(), 2);
        assert_eq!(network.num_outputs(), 1);
    }

    #[test]
    fn test_neural_network_activate() {
        let mut genome = Genome::new(2, 1);
        genome.add_connection(ConnectionGene {
            in_node: 0,
            out_node: 3,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        });
        genome.add_connection(ConnectionGene {
            in_node: 1,
            out_node: 3,
            weight: 1.0,
            enabled: true,
            innovation: 1,
        });

        let network = NeuralNetwork::new(&genome);
        let outputs = network.activate(&[1.0, 0.5]);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_activation_bounds() {
        let genome = Genome::new(2, 1);
        let network = NeuralNetwork::new(&genome);
        let outputs = network.activate(&[10.0, 10.0]);
        for o in &outputs {
            assert!(*o >= -1.0 && *o <= 1.0);
        }
    }
}