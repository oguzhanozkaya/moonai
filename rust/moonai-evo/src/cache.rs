use moonai_core::{INVALID_ENTITY, OUTPUT_COUNT};
use std::usize;

use crate::genome::Genome;
use crate::network::NeuralNetwork;

#[derive(Clone, Debug)]
pub struct CompiledNetwork {
    pub num_inputs: i32,
    pub num_outputs: i32,
    pub num_nodes: i32,
    pub eval_order: Vec<i32>,
    pub conn_from: Vec<i32>,
    pub conn_weights: Vec<f32>,
    pub conn_ptr: Vec<i32>,
    pub output_indices: [i32; OUTPUT_COUNT],
}

impl CompiledNetwork {
    pub fn num_eval(&self) -> i32 {
        self.eval_order.len() as i32
    }

    pub fn num_connections(&self) -> i32 {
        self.conn_from.len() as i32
    }
}

fn compile_network(network: &NeuralNetwork) -> CompiledNetwork {
    let mut compiled = CompiledNetwork {
        num_inputs: network.num_inputs(),
        num_outputs: network.num_outputs(),
        num_nodes: network.num_nodes(),
        eval_order: network.eval_order_indices().iter().map(|&i| i as i32).collect(),
        conn_from: Vec::new(),
        conn_weights: Vec::new(),
        conn_ptr: Vec::new(),
        output_indices: [0; OUTPUT_COUNT],
    };

    let incoming = network.incoming();
    let mut ptr_val = 0i32;
    for &node_idx in compiled.eval_order.iter() {
        compiled.conn_ptr.push(ptr_val);
        let node_idx_usize = node_idx as usize;
        if node_idx_usize < incoming.len() {
            for &(from_idx, weight) in &incoming[node_idx_usize] {
                compiled.conn_from.push(from_idx as i32);
                compiled.conn_weights.push(weight);
                ptr_val += 1;
            }
        }
    }
    compiled.conn_ptr.push(ptr_val);

    let output_indices = network.output_indices();
    for (i, &idx) in output_indices.iter().take(OUTPUT_COUNT).enumerate() {
        compiled.output_indices[i] = idx as i32;
    }

    compiled
}

pub struct NetworkCache {
    networks: Vec<Option<NeuralNetwork>>,
    compiled: Vec<Option<CompiledNetwork>>,
}

impl NetworkCache {
    pub fn new() -> Self {
        Self {
            networks: Vec::new(),
            compiled: Vec::new(),
        }
    }

    pub fn assign(&mut self, e: u32, genome: &Genome) {
        let e_idx = e as usize;
        if e_idx >= self.networks.len() {
            self.networks.resize(e_idx + 1, None);
            self.compiled.resize(e_idx + 1, None);
        }
        self.networks[e_idx] = Some(NeuralNetwork::new(genome));
        self.compiled[e_idx] = Some(compile_network(self.networks[e_idx].as_ref().unwrap()));
    }

    pub fn get(&self, e: u32) -> Option<&NeuralNetwork> {
        if e == INVALID_ENTITY || e as usize >= self.networks.len() {
            return None;
        }
        self.networks[e as usize].as_ref()
    }

    pub fn get_compiled(&self, e: u32) -> Option<&CompiledNetwork> {
        if e == INVALID_ENTITY || e as usize >= self.compiled.len() {
            return None;
        }
        self.compiled[e as usize].as_ref()
    }

    pub fn remove(&mut self, e: u32) {
        if e == INVALID_ENTITY || e as usize >= self.networks.len() {
            return;
        }
        if e as usize + 1 == self.networks.len() {
            self.networks.pop();
            self.compiled.pop();
            return;
        }
        self.networks[e as usize] = None;
        self.compiled[e as usize] = None;
    }

    pub fn move_entity(&mut self, from: u32, to: u32) {
        if from == to {
            return;
        }
        if from == INVALID_ENTITY || from as usize >= self.networks.len() {
            return;
        }
        if self.networks[from as usize].is_none() {
            return;
        }

        let to_idx = to as usize;
        if to_idx >= self.networks.len() {
            self.networks.resize(to_idx + 1, None);
            self.compiled.resize(to_idx + 1, None);
        }

        self.networks[to_idx] = self.networks[from as usize].take();
        self.compiled[to_idx] = self.compiled[from as usize].take();
    }

    pub fn has(&self, e: u32) -> bool {
        self.get(e).is_some()
    }

    pub fn clear(&mut self) {
        self.networks.clear();
        self.compiled.clear();
    }

    pub fn size(&self) -> usize {
        self.networks.iter().filter(|n| n.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.networks.iter().all(|n| n.is_none())
    }
}

impl Default for NetworkCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_cache_assign() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        assert!(cache.has(0));
        assert!(!cache.has(1));
    }

    #[test]
    fn test_network_cache_get() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        let network = cache.get(0);
        assert!(network.is_some());
        assert_eq!(network.unwrap().num_inputs(), 2);
    }

    #[test]
    fn test_network_cache_get_compiled() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        let compiled = cache.get_compiled(0);
        assert!(compiled.is_some());
    }

    #[test]
    fn test_network_cache_remove() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        cache.remove(0);
        assert!(!cache.has(0));
    }

    #[test]
    fn test_network_cache_move_entity() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        cache.move_entity(0, 1);
        assert!(!cache.has(0));
        assert!(cache.has(1));
    }

    #[test]
    fn test_network_cache_clear() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        cache.assign(1, &genome);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_compiled_network_num_eval() {
        let mut cache = NetworkCache::new();
        let genome = Genome::new(2, 1);
        cache.assign(0, &genome);
        let compiled = cache.get_compiled(0).unwrap();
        assert!(compiled.num_eval() > 0);
    }
}