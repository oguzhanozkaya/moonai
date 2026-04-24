use moonai_core::{Random, INVALID_ENTITY};
use moonai_evo::{Genome, InnovationTracker, NetworkCache, Species};

pub struct AgentRegistry {
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub energy: Vec<f32>,
    pub age: Vec<i32>,
    pub alive: Vec<u8>,
    pub species_id: Vec<u32>,
    pub entity_id: Vec<u32>,
    pub generation: Vec<i32>,

    pub innovation_tracker: InnovationTracker,
    pub species: Vec<Species>,
    pub genomes: Vec<Genome>,
    pub network_cache: NetworkCache,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            energy: Vec::new(),
            age: Vec::new(),
            alive: Vec::new(),
            species_id: Vec::new(),
            entity_id: Vec::new(),
            generation: Vec::new(),
            innovation_tracker: InnovationTracker::new(),
            species: Vec::new(),
            genomes: Vec::new(),
            network_cache: NetworkCache::new(),
        }
    }

    pub fn create(&mut self) -> u32 {
        let entity = self.size() as u32;
        self.resize(self.size() + 1);
        entity
    }

    pub fn valid(&self, entity: u32) -> bool {
        entity != INVALID_ENTITY && entity < self.size() as u32
    }

    pub fn size(&self) -> usize {
        self.pos_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    pub fn clear(&mut self) {
        let new_size = 0;
        self.resize(new_size);
        self.species.clear();
        self.genomes.clear();
        self.network_cache.clear();
    }

    pub fn compact(&mut self) {
        let mut i = 0;
        while i < self.size() {
            if self.alive[i] != 0 {
                i += 1;
                continue;
            }

            let last = self.size() - 1;
            if i != last {
                self.swap_entities(i, last);

                if last < self.genomes.len() {
                    if i >= self.genomes.len() {
                        self.genomes.resize(i + 1, Genome::new(0, 0));
                    }
                    self.genomes[i] = self.genomes[last].clone();
                }

                self.network_cache
                    .move_entity(last as u32, i as u32);

                if last + 1 == self.genomes.len() {
                    self.genomes.pop();
                }
                self.network_cache.remove(last as u32);
            } else {
                if !self.genomes.is_empty() && last < self.genomes.len() {
                    self.genomes.pop();
                }
                self.network_cache.remove(last as u32);
            }

            self.pop_back();
        }
    }

    pub fn find_by_agent_id(&self, agent_id: u32) -> u32 {
        if let Some(pos) = self.entity_id.iter().position(|&id| id == agent_id) {
            pos as u32
        } else {
            INVALID_ENTITY
        }
    }

    fn resize(&mut self, new_size: usize) {
        self.pos_x.resize(new_size, 0.0);
        self.pos_y.resize(new_size, 0.0);
        self.vel_x.resize(new_size, 0.0);
        self.vel_y.resize(new_size, 0.0);
        self.energy.resize(new_size, 0.0);
        self.age.resize(new_size, 0);
        self.alive.resize(new_size, 0);
        self.species_id.resize(new_size, 0);
        self.entity_id.resize(new_size, 0);
        self.generation.resize(new_size, 0);
    }

    fn swap_entities(&mut self, a: usize, b: usize) {
        self.pos_x.swap(a, b);
        self.pos_y.swap(a, b);
        self.vel_x.swap(a, b);
        self.vel_y.swap(a, b);
        self.energy.swap(a, b);
        self.age.swap(a, b);
        self.alive.swap(a, b);
        self.species_id.swap(a, b);
        self.entity_id.swap(a, b);
        self.generation.swap(a, b);
    }

    fn pop_back(&mut self) {
        self.pos_x.pop();
        self.pos_y.pop();
        self.vel_x.pop();
        self.vel_y.pop();
        self.energy.pop();
        self.age.pop();
        self.alive.pop();
        self.species_id.pop();
        self.entity_id.pop();
        self.generation.pop();
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AgentRegistry {
    fn clone(&self) -> Self {
        Self {
            pos_x: self.pos_x.clone(),
            pos_y: self.pos_y.clone(),
            vel_x: self.vel_x.clone(),
            vel_y: self.vel_y.clone(),
            energy: self.energy.clone(),
            age: self.age.clone(),
            alive: self.alive.clone(),
            species_id: self.species_id.clone(),
            entity_id: self.entity_id.clone(),
            generation: self.generation.clone(),
            innovation_tracker: InnovationTracker::new(),
            species: Vec::new(),
            genomes: self.genomes.clone(),
            network_cache: NetworkCache::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registry_creation() {
        let registry = AgentRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.size(), 0);
    }

    #[test]
    fn test_agent_registry_create() {
        let mut registry = AgentRegistry::new();
        let entity = registry.create();
        assert_eq!(entity, 0);
        assert_eq!(registry.size(), 1);
    }

    #[test]
    fn test_agent_registry_valid() {
        let mut registry = AgentRegistry::new();
        registry.create();
        assert!(registry.valid(0));
        assert!(!registry.valid(1));
        assert!(!registry.valid(INVALID_ENTITY));
    }

    #[test]
    fn test_agent_registry_compact() {
        let mut registry = AgentRegistry::new();
        registry.create();
        registry.create();
        registry.create();

        registry.alive[0] = 1;
        registry.alive[1] = 0;
        registry.alive[2] = 1;

        registry.compact();

        assert_eq!(registry.size(), 2);
        assert_eq!(registry.alive[0], 1);
        assert_eq!(registry.alive[1], 1);
    }

    #[test]
    fn test_agent_registry_find_by_agent_id() {
        let mut registry = AgentRegistry::new();
        registry.create();
        registry.entity_id[0] = 100;
        registry.create();
        registry.entity_id[1] = 200;

        assert_eq!(registry.find_by_agent_id(100), 0);
        assert_eq!(registry.find_by_agent_id(200), 1);
        assert_eq!(registry.find_by_agent_id(999), INVALID_ENTITY);
    }
}