use moonai_core::{SimulationConfig, SENSOR_COUNT, OUTPUT_COUNT};
use moonai_ffi::{SimContext, SimulationConfig as FFISimulationConfig, SyncBuffers};
use moonai_evo::{EvolutionManager, Genome, Species};
use moonai_metrics::{Logger, refresh_metrics, AgentMetricsData, FoodMetricsData};
use moonai_state::{AppState, AgentRegistry};

#[derive(Clone)]
pub struct AppConfig {
    pub sim_config: SimulationConfig,
    pub experiment_name: String,
    pub headless: bool,
    pub speed_multiplier: i32,
    pub run_name_override: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            sim_config: SimulationConfig::default(),
            experiment_name: String::new(),
            headless: false,
            speed_multiplier: 1,
            run_name_override: None,
        }
    }
}

pub struct App {
    cfg: AppConfig,
    pub state: AppState,
    evolution: EvolutionManager,
    logger: Logger,
    sim_ctx: Option<SimContext>,
}

impl App {
    pub fn new(cfg: AppConfig) -> Self {
        let seed = cfg.sim_config.seed;
        let cfg_clone = cfg.clone();
        let mut app = Self {
            cfg: cfg_clone,
            state: AppState::new(seed),
            evolution: EvolutionManager::new(&cfg.sim_config),
            logger: Logger::new(
                &cfg.sim_config.output_dir,
                cfg.sim_config.seed,
                cfg.run_name_override
                    .as_ref()
                    .unwrap_or(&cfg.experiment_name),
            ),
            sim_ctx: None,
        };

        app.state.ui.speed_multiplier = cfg.speed_multiplier;

        app.initialize();

        app
    }

    fn initialize(&mut self) {
        // Create simulation context (GPU)
        let mut ctx = SimContext::new()
            .expect("Failed to create simulation context");

        let ffi_config = FFISimulationConfig {
            grid_size: self.cfg.sim_config.grid_size,
            predator_count: self.cfg.sim_config.predator_count,
            prey_count: self.cfg.sim_config.prey_count,
            food_count: self.cfg.sim_config.food_count,
            predator_speed: self.cfg.sim_config.predator_speed,
            prey_speed: self.cfg.sim_config.prey_speed,
            vision_range: self.cfg.sim_config.vision_range,
            interaction_range: self.cfg.sim_config.interaction_range,
            mate_range: self.cfg.sim_config.mate_range,
            food_respawn_rate: self.cfg.sim_config.food_respawn_rate,
            energy_drain_per_step: self.cfg.sim_config.energy_drain_per_step,
            energy_gain_from_kill: self.cfg.sim_config.energy_gain_from_kill,
            energy_gain_from_food: self.cfg.sim_config.energy_gain_from_food,
            initial_energy: self.cfg.sim_config.initial_energy,
            max_energy: self.cfg.sim_config.max_energy,
            reproduction_energy_threshold: self.cfg.sim_config.reproduction_energy_threshold,
            reproduction_energy_cost: self.cfg.sim_config.reproduction_energy_cost,
            offspring_initial_energy: self.cfg.sim_config.offspring_initial_energy,
            max_age: self.cfg.sim_config.max_age,
            mutation_rate: self.cfg.sim_config.mutation_rate,
            weight_mutation_power: self.cfg.sim_config.weight_mutation_power,
            add_node_rate: self.cfg.sim_config.add_node_rate,
            add_connection_rate: self.cfg.sim_config.add_connection_rate,
            delete_connection_rate: self.cfg.sim_config.delete_connection_rate,
            max_hidden_nodes: self.cfg.sim_config.max_hidden_nodes,
            max_steps: self.cfg.sim_config.max_steps,
            compatibility_threshold: self.cfg.sim_config.compatibility_threshold,
            compatibility_min_normalization: self.cfg.sim_config.compatibility_min_normalization,
            c1_excess: self.cfg.sim_config.c1_excess,
            c2_disjoint: self.cfg.sim_config.c2_disjoint,
            c3_weight: self.cfg.sim_config.c3_weight,
            seed: self.cfg.sim_config.seed,
            report_interval_steps: self.cfg.sim_config.report_interval_steps,
        };

        if !ctx.init(&ffi_config) {
            panic!("Failed to initialize simulation: {}", ctx.error());
        }

        self.sim_ctx = Some(ctx);

        // Initialize food via CPU
        self.state
            .food
            .initialize(&self.cfg.sim_config, &mut self.state.runtime.rng);

        // Initialize evolution manager
        self.evolution
            .initialize(SENSOR_COUNT as i32, OUTPUT_COUNT as i32);

        // Seed initial population
        self.seed_initial_population();
        self.refresh_species();
        self.refresh_metrics();

        if let Err(e) = self.logger.initialize(&self.cfg.sim_config) {
            eprintln!("Warning: Failed to initialize logger: {}", e);
        }

        log::info!("Simulation initialized with GPU");
    }

    fn seed_initial_population(&mut self) {
        let grid_size = self.cfg.sim_config.grid_size as f32;
        let num_inputs = SENSOR_COUNT as i32;
        let num_outputs = OUTPUT_COUNT as i32;

        for _ in 0..self.cfg.sim_config.predator_count {
            let idx = self.state.predator.create();
            let genome = self
                .evolution
                .create_initial_genome(
                    &mut self.state.predator.innovation_tracker,
                    &mut self.state.runtime.rng,
                );

            if idx as usize >= self.state.predator.genomes.len() {
                self.state
                    .predator
                    .genomes
                    .resize(idx as usize + 1, Genome::new(num_inputs, num_outputs));
            }
            self.state.predator.genomes[idx as usize] = genome;
            self.state
                .predator
                .network_cache
                .assign(idx, &self.state.predator.genomes[idx as usize]);

            let idx_usize = idx as usize;
            self.state.predator.pos_x[idx_usize] = self.state.runtime.rng.next_float(0.0, grid_size);
            self.state.predator.pos_y[idx_usize] = self.state.runtime.rng.next_float(0.0, grid_size);
            self.state.predator.vel_x[idx_usize] = 0.0;
            self.state.predator.vel_y[idx_usize] = 0.0;
            self.state.predator.energy[idx_usize] =
                self.cfg.sim_config.initial_energy.min(self.cfg.sim_config.max_energy);
            self.state.predator.age[idx_usize] = 0;
            self.state.predator.alive[idx_usize] = 1;
            self.state.predator.species_id[idx_usize] = 0;
            self.state.predator.entity_id[idx_usize] = self.state.runtime.next_agent_id;
            self.state.predator.generation[idx_usize] = 0;
            self.state.runtime.next_agent_id += 1;
        }

        for _ in 0..self.cfg.sim_config.prey_count {
            let idx = self.state.prey.create();
            let genome = self.evolution.create_initial_genome(
                &mut self.state.prey.innovation_tracker,
                &mut self.state.runtime.rng,
            );

            if idx as usize >= self.state.prey.genomes.len() {
                self.state
                    .prey
                    .genomes
                    .resize(idx as usize + 1, Genome::new(num_inputs, num_outputs));
            }
            self.state.prey.genomes[idx as usize] = genome;
            self.state
                .prey
                .network_cache
                .assign(idx, &self.state.prey.genomes[idx as usize]);

            let idx_usize = idx as usize;
            self.state.prey.pos_x[idx_usize] = self.state.runtime.rng.next_float(0.0, grid_size);
            self.state.prey.pos_y[idx_usize] = self.state.runtime.rng.next_float(0.0, grid_size);
            self.state.prey.vel_x[idx_usize] = 0.0;
            self.state.prey.vel_y[idx_usize] = 0.0;
            self.state.prey.energy[idx_usize] =
                self.cfg.sim_config.initial_energy.min(self.cfg.sim_config.max_energy);
            self.state.prey.age[idx_usize] = 0;
            self.state.prey.alive[idx_usize] = 1;
            self.state.prey.species_id[idx_usize] = 0;
            self.state.prey.entity_id[idx_usize] = self.state.runtime.next_agent_id;
            self.state.prey.generation[idx_usize] = 0;
            self.state.runtime.next_agent_id += 1;
        }
    }

    fn refresh_species(&mut self) {
        self.refresh_population_species();
        self.refresh_population_species_prey();
    }

    fn refresh_population_species(&mut self) {
        let registry = &mut self.state.predator;
        for species in &mut registry.species {
            species.clear_members();
        }

        let entity_count = registry.size() as u32;
        for idx in 0..entity_count {
            if idx as usize >= registry.genomes.len() {
                continue;
            }

            let genome = &registry.genomes[idx as usize];
            let mut assigned_species_id = -1i32;

            for species in &mut registry.species {
                if species.is_compatible(
                    genome,
                    self.cfg.sim_config.compatibility_threshold,
                    self.cfg.sim_config.c1_excess,
                    self.cfg.sim_config.c2_disjoint,
                    self.cfg.sim_config.c3_weight,
                    self.cfg.sim_config.compatibility_min_normalization,
                ) {
                    species.add_member(idx, genome);
                    assigned_species_id = species.id();
                    break;
                }
            }

            if assigned_species_id < 0 {
                let mut new_species = Species::new(genome);
                new_species.add_member(idx, genome);
                assigned_species_id = new_species.id();
                registry.species.push(new_species);
            }

            registry.species_id[idx as usize] = assigned_species_id as u32;
        }

        for species in &mut registry.species {
            species.refresh_summary();
        }

        registry.species.retain(|s| !s.members().is_empty());
    }

    fn refresh_population_species_prey(&mut self) {
        let registry = &mut self.state.prey;
        for species in &mut registry.species {
            species.clear_members();
        }

        let entity_count = registry.size() as u32;
        for idx in 0..entity_count {
            if idx as usize >= registry.genomes.len() {
                continue;
            }

            let genome = &registry.genomes[idx as usize];
            let mut assigned_species_id = -1i32;

            for species in &mut registry.species {
                if species.is_compatible(
                    genome,
                    self.cfg.sim_config.compatibility_threshold,
                    self.cfg.sim_config.c1_excess,
                    self.cfg.sim_config.c2_disjoint,
                    self.cfg.sim_config.c3_weight,
                    self.cfg.sim_config.compatibility_min_normalization,
                ) {
                    species.add_member(idx, genome);
                    assigned_species_id = species.id();
                    break;
                }
            }

            if assigned_species_id < 0 {
                let mut new_species = Species::new(genome);
                new_species.add_member(idx, genome);
                assigned_species_id = new_species.id();
                registry.species.push(new_species);
            }

            registry.species_id[idx as usize] = assigned_species_id as u32;
        }

        for species in &mut registry.species {
            species.refresh_summary();
        }

        registry.species.retain(|s| !s.members().is_empty());
    }

    fn refresh_metrics(&mut self) {
        let predator_data = AgentMetricsData {
            energy: &self.state.predator.energy,
            genomes: &self.state.predator.genomes,
            generation: &self.state.predator.generation,
            species_count: self.state.predator.species.len(),
        };

        let prey_data = AgentMetricsData {
            energy: &self.state.prey.energy,
            genomes: &self.state.prey.genomes,
            generation: &self.state.prey.generation,
            species_count: self.state.prey.species.len(),
        };

        let food_data = FoodMetricsData {
            active: &self.state.food.active,
        };

        refresh_metrics(
            &mut self.state.metrics,
            self.state.runtime.step,
            predator_data,
            prey_data,
            food_data,
            0,
            0,
            0,
            0,
        );
    }

    pub fn step(&mut self) -> bool {
        let sim_ctx = self.sim_ctx.as_mut().expect("Simulation context not initialized");

        // GPU: run full simulation step
        if !sim_ctx.step() {
            eprintln!("Simulation step failed: {}", sim_ctx.error());
            return false;
        }

        // Update Rust state from GPU state
        self.state.runtime.step = sim_ctx.get_step();

        // Sync predator state from C++
        let mut buffers = SyncBuffers::default();
        sim_ctx.sync_predator_state(&mut buffers);

        let pred_count = buffers.predator_pos_x.len();
        self.state.predator.pos_x.resize(pred_count, 0.0);
        self.state.predator.pos_y.resize(pred_count, 0.0);
        self.state.predator.vel_x.resize(pred_count, 0.0);
        self.state.predator.vel_y.resize(pred_count, 0.0);
        self.state.predator.energy.resize(pred_count, 0.0);
        self.state.predator.age.resize(pred_count, 0);
        self.state.predator.alive.resize(pred_count, 0);

        self.state.predator.pos_x.copy_from_slice(&buffers.predator_pos_x);
        self.state.predator.pos_y.copy_from_slice(&buffers.predator_pos_y);
        self.state.predator.vel_x.copy_from_slice(&buffers.predator_vel_x);
        self.state.predator.vel_y.copy_from_slice(&buffers.predator_vel_y);
        self.state.predator.energy.copy_from_slice(&buffers.predator_energy);
        self.state.predator.age.copy_from_slice(&buffers.predator_age);
        self.state.predator.alive.copy_from_slice(&buffers.predator_alive);

        // Sync prey state from C++
        sim_ctx.sync_prey_state(&mut buffers);

        let prey_count = buffers.prey_pos_x.len();
        self.state.prey.pos_x.resize(prey_count, 0.0);
        self.state.prey.pos_y.resize(prey_count, 0.0);
        self.state.prey.vel_x.resize(prey_count, 0.0);
        self.state.prey.vel_y.resize(prey_count, 0.0);
        self.state.prey.energy.resize(prey_count, 0.0);
        self.state.prey.age.resize(prey_count, 0);
        self.state.prey.alive.resize(prey_count, 0);

        self.state.prey.pos_x.copy_from_slice(&buffers.prey_pos_x);
        self.state.prey.pos_y.copy_from_slice(&buffers.prey_pos_y);
        self.state.prey.vel_x.copy_from_slice(&buffers.prey_vel_x);
        self.state.prey.vel_y.copy_from_slice(&buffers.prey_vel_y);
        self.state.prey.energy.copy_from_slice(&buffers.prey_energy);
        self.state.prey.age.copy_from_slice(&buffers.prey_age);
        self.state.prey.alive.copy_from_slice(&buffers.prey_alive);

        self.refresh_metrics();

        true
    }

    fn reproduce_population_predator(&mut self) {
        let world_size = self.cfg.sim_config.grid_size as f32;
        let num_inputs = SENSOR_COUNT as i32;
        let num_outputs = OUTPUT_COUNT as i32;
        let registry = &mut self.state.predator;

        for i in 0..registry.size() {
            if registry.energy[i] < self.cfg.sim_config.reproduction_energy_threshold {
                continue;
            }

            let mut best_mate = None;
            let mut best_dist_sq = f32::MAX;

            for j in 0..registry.size() {
                if i == j {
                    continue;
                }
                if registry.energy[j] < self.cfg.sim_config.reproduction_energy_threshold {
                    continue;
                }

                let dx = registry.pos_x[j] - registry.pos_x[i];
                let dy = registry.pos_y[j] - registry.pos_y[i];
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < best_dist_sq
                    && dist_sq
                        < self.cfg.sim_config.mate_range * self.cfg.sim_config.mate_range
                {
                    best_dist_sq = dist_sq;
                    best_mate = Some(j);
                }
            }

            if let Some(mate_idx) = best_mate {
                let mid_x = (registry.pos_x[i] + registry.pos_x[mate_idx]) * 0.5;
                let mid_y = (registry.pos_y[i] + registry.pos_y[mate_idx]) * 0.5;
                let spawn_x = mid_x.clamp(0.0, world_size);
                let spawn_y = mid_y.clamp(0.0, world_size);

                let child_genome = self.evolution.create_child_genome(
                    &mut registry.innovation_tracker,
                    &mut self.state.runtime.rng,
                    &registry.genomes[i],
                    &registry.genomes[mate_idx],
                );

                let child_idx = registry.create();
                registry.pos_x[child_idx as usize] = spawn_x;
                registry.pos_y[child_idx as usize] = spawn_y;
                registry.vel_x[child_idx as usize] = 0.0;
                registry.vel_y[child_idx as usize] = 0.0;
                registry.energy[child_idx as usize] = self
                    .cfg
                    .sim_config
                    .offspring_initial_energy
                    .min(self.cfg.sim_config.max_energy);
                registry.age[child_idx as usize] = 0;
                registry.alive[child_idx as usize] = 1;
                registry.species_id[child_idx as usize] = registry.species_id[i];
                registry.entity_id[child_idx as usize] = self.state.runtime.next_agent_id;
                registry.generation[child_idx as usize] =
                    (registry.generation[i].max(registry.generation[mate_idx])) + 1;
                self.state.runtime.next_agent_id += 1;

                if child_idx as usize >= registry.genomes.len() {
                    registry.genomes.resize(
                        child_idx as usize + 1,
                        Genome::new(num_inputs, num_outputs),
                    );
                }
                registry.genomes[child_idx as usize] = child_genome;
                registry
                    .network_cache
                    .assign(child_idx, &registry.genomes[child_idx as usize]);

                registry.energy[i] -= self.cfg.sim_config.reproduction_energy_cost;
                registry.energy[mate_idx] -= self.cfg.sim_config.reproduction_energy_cost;
            }
        }
    }

    fn reproduce_population_prey(&mut self) {
        let world_size = self.cfg.sim_config.grid_size as f32;
        let num_inputs = SENSOR_COUNT as i32;
        let num_outputs = OUTPUT_COUNT as i32;
        let registry = &mut self.state.prey;

        for i in 0..registry.size() {
            if registry.energy[i] < self.cfg.sim_config.reproduction_energy_threshold {
                continue;
            }

            let mut best_mate = None;
            let mut best_dist_sq = f32::MAX;

            for j in 0..registry.size() {
                if i == j {
                    continue;
                }
                if registry.energy[j] < self.cfg.sim_config.reproduction_energy_threshold {
                    continue;
                }

                let dx = registry.pos_x[j] - registry.pos_x[i];
                let dy = registry.pos_y[j] - registry.pos_y[i];
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < best_dist_sq
                    && dist_sq
                        < self.cfg.sim_config.mate_range * self.cfg.sim_config.mate_range
                {
                    best_dist_sq = dist_sq;
                    best_mate = Some(j);
                }
            }

            if let Some(mate_idx) = best_mate {
                let mid_x = (registry.pos_x[i] + registry.pos_x[mate_idx]) * 0.5;
                let mid_y = (registry.pos_y[i] + registry.pos_y[mate_idx]) * 0.5;
                let spawn_x = mid_x.clamp(0.0, world_size);
                let spawn_y = mid_y.clamp(0.0, world_size);

                let child_genome = self.evolution.create_child_genome(
                    &mut registry.innovation_tracker,
                    &mut self.state.runtime.rng,
                    &registry.genomes[i],
                    &registry.genomes[mate_idx],
                );

                let child_idx = registry.create();
                registry.pos_x[child_idx as usize] = spawn_x;
                registry.pos_y[child_idx as usize] = spawn_y;
                registry.vel_x[child_idx as usize] = 0.0;
                registry.vel_y[child_idx as usize] = 0.0;
                registry.energy[child_idx as usize] = self
                    .cfg
                    .sim_config
                    .offspring_initial_energy
                    .min(self.cfg.sim_config.max_energy);
                registry.age[child_idx as usize] = 0;
                registry.alive[child_idx as usize] = 1;
                registry.species_id[child_idx as usize] = registry.species_id[i];
                registry.entity_id[child_idx as usize] = self.state.runtime.next_agent_id;
                registry.generation[child_idx as usize] =
                    (registry.generation[i].max(registry.generation[mate_idx])) + 1;
                self.state.runtime.next_agent_id += 1;

                if child_idx as usize >= registry.genomes.len() {
                    registry.genomes.resize(
                        child_idx as usize + 1,
                        Genome::new(num_inputs, num_outputs),
                    );
                }
                registry.genomes[child_idx as usize] = child_genome;
                registry
                    .network_cache
                    .assign(child_idx, &registry.genomes[child_idx as usize]);

                registry.energy[i] -= self.cfg.sim_config.reproduction_energy_cost;
                registry.energy[mate_idx] -= self.cfg.sim_config.reproduction_energy_cost;
            }
        }
    }

    pub fn run(&mut self) -> bool {
        let max_steps = self.cfg.sim_config.max_steps;

        while self.state.runtime.step < max_steps || max_steps == 0 {
            let steps_to_run = if self.state.ui.paused {
                if self.state.ui.step_requested {
                    self.state.ui.step_requested = false;
                    1
                } else {
                    0
                }
            } else {
                self.state.ui.speed_multiplier.max(1)
            };

            for _ in 0..steps_to_run {
                if !self.step() {
                    return false;
                }
            }

            if self.state.runtime.step % self.cfg.sim_config.report_interval_steps == 0 {
                self.record_and_log();
            }
        }

        self.logger.flush();
        log::info!("Output saved to: {}", self.logger.run_dir());
        log::info!("Simulation ended with 'max_steps'.");
        true
    }

    fn record_and_log(&mut self) {
        self.refresh_species();
        self.refresh_metrics();

        self.logger.log_report(&self.state.metrics);

        let mut best_genome: Option<&Genome> = None;
        let mut best_complexity = 0;

        for genome in &self.state.predator.genomes {
            let complexity = genome.nodes().len() + genome.connections().len();
            if best_genome.is_none() || complexity > best_complexity {
                best_genome = Some(genome);
                best_complexity = complexity;
            }
        }

        for genome in &self.state.prey.genomes {
            let complexity = genome.nodes().len() + genome.connections().len();
            if best_genome.is_none() || complexity > best_complexity {
                best_genome = Some(genome);
                best_complexity = complexity;
            }
        }

        if let Some(genome) = best_genome {
            self.logger
                .log_best_genome(self.state.metrics.step, genome);
        }

        self.logger.log_species(
            self.state.metrics.step,
            &self.state.predator.species,
            "predator",
        );
        self.logger
            .log_species(self.state.metrics.step, &self.state.prey.species, "prey");
        self.logger.flush();

        log::info!(
            "Step {:6}: predators={} prey={} pred_births={} prey_births={}",
            self.state.metrics.step,
            self.state.metrics.predator_count,
            self.state.metrics.prey_count,
            self.state.metrics.predator_births,
            self.state.metrics.prey_births,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_clone() {
        let cfg = AppConfig::default();
        let _cloned = cfg.clone();
        assert_eq!(cfg.speed_multiplier, 1);
        assert!(!cfg.headless);
    }
}