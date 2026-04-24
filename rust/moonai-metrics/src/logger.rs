use chrono::Local;
use moonai_core::{save_config, SimulationConfig};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use moonai_evo::{Genome, Species};
use moonai_core::MetricsSnapshot;

pub struct Logger {
    base_dir: String,
    run_dir: String,
    name: String,
    seed: i32,
    stats_file: Option<BufWriter<File>>,
    genomes_file: Option<BufWriter<File>>,
    species_file: Option<BufWriter<File>>,
    genomes_first_entry: bool,
}

impl Logger {
    pub fn new(output_dir: &str, seed: i32, name: &str) -> Self {
        Self {
            base_dir: output_dir.to_string(),
            run_dir: String::new(),
            name: name.to_string(),
            seed,
            stats_file: None,
            genomes_file: None,
            species_file: None,
            genomes_first_entry: true,
        }
    }

    pub fn initialize(&mut self, config: &SimulationConfig) -> Result<(), String> {
        let dir_name = if !self.name.is_empty() {
            self.name.clone()
        } else {
            let now = Local::now();
            format!("{}_{}_seed{}", now.format("%Y%m%d_%H%M%S"), now.format("%S"), self.seed)
        };

        self.run_dir = format!("{}/{}", self.base_dir, dir_name);

        let mut candidate_dir = self.run_dir.clone();
        let mut suffix = 2;
        while fs::metadata(&candidate_dir).is_ok() {
            candidate_dir = format!("{}_{}", self.run_dir, suffix);
            suffix += 1;
            if suffix > 1000 {
                return Err("Could not find available directory after 1000 attempts".to_string());
            }
        }
        self.run_dir = candidate_dir;

        fs::create_dir_all(&self.run_dir).map_err(|e| e.to_string())?;

        let config_path = PathBuf::from(&self.run_dir).join("config.json");
        save_config(config, &config_path).map_err(|e| e.to_string())?;

        self.open_stats_file()?;
        self.open_species_file()?;
        self.open_genomes_file()?;

        Ok(())
    }

    fn open_stats_file(&mut self) -> Result<(), String> {
        let stats_path = PathBuf::from(&self.run_dir).join("stats.csv");
        let file = File::create(stats_path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# master_seed={}", self.seed).map_err(|e| e.to_string())?;
        writeln!(
            writer,
            "step,predator_count,prey_count,predator_births,prey_births,predator_deaths,prey_deaths,predator_species,prey_species,avg_predator_complexity,avg_prey_complexity,avg_predator_energy,avg_prey_energy,max_predator_generation,avg_predator_generation,max_prey_generation,avg_prey_generation"
        )
        .map_err(|e| e.to_string())?;

        self.stats_file = Some(writer);
        Ok(())
    }

    fn open_species_file(&mut self) -> Result<(), String> {
        let species_path = PathBuf::from(&self.run_dir).join("species.csv");
        let file = File::create(species_path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# master_seed={}", self.seed).map_err(|e| e.to_string())?;
        writeln!(writer, "step,population,species_id,size,avg_complexity").map_err(|e| e.to_string())?;

        self.species_file = Some(writer);
        Ok(())
    }

    fn open_genomes_file(&mut self) -> Result<(), String> {
        let genomes_path = PathBuf::from(&self.run_dir).join("genomes.json");
        let file = File::create(genomes_path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);

        write!(writer, "[").map_err(|e| e.to_string())?;

        self.genomes_file = Some(writer);
        Ok(())
    }

    pub fn log_report(&mut self, metrics: &MetricsSnapshot) {
        if let Some(ref mut writer) = self.stats_file {
            let result = writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                metrics.step,
                metrics.predator_count,
                metrics.prey_count,
                metrics.predator_births,
                metrics.prey_births,
                metrics.predator_deaths,
                metrics.prey_deaths,
                metrics.predator_species,
                metrics.prey_species,
                metrics.avg_predator_complexity,
                metrics.avg_prey_complexity,
                metrics.avg_predator_energy,
                metrics.avg_prey_energy,
                metrics.max_predator_generation,
                metrics.avg_predator_generation,
                metrics.max_prey_generation,
                metrics.avg_prey_generation
            );
            if result.is_err() {
                eprintln!("Error writing stats: {:?}", result);
            }
        }
    }

    pub fn log_best_genome(&mut self, step: i32, genome: &Genome) {
        if let Some(ref mut writer) = self.genomes_file {
            let j = serde_json::json!({
                "step": step,
                "num_nodes": genome.nodes().len(),
                "num_connections": genome.connections().len(),
                "genome": serde_json::from_str::<serde_json::Value>(&genome.to_json()).unwrap_or(serde_json::Value::Null),
            });

            if !self.genomes_first_entry {
                if let Err(e) = write!(writer, ",") {
                    eprintln!("Error writing comma: {:?}", e);
                    return;
                }
            }
            self.genomes_first_entry = false;

            if let Err(e) = writeln!(writer, "\n{}", j.to_string()) {
                eprintln!("Error writing genome: {:?}", e);
            }
        }
    }

    pub fn log_species(&mut self, step: i32, species: &[Species], population_name: &str) {
        if let Some(ref mut writer) = self.species_file {
            for entry in species {
                if let Err(e) = writeln!(
                    writer,
                    "{},{},{},{},{}",
                    step,
                    population_name,
                    entry.id(),
                    entry.members().len(),
                    entry.average_complexity()
                ) {
                    eprintln!("Error writing species: {:?}", e);
                }
            }
        }
    }

    pub fn flush(&mut self) {
        if let Some(ref mut writer) = self.stats_file {
            let _ = writer.flush();
        }
        if let Some(ref mut writer) = self.genomes_file {
            let _ = writer.flush();
        }
        if let Some(ref mut writer) = self.species_file {
            let _ = writer.flush();
        }
    }

    pub fn run_dir(&self) -> &str {
        &self.run_dir
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        if let Some(ref mut writer) = self.genomes_file {
            let _ = writeln!(writer, "\n]");
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonai_core::SimulationConfig;
    use std::fs;

    #[test]
    fn test_logger_creation() {
        let logger = Logger::new("output/test", 42, "test_run");
        assert_eq!(logger.seed, 42);
        assert_eq!(logger.run_dir(), "");
    }

    #[test]
    fn test_logger_initialize() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("moonai_test_logger_init");

        let mut logger = Logger::new(test_dir.to_str().unwrap(), 42, "test_init");
        let config = SimulationConfig::default();

        let result = logger.initialize(&config);
        assert!(result.is_ok());

        let run_dir = PathBuf::from(logger.run_dir());
        assert!(fs::metadata(&run_dir.join("config.json")).is_ok());
        assert!(fs::metadata(&run_dir.join("stats.csv")).is_ok());
        assert!(fs::metadata(&run_dir.join("species.csv")).is_ok());
        assert!(fs::metadata(&run_dir.join("genomes.json")).is_ok());

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_logger_log_report() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("moonai_test_logger_log");

        let mut logger = Logger::new(test_dir.to_str().unwrap(), 42, "test_log");
        let config = SimulationConfig::default();

        logger.initialize(&config).unwrap();

        let mut metrics = MetricsSnapshot::default();
        metrics.step = 100;
        metrics.predator_count = 500;
        metrics.prey_count = 2000;

        logger.log_report(&metrics);
        logger.flush();

        let stats_content = fs::read_to_string(PathBuf::from(&logger.run_dir()).join("stats.csv")).unwrap();
        assert!(stats_content.contains("100"));
        assert!(stats_content.contains("500"));
        assert!(stats_content.contains("2000"));

        let _ = fs::remove_dir_all(&test_dir);
    }
}