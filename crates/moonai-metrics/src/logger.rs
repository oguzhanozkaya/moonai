use std::path::Path;

pub struct Logger;

impl Logger {
    pub fn new(_output_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub fn log_stats(&mut self, _step: u64) {}
    pub fn log_species(&mut self, _step: u64) {}
    pub fn log_genomes(&mut self, _step: u64) {}
}
