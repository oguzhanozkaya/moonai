use moonai_core::MetricsSnapshot;
use moonai_evo::Genome;

pub struct AgentMetricsData<'a> {
    pub energy: &'a [f32],
    pub genomes: &'a [Genome],
    pub generation: &'a [i32],
    pub species_count: usize,
}

pub struct FoodMetricsData<'a> {
    pub active: &'a [u8],
}

pub fn refresh_metrics(
    metrics: &mut MetricsSnapshot,
    step: i32,
    predator_data: AgentMetricsData,
    prey_data: AgentMetricsData,
    food_data: FoodMetricsData,
    predator_births: i32,
    prey_births: i32,
    predator_deaths: i32,
    prey_deaths: i32,
) {
    metrics.step = step;
    metrics.predator_count = predator_data.energy.len() as i32;
    metrics.prey_count = prey_data.energy.len() as i32;
    metrics.active_food = count_active_food(food_data.active);
    metrics.predator_species = predator_data.species_count as i32;
    metrics.prey_species = prey_data.species_count as i32;
    metrics.predator_births = predator_births;
    metrics.prey_births = prey_births;
    metrics.predator_deaths = predator_deaths;
    metrics.prey_deaths = prey_deaths;

    metrics.avg_predator_energy = calculate_avg_energy(predator_data.energy);
    metrics.avg_prey_energy = calculate_avg_energy(prey_data.energy);

    metrics.avg_predator_complexity = calculate_avg_complexity(predator_data.genomes);
    metrics.avg_prey_complexity = calculate_avg_complexity(prey_data.genomes);

    let (max_pred_gen, avg_pred_gen) = calculate_generation_metrics(predator_data.generation);
    metrics.max_predator_generation = max_pred_gen;
    metrics.avg_predator_generation = avg_pred_gen;

    let (max_prey_gen, avg_prey_gen) = calculate_generation_metrics(prey_data.generation);
    metrics.max_prey_generation = max_prey_gen;
    metrics.avg_prey_generation = avg_prey_gen;
}

fn count_active_food(active: &[u8]) -> i32 {
    active.iter().filter(|&&a| a != 0).count() as i32
}

fn calculate_avg_energy(energy: &[f32]) -> f32 {
    if energy.is_empty() {
        return 0.0f32;
    }
    let sum: f32 = energy.iter().sum();
    sum / energy.len() as f32
}

fn calculate_avg_complexity(genomes: &[Genome]) -> f32 {
    if genomes.is_empty() {
        return 0.0f32;
    }
    let sum: f32 = genomes
        .iter()
        .map(|g| (g.nodes().len() + g.connections().len()) as f32)
        .sum();
    sum / genomes.len() as f32
}

fn calculate_generation_metrics(generation: &[i32]) -> (i32, f32) {
    if generation.is_empty() {
        return (0, 0.0f32);
    }
    let max_gen = *generation.iter().max().unwrap_or(&0);
    let sum: i64 = generation.iter().map(|&g| g as i64).sum();
    let avg_gen = sum as f32 / generation.len() as f32;
    (max_gen, avg_gen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonai_evo::Genome;

    #[test]
    fn test_count_active_food() {
        let active = vec![1u8, 0, 1, 1, 0];
        assert_eq!(count_active_food(&active), 3);
    }

    #[test]
    fn test_calculate_avg_energy_empty() {
        let energy: Vec<f32> = vec![];
        assert_eq!(calculate_avg_energy(&energy), 0.0);
    }

    #[test]
    fn test_calculate_avg_energy() {
        let energy = vec![1.0, 2.0, 3.0];
        assert_eq!(calculate_avg_energy(&energy), 2.0);
    }

    #[test]
    fn test_calculate_avg_complexity_empty() {
        let genomes: Vec<Genome> = vec![];
        assert_eq!(calculate_avg_complexity(&genomes), 0.0);
    }

    #[test]
    fn test_calculate_generation_metrics_empty() {
        let generation: Vec<i32> = vec![];
        let (max, avg) = calculate_generation_metrics(&generation);
        assert_eq!(max, 0);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_calculate_generation_metrics() {
        let generation = vec![1, 2, 3, 4, 5];
        let (max, avg) = calculate_generation_metrics(&generation);
        assert_eq!(max, 5);
        assert_eq!(avg, 3.0);
    }
}