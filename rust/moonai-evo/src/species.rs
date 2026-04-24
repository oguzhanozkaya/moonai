use crate::genome::Genome;

static mut SPECIES_ID_COUNTER: i32 = 0;

pub struct Species {
    id: i32,
    representative: Genome,
    members: Vec<SpeciesMember>,
    average_complexity: f32,
}

#[derive(Clone, Debug)]
pub struct SpeciesMember {
    pub entity: u32,
    pub complexity: i32,
}

impl Species {
    pub fn new(representative: &Genome) -> Self {
        let id = unsafe {
            let current = SPECIES_ID_COUNTER;
            SPECIES_ID_COUNTER += 1;
            current
        };

        Self {
            id,
            representative: representative.clone(),
            members: Vec::new(),
            average_complexity: 0.0,
        }
    }

    pub fn is_compatible(
        &self,
        genome: &Genome,
        threshold: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        min_normalization: f32,
    ) -> bool {
        Genome::compatibility_distance(
            &self.representative,
            genome,
            c1,
            c2,
            c3,
            min_normalization,
        ) < threshold
    }

    pub fn add_member(&mut self, entity: u32, genome: &Genome) {
        self.members.push(SpeciesMember {
            entity,
            complexity: genome.complexity(),
        });
    }

    pub fn clear_members(&mut self) {
        self.members.clear();
        self.average_complexity = 0.0;
    }

    pub fn refresh_summary(&mut self) {
        if self.members.is_empty() {
            self.average_complexity = 0.0;
            return;
        }

        let size = self.members.len() as f32;
        let total_complexity: f32 = self.members.iter().map(|m| m.complexity as f32).sum();
        self.average_complexity = total_complexity / size;
    }

    pub fn set_representative(&mut self, genome: &Genome) {
        self.representative = genome.clone();
    }

    pub fn representative(&self) -> &Genome {
        &self.representative
    }

    pub fn members(&self) -> &[SpeciesMember] {
        &self.members
    }

    pub fn average_complexity(&self) -> f32 {
        self.average_complexity
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn next_species_id() -> i32 {
        unsafe {
            let current = SPECIES_ID_COUNTER;
            SPECIES_ID_COUNTER += 1;
            current
        }
    }

    pub fn reset_id_counter() {
        unsafe {
            SPECIES_ID_COUNTER = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_species_counter() {
        Species::reset_id_counter();
    }

    #[test]
    fn test_species_creation() {
        reset_species_counter();
        let genome = Genome::new(2, 1);
        let species = Species::new(&genome);
        assert_eq!(species.id(), 0);
    }

    #[test]
    fn test_species_is_compatible() {
        reset_species_counter();
        let genome = Genome::new(2, 1);
        let species = Species::new(&genome);

        assert!(species.is_compatible(&genome, 10.0, 1.0, 1.0, 0.4, 1.0));
    }

    #[test]
    fn test_species_add_member() {
        reset_species_counter();
        let genome = Genome::new(2, 1);
        let mut species = Species::new(&genome);
        species.add_member(1, &genome);
        assert_eq!(species.members().len(), 1);
    }

    #[test]
    fn test_species_refresh_summary() {
        reset_species_counter();
        let genome = Genome::new(2, 1);
        let mut species = Species::new(&genome);
        species.add_member(1, &genome);
        species.add_member(2, &genome);
        species.refresh_summary();
        assert!(species.average_complexity() > 0.0);
    }

    #[test]
    fn test_reset_id_counter() {
        Species::reset_id_counter();
        let genome = Genome::new(2, 1);
        let species1 = Species::new(&genome);
        let species2 = Species::new(&genome);
        assert_eq!(species1.id(), 0);
        assert_eq!(species2.id(), 1);
    }
}