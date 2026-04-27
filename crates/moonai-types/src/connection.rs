#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub in_node: u32,
    pub out_node: u32,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

impl ConnectionGene {
    pub const fn new(in_node: u32, out_node: u32, weight: f32, enabled: bool, innovation: u32) -> Self {
        Self { in_node, out_node, weight, enabled, innovation }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_gene_new() {
        let gene = ConnectionGene::new(0, 1, 0.5, true, 1);
        assert_eq!(gene.in_node, 0);
        assert_eq!(gene.out_node, 1);
        assert_eq!(gene.weight, 0.5);
        assert!(gene.enabled);
        assert_eq!(gene.innovation, 1);
    }

    #[test]
    fn connection_gene_disabled() {
        let gene = ConnectionGene::new(2, 3, -1.0, false, 5);
        assert!(!gene.enabled);
        assert_eq!(gene.weight, -1.0);
    }

    #[test]
    fn connection_gene_clone() {
        let gene = ConnectionGene::new(1, 2, 0.75, true, 3);
        let cloned = gene.clone();
        assert_eq!(gene.in_node, cloned.in_node);
        assert_eq!(gene.out_node, cloned.out_node);
        assert_eq!(gene.weight, cloned.weight);
        assert_eq!(gene.enabled, cloned.enabled);
        assert_eq!(gene.innovation, cloned.innovation);
    }
}