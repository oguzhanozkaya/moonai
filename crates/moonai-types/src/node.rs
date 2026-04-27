#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NodeType,
}

impl NodeGene {
    pub const fn new(id: u32, node_type: NodeType) -> Self {
        Self { id, node_type }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_gene_new() {
        let gene = NodeGene::new(0, NodeType::Input);
        assert_eq!(gene.id, 0);
        assert_eq!(gene.node_type, NodeType::Input);
    }

    #[test]
    fn node_type_variants() {
        assert_eq!(NodeType::Input, NodeType::Input);
        assert_eq!(NodeType::Hidden, NodeType::Hidden);
        assert_eq!(NodeType::Output, NodeType::Output);
        assert_eq!(NodeType::Bias, NodeType::Bias);
    }

    #[test]
    fn node_type_equality() {
        let a = NodeType::Hidden;
        let b = NodeType::Hidden;
        let c = NodeType::Output;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn node_gene_clone() {
        let gene = NodeGene::new(42, NodeType::Hidden);
        let cloned = gene.clone();
        assert_eq!(gene.id, cloned.id);
        assert_eq!(gene.node_type, cloned.node_type);
    }
}