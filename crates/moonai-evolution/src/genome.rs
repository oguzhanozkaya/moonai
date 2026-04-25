pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NodeType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub in_node: u32,
    pub out_node: u32,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

impl Genome {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self { nodes: Vec::new(), connections: Vec::new(), num_inputs, num_outputs }
    }
}
