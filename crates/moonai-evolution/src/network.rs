pub struct NeuralNetwork {
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self { inputs: Vec::new(), outputs: Vec::new() }
    }
}
