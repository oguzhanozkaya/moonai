#pragma once

namespace moonai::gpu {

// Per-agent network descriptor for CSR-packed flat GPU layout
struct GpuNetDesc {
    int num_nodes;   // total node count (input+bias+hidden+output)
    int num_eval;    // evaluation order length (hidden+output only)
    int num_inputs;  // number of Input type nodes (excluding bias)
    int num_outputs; // number of output nodes
    int node_off;    // offset into d_node_vals[], d_node_types[]
    int eval_off;    // offset into d_eval_order[], d_conn_ptr[], d_in_count[]
    int conn_off;    // offset into d_conn_from[], d_conn_w[]
    int out_off;     // offset into d_out_indices[]
};

// Per-agent stats for GPU fitness evaluation (pre-computed ratios on CPU)
struct GpuAgentStats {
    float age_ratio;     // age / generation_ticks
    float kills_or_food; // kills (predator) or food_eaten (prey)
    float energy_ratio;  // max(0, energy) / initial_energy
    float alive_bonus;   // 1.0f if alive at end, else 0.0f
    float dist_ratio;    // distance_traveled / max_possible_dist (clamped 0..1)
    float complexity;    // genome.complexity() as float
};

// Fitness weight parameters passed to the GPU fitness kernel
struct GpuFitnessWeights {
    float survival_w;
    float kill_w;
    float energy_w;
    float dist_w;
    float complexity_w;
};

} // namespace moonai::gpu
