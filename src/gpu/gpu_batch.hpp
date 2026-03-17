#pragma once

// gpu_batch.hpp — C++ header with NO CUDA dependency.
// Can be included from regular .cpp files when MOONAI_ENABLE_CUDA is ON.
// The CUDA implementation lives in gpu_batch.cu.

#include "gpu/gpu_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moonai::gpu {

// Pre-extracted flat network data (built on CPU, then uploaded to device).
// Separates the NeuralNetwork-specific packing logic from the GPU batch class
// so that moonai_gpu has no link-time dependency on moonai_evolution.
struct GpuNetworkData {
    std::vector<GpuNetDesc> descs;       // one per agent
    std::vector<uint8_t>    node_types;  // flat: 0=Input,1=Bias,2=Hidden,3=Output
    std::vector<int>        eval_order;  // flat: node indices in topo order per agent
    std::vector<int>        conn_ptr;    // flat: start idx in conn arrays per eval step
    std::vector<int>        in_count;    // flat: # incoming edges per eval step
    std::vector<int>        conn_from;   // flat: source node index per edge
    std::vector<float>      conn_w;      // flat: edge weight per edge
    std::vector<int>        out_indices; // flat: output node positions per agent
    int activation_fn_id = 0;           // 0=sigmoid, 1=tanh, 2=relu
};

// RAII wrapper owning all device memory for a generation's batch inference.
// Fixed-size arrays (inputs, outputs, descs, stats, fitness) are allocated
// in the constructor. Topology arrays are freed and reallocated each
// generation by upload_network_data().
class GpuBatch {
public:
    GpuBatch(int num_agents, int num_inputs, int num_outputs);
    ~GpuBatch();

    GpuBatch(const GpuBatch&)            = delete;
    GpuBatch& operator=(const GpuBatch&) = delete;

    // ── Per-generation setup ─────────────────────────────────────────────
    // Accepts pre-extracted network data (built by caller) and uploads to GPU.
    void upload_network_data(const GpuNetworkData& data);

    // ── Per-tick I/O ─────────────────────────────────────────────────────
    // flat_inputs: size == num_agents * num_inputs (agent-major order)
    void pack_inputs(const std::vector<float>& flat_inputs);
    // flat_out: size == num_agents * num_outputs
    void unpack_outputs(std::vector<float>& flat_out) const;

    // ── Post-generation fitness ──────────────────────────────────────────
    void pack_agent_stats(const std::vector<GpuAgentStats>& stats);
    void unpack_fitness(std::vector<float>& fitness_out) const;

    // ── Kernel-facing accessors (device pointers) ────────────────────────
    const GpuNetDesc* d_descs()       const { return d_descs_; }
    float*            d_node_vals()         { return d_node_vals_; }
    const uint8_t*    d_node_types()  const { return d_node_types_; }
    const int*        d_eval_order()  const { return d_eval_order_; }
    const int*        d_conn_ptr()    const { return d_conn_ptr_; }
    const int*        d_in_count()    const { return d_in_count_; }
    const int*        d_conn_from()   const { return d_conn_from_; }
    const float*      d_conn_w()      const { return d_conn_w_; }
    const int*        d_out_indices() const { return d_out_indices_; }
    const float*      d_inputs()      const { return d_inputs_; }
    float*            d_outputs()           { return d_outputs_; }
    const GpuAgentStats* d_agent_stats() const { return d_stats_; }
    float*            d_fitness_out()       { return d_fitness_; }

    int num_agents()       const { return num_agents_; }
    int num_inputs()       const { return num_inputs_; }
    int num_outputs()      const { return num_outputs_; }
    int activation_fn_id() const { return activation_fn_id_; }

private:
    // Fixed-size device arrays (allocated in constructor, freed in destructor)
    GpuNetDesc*    d_descs_     = nullptr;  // [num_agents]
    float*         d_inputs_    = nullptr;  // [num_agents * num_inputs]
    float*         d_outputs_   = nullptr;  // [num_agents * num_outputs]
    GpuAgentStats* d_stats_     = nullptr;  // [num_agents]
    float*         d_fitness_   = nullptr;  // [num_agents]

    // Topology arrays (reallocated each generation by upload_network_data)
    float*   d_node_vals_   = nullptr;  // [Σ num_nodes]   — scratch per tick
    uint8_t* d_node_types_  = nullptr;  // [Σ num_nodes]   — node kind
    int*     d_eval_order_  = nullptr;  // [Σ num_eval]    — node indices in topo order
    int*     d_conn_ptr_    = nullptr;  // [Σ num_eval]    — start in conn arrays
    int*     d_in_count_    = nullptr;  // [Σ num_eval]    — # incoming edges
    int*     d_conn_from_   = nullptr;  // [Σ total_conn]  — source node index
    float*   d_conn_w_      = nullptr;  // [Σ total_conn]  — edge weight
    int*     d_out_indices_ = nullptr;  // [Σ num_outputs] — output node positions

    int num_agents_;
    int num_inputs_;
    int num_outputs_;
    int activation_fn_id_ = 0;  // 0=sigmoid, 1=tanh, 2=relu
};

// ── Free functions (implemented in neural_inference.cu / fitness_eval.cu) ──
void batch_neural_inference(GpuBatch& batch);
void batch_fitness_eval(GpuBatch& batch, GpuFitnessWeights weights);

} // namespace moonai::gpu
