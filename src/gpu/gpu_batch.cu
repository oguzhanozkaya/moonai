#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

namespace moonai::gpu {

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs)
    : num_agents_(num_agents)
    , num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    CUDA_CHECK_ABORT(cudaMalloc(&d_descs_,   num_agents * sizeof(GpuNetDesc)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_inputs_,  num_agents * num_inputs  * sizeof(float)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_outputs_, num_agents * num_outputs * sizeof(float)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_stats_,   num_agents * sizeof(GpuAgentStats)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_fitness_, num_agents * sizeof(float)));
}

GpuBatch::~GpuBatch() {
    if (d_descs_)       cudaFree(d_descs_);
    if (d_inputs_)      cudaFree(d_inputs_);
    if (d_outputs_)     cudaFree(d_outputs_);
    if (d_stats_)       cudaFree(d_stats_);
    if (d_fitness_)     cudaFree(d_fitness_);

    if (d_node_vals_)   cudaFree(d_node_vals_);
    if (d_node_types_)  cudaFree(d_node_types_);
    if (d_eval_order_)  cudaFree(d_eval_order_);
    if (d_conn_ptr_)    cudaFree(d_conn_ptr_);
    if (d_in_count_)    cudaFree(d_in_count_);
    if (d_conn_from_)   cudaFree(d_conn_from_);
    if (d_conn_w_)      cudaFree(d_conn_w_);
    if (d_out_indices_) cudaFree(d_out_indices_);
}

// ── upload_network_data ───────────────────────────────────────────────────────

void GpuBatch::upload_network_data(const GpuNetworkData& data) {
    activation_fn_id_ = data.activation_fn_id;

    int n            = num_agents_;
    int total_nodes  = static_cast<int>(data.node_types.size());
    int total_eval   = static_cast<int>(data.eval_order.size());
    int total_conn   = static_cast<int>(data.conn_from.size());
    int total_out    = static_cast<int>(data.out_indices.size());

    // ── Free old topology allocations ────────────────────────────────────
    if (d_node_vals_)   { cudaFree(d_node_vals_);   d_node_vals_   = nullptr; }
    if (d_node_types_)  { cudaFree(d_node_types_);  d_node_types_  = nullptr; }
    if (d_eval_order_)  { cudaFree(d_eval_order_);  d_eval_order_  = nullptr; }
    if (d_conn_ptr_)    { cudaFree(d_conn_ptr_);    d_conn_ptr_    = nullptr; }
    if (d_in_count_)    { cudaFree(d_in_count_);    d_in_count_    = nullptr; }
    if (d_conn_from_)   { cudaFree(d_conn_from_);   d_conn_from_   = nullptr; }
    if (d_conn_w_)      { cudaFree(d_conn_w_);      d_conn_w_      = nullptr; }
    if (d_out_indices_) { cudaFree(d_out_indices_); d_out_indices_ = nullptr; }

    // ── Allocate device topology arrays ──────────────────────────────────
    CUDA_CHECK_ABORT(cudaMalloc(&d_node_vals_,  total_nodes * sizeof(float)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_node_types_, total_nodes * sizeof(uint8_t)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_eval_order_, total_eval  * sizeof(int)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_conn_ptr_,   total_eval  * sizeof(int)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_in_count_,   total_eval  * sizeof(int)));
    if (total_conn > 0) {
        CUDA_CHECK_ABORT(cudaMalloc(&d_conn_from_, total_conn * sizeof(int)));
        CUDA_CHECK_ABORT(cudaMalloc(&d_conn_w_,    total_conn * sizeof(float)));
    }
    if (total_out > 0) {
        CUDA_CHECK_ABORT(cudaMalloc(&d_out_indices_, total_out * sizeof(int)));
    }

    // ── Upload to device ─────────────────────────────────────────────────
    CUDA_CHECK_ABORT(cudaMemcpy(d_descs_, data.descs.data(),
        n * sizeof(GpuNetDesc), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_node_types_, data.node_types.data(),
        total_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_eval_order_, data.eval_order.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_conn_ptr_, data.conn_ptr.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_in_count_, data.in_count.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    if (total_conn > 0) {
        CUDA_CHECK_ABORT(cudaMemcpy(d_conn_from_, data.conn_from.data(),
            total_conn * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK_ABORT(cudaMemcpy(d_conn_w_, data.conn_w.data(),
            total_conn * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (total_out > 0) {
        CUDA_CHECK_ABORT(cudaMemcpy(d_out_indices_, data.out_indices.data(),
            total_out * sizeof(int), cudaMemcpyHostToDevice));
    }
    // d_node_vals_ is scratch — no initial upload needed; kernel initializes it each tick
}

// ── Per-tick I/O ─────────────────────────────────────────────────────────────

void GpuBatch::pack_inputs(const std::vector<float>& flat_inputs) {
    CUDA_CHECK(cudaMemcpy(d_inputs_, flat_inputs.data(),
        flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void GpuBatch::unpack_outputs(std::vector<float>& flat_out) const {
    flat_out.resize(num_agents_ * num_outputs_);
    CUDA_CHECK(cudaMemcpy(flat_out.data(), d_outputs_,
        flat_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

// ── Post-generation fitness ───────────────────────────────────────────────────

void GpuBatch::pack_agent_stats(const std::vector<GpuAgentStats>& stats) {
    CUDA_CHECK(cudaMemcpy(d_stats_, stats.data(),
        stats.size() * sizeof(GpuAgentStats), cudaMemcpyHostToDevice));
}

void GpuBatch::unpack_fitness(std::vector<float>& fitness_out) const {
    fitness_out.resize(num_agents_);
    CUDA_CHECK(cudaMemcpy(fitness_out.data(), d_fitness_,
        fitness_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

} // namespace moonai::gpu
