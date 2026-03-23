#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

#include <spdlog/spdlog.h>

namespace moonai::gpu {

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs)
    : num_agents_(num_agents)
    , num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    size_t in_bytes  = num_agents * num_inputs  * sizeof(float);
    size_t out_bytes = num_agents * num_outputs * sizeof(float);

    // Device arrays
    CUDA_CHECK_ABORT(cudaMalloc(&d_descs_,   num_agents * sizeof(GpuNetDesc)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_inputs_,  in_bytes));
    CUDA_CHECK_ABORT(cudaMalloc(&d_outputs_, out_bytes));

    // Pinned host memory for async transfers
    CUDA_CHECK_ABORT(cudaMallocHost(&h_pinned_in_,  in_bytes));
    CUDA_CHECK_ABORT(cudaMallocHost(&h_pinned_out_, out_bytes));

    // Single CUDA stream
    cudaStream_t s;
    CUDA_CHECK_ABORT(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

GpuBatch::~GpuBatch() {
    // Device arrays
    if (d_descs_)       cudaFree(d_descs_);
    if (d_inputs_)      cudaFree(d_inputs_);
    if (d_outputs_)     cudaFree(d_outputs_);

    // Pinned host memory
    if (h_pinned_in_)   cudaFreeHost(h_pinned_in_);
    if (h_pinned_out_)  cudaFreeHost(h_pinned_out_);

    // Stream
    if (stream_)        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));

    // Topology arrays
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

    cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // ── Reallocate topology arrays only when capacity is exceeded ──────
    auto realloc_if_needed = [&](auto** ptr, int count, int& capacity, size_t elem_size) {
        if (count > capacity) {
            if (*ptr) cudaFree(*ptr);
            CUDA_CHECK_ABORT(cudaMalloc(ptr, count * elem_size));
            capacity = count;
        }
    };

    realloc_if_needed(&d_node_vals_,  total_nodes, capacity_nodes_, sizeof(float));
    realloc_if_needed(&d_node_types_, total_nodes, capacity_nodes_, sizeof(uint8_t));
    realloc_if_needed(&d_eval_order_, total_eval,  capacity_eval_,  sizeof(int));
    realloc_if_needed(&d_conn_ptr_,   total_eval,  capacity_eval_,  sizeof(int));
    realloc_if_needed(&d_in_count_,   total_eval,  capacity_eval_,  sizeof(int));
    if (total_conn > 0) {
        realloc_if_needed(&d_conn_from_, total_conn, capacity_conn_, sizeof(int));
        realloc_if_needed(&d_conn_w_,    total_conn, capacity_conn_, sizeof(float));
    }
    if (total_out > 0) {
        realloc_if_needed(&d_out_indices_, total_out, capacity_out_, sizeof(int));
    }

    // ── Upload to device (async on batch stream) ──────────────────────
    CUDA_CHECK(cudaMemcpyAsync(d_descs_, data.descs.data(),
        n * sizeof(GpuNetDesc), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_node_types_, data.node_types.data(),
        total_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_eval_order_, data.eval_order.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_conn_ptr_, data.conn_ptr.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_in_count_, data.in_count.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s));
    if (total_conn > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_conn_from_, data.conn_from.data(),
            total_conn * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d_conn_w_, data.conn_w.data(),
            total_conn * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    if (total_out > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_out_indices_, data.out_indices.data(),
            total_out * sizeof(int), cudaMemcpyHostToDevice, s));
    }

    // Ensure upload completes before first inference tick
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ── Per-tick I/O ─────────────────────────────────────────────────────────────

void GpuBatch::pack_inputs_async(const float* flat_inputs, int count) {
    size_t bytes = count * sizeof(float);
    memcpy(h_pinned_in_, flat_inputs, bytes);
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_inputs_, h_pinned_in_,
        bytes, cudaMemcpyHostToDevice, s));
}

void GpuBatch::launch_inference_async() {
    batch_neural_inference(*this);
}

void GpuBatch::start_unpack_async() {
    size_t bytes = num_agents_ * num_outputs_ * sizeof(float);
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out_, d_outputs_,
        bytes, cudaMemcpyDeviceToHost, s));
}

void GpuBatch::finish_unpack(float* dst, int count) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    memcpy(dst, h_pinned_out_, count * sizeof(float));
}

// ── CUDA initialization ──────────────────────────────────────────────────────

bool init_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(0));
    return true;
}

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    spdlog::info("CUDA Device: {}", prop.name);
    spdlog::info("  Compute capability: {}.{}", prop.major, prop.minor);
    spdlog::info("  Total memory: {:.1f} MB", prop.totalGlobalMem / (1024.0 * 1024.0));
    spdlog::info("  SM count: {}", prop.multiProcessorCount);
}

} // namespace moonai::gpu
