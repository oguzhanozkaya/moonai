#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

namespace moonai::gpu {

// One thread per agent. Computes fitness from pre-packed GpuAgentStats.
// Mirrors the CPU formula in EvolutionManager::compute_fitness().
__global__ void fitness_eval_kernel(
    const GpuAgentStats*  stats,
    float*                fitness_out,
    int                   num_agents,
    GpuFitnessWeights     weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;

    const GpuAgentStats& s = stats[idx];

    float f = weights.survival_w  * s.age_ratio
            + weights.kill_w      * s.kills_or_food
            + weights.energy_w    * s.energy_ratio
            + s.alive_bonus                          // coefficient is 1.0
            + weights.dist_w      * s.dist_ratio
            - weights.complexity_w * s.complexity;

    fitness_out[idx] = fmaxf(0.0f, f);
}

void batch_fitness_eval(GpuBatch& batch, GpuFitnessWeights weights) {
    int n = batch.num_agents();
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;

    fitness_eval_kernel<<<grid_size, kBlockSize>>>(
        batch.d_agent_stats(),
        batch.d_fitness_out(),
        n,
        weights
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

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
    printf("CUDA Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  SM count: %d\n", prop.multiProcessorCount);
}

} // namespace moonai::gpu
