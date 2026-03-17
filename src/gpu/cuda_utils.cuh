#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpu/gpu_types.hpp"

// Log the CUDA error but continue
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
        }                                                                        \
    } while (0)

// Log the CUDA error and abort (used for allocations that must succeed)
#define CUDA_CHECK_ABORT(call)                                                   \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA fatal error at %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                          \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

namespace moonai::gpu {

bool init_cuda();
void print_device_info();

} // namespace moonai::gpu
