#include "core/profiler_macros.hpp"
#include "core/types.hpp"
#include "evolution/inference_cache.hpp"
#include "evolution/network_cache.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/spdlog.h>

namespace moonai::evolution {

namespace {

constexpr int kBlockSize = 256;
constexpr int kOutputSlots = OUTPUT_COUNT;
constexpr int kSensorInputs = SENSOR_COUNT;

__device__ __forceinline__ float activate(float x) {
  return __tanhf(x);
}

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      spdlog::error("CUDA error in {} at {}: {}", #call, __FILE__, __LINE__, cudaGetErrorString(err));                 \
    }                                                                                                                  \
  } while (0)

} // namespace

__global__ void kernel_neural_inference(const NetworkDescriptor *__restrict__ descriptors,
                                        float *__restrict__ node_values, const int *__restrict__ eval_order,
                                        const int *__restrict__ conn_from, const float *__restrict__ conn_weights,
                                        const int *__restrict__ conn_ptr, const int *__restrict__ out_indices,
                                        const float *__restrict__ sensor_inputs, float *__restrict__ brain_outputs,
                                        int network_count) {
  const int network_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (network_idx >= network_count) {
    return;
  }

  const int slot = network_idx;
  const NetworkDescriptor &desc = descriptors[network_idx];

  float *my_nodes = node_values + desc.node_off;
  const int *my_eval = eval_order + desc.eval_off;
  const int *my_conn_from = conn_from + desc.conn_off;
  const float *my_weights = conn_weights + desc.conn_off;
  const int *my_conn_ptr_local = conn_ptr + desc.ptr_off;
  const int *my_out = out_indices + desc.out_off;

  const float *my_inputs = sensor_inputs + slot * kSensorInputs;
  for (int i = 0; i < desc.num_inputs; ++i) {
    my_nodes[i] = my_inputs[i];
  }
  my_nodes[desc.num_inputs] = 1.0f;

  for (int e = 0; e < desc.num_eval; ++e) {
    const int node_idx = my_eval[e];
    float sum = 0.0f;
    const int start = my_conn_ptr_local[e];
    const int end = my_conn_ptr_local[e + 1];

    for (int c = start; c < end; ++c) {
      sum += my_nodes[my_conn_from[c]] * my_weights[c];
    }

    my_nodes[node_idx] = activate(sum);
  }

  float *my_outputs = brain_outputs + slot * kOutputSlots;
  for (int i = 0; i < desc.num_outputs && i < kOutputSlots; ++i) {
    my_outputs[i] = my_nodes[my_out[i]];
  }
}

InferenceCache::InferenceCache() = default;

InferenceCache::~InferenceCache() {
  free_device_memory();
}

void InferenceCache::clear() {
  entries_.clear();
  slot_to_entry_.clear();
  free_entries_.clear();
  launch_descriptors_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  pending_entry_uploads_.clear();
  entry_upload_pending_.clear();

  dirty_ = true;
  launch_dirty_ = true;
  full_upload_required_ = false;
  node_extent_ = 0;
  eval_extent_ = 0;
  conn_extent_ = 0;
  ptr_extent_ = 0;
  output_extent_ = 0;

  free_device_memory();
}

void InferenceCache::allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity,
                                            std::size_t conn_capacity, std::size_t ptr_capacity,
                                            std::size_t output_capacity, std::size_t entity_capacity) {
  free_device_memory();

  CUDA_CHECK(cudaMalloc(&d_node_values_, node_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_eval_order_, eval_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_from_, conn_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_weights_, conn_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conn_ptr_, ptr_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_indices_, output_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_descriptors_, entity_capacity * sizeof(NetworkDescriptor)));

  entity_capacity_ = entity_capacity;
  node_capacity_ = node_capacity;
  eval_capacity_ = eval_capacity;
  conn_capacity_ = conn_capacity;
  ptr_capacity_ = ptr_capacity;
  output_capacity_ = output_capacity;
}

bool InferenceCache::needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                                        std::size_t ptr_capacity, std::size_t output_capacity,
                                        std::size_t entity_capacity) const {
  return entity_capacity > entity_capacity_ || node_capacity > node_capacity_ || eval_capacity > eval_capacity_ ||
         conn_capacity > conn_capacity_ || ptr_capacity > ptr_capacity_ || output_capacity > output_capacity_;
}

void InferenceCache::free_device_memory() {
  if (d_node_values_) {
    cudaFree(d_node_values_);
    d_node_values_ = nullptr;
  }
  if (d_eval_order_) {
    cudaFree(d_eval_order_);
    d_eval_order_ = nullptr;
  }
  if (d_conn_from_) {
    cudaFree(d_conn_from_);
    d_conn_from_ = nullptr;
  }
  if (d_conn_weights_) {
    cudaFree(d_conn_weights_);
    d_conn_weights_ = nullptr;
  }
  if (d_conn_ptr_) {
    cudaFree(d_conn_ptr_);
    d_conn_ptr_ = nullptr;
  }
  if (d_out_indices_) {
    cudaFree(d_out_indices_);
    d_out_indices_ = nullptr;
  }
  if (d_descriptors_) {
    cudaFree(d_descriptors_);
    d_descriptors_ = nullptr;
  }
  entity_capacity_ = 0;
  node_capacity_ = 0;
  eval_capacity_ = 0;
  conn_capacity_ = 0;
  ptr_capacity_ = 0;
  output_capacity_ = 0;
}

namespace {

std::size_t grow_capacity(std::size_t current, std::size_t required) {
  if (required == 0) {
    return 0;
  }

  std::size_t capacity = std::max<std::size_t>(1, current);
  while (capacity < required) {
    capacity *= 2;
  }
  return capacity;
}

} // namespace

void InferenceCache::assign_entry_data(Entry &entry, const CompiledNetwork &compiled) {
  entry.descriptor.num_nodes = compiled.num_nodes;
  entry.descriptor.num_eval = compiled.num_eval();
  entry.descriptor.num_inputs = compiled.num_inputs;
  entry.descriptor.num_outputs = compiled.num_outputs;
  entry.descriptor.padding0 = 0;
  entry.descriptor.padding = 0;

  std::copy(compiled.eval_order.begin(), compiled.eval_order.end(),
            h_eval_order_.begin() + entry.descriptor.eval_off);
  std::copy(compiled.conn_from.begin(), compiled.conn_from.end(), h_conn_from_.begin() + entry.descriptor.conn_off);
  std::copy(compiled.conn_weights.begin(), compiled.conn_weights.end(),
            h_conn_weights_.begin() + entry.descriptor.conn_off);
  std::copy(compiled.conn_ptr.begin(), compiled.conn_ptr.end(), h_conn_ptr_.begin() + entry.descriptor.ptr_off);
  std::copy(compiled.output_indices.begin(), compiled.output_indices.end(),
            h_out_indices_.begin() + entry.descriptor.out_off);
}

uint32_t InferenceCache::acquire_entry(const CompiledNetwork &compiled) {
  const auto free_it = std::find_if(free_entries_.begin(), free_entries_.end(), [&](const uint32_t entry_index) {
    const Entry &entry = entries_[entry_index];
    return compiled.num_nodes <= entry.node_capacity && compiled.num_eval() <= entry.eval_capacity &&
           compiled.num_connections() <= entry.conn_capacity && static_cast<int>(compiled.conn_ptr.size()) <= entry.ptr_capacity;
  });

  if (free_it != free_entries_.end()) {
    const uint32_t entry_index = *free_it;
    free_entries_.erase(free_it);
    assign_entry_data(entries_[entry_index], compiled);
    mark_entry_for_upload(entry_index);
    return entry_index;
  }

  Entry entry;
  entry.descriptor.node_off = static_cast<int>(node_extent_);
  entry.descriptor.eval_off = static_cast<int>(eval_extent_);
  entry.descriptor.conn_off = static_cast<int>(conn_extent_);
  entry.descriptor.ptr_off = static_cast<int>(ptr_extent_);
  entry.descriptor.out_off = static_cast<int>(output_extent_);
  entry.node_capacity = compiled.num_nodes;
  entry.eval_capacity = compiled.num_eval();
  entry.conn_capacity = compiled.num_connections();
  entry.ptr_capacity = static_cast<int>(compiled.conn_ptr.size());

  node_extent_ += static_cast<std::size_t>(entry.node_capacity);
  eval_extent_ += static_cast<std::size_t>(entry.eval_capacity);
  conn_extent_ += static_cast<std::size_t>(entry.conn_capacity);
  ptr_extent_ += static_cast<std::size_t>(entry.ptr_capacity);
  output_extent_ += compiled.output_indices.size();

  h_eval_order_.resize(eval_extent_);
  h_conn_from_.resize(conn_extent_);
  h_conn_weights_.resize(conn_extent_);
  h_conn_ptr_.resize(ptr_extent_);
  h_out_indices_.resize(output_extent_);

  assign_entry_data(entry, compiled);

  const uint32_t entry_index = static_cast<uint32_t>(entries_.size());
  entries_.push_back(entry);
  entry_upload_pending_.push_back(0);
  full_upload_required_ = true;
  return entry_index;
}

void InferenceCache::mark_entry_for_upload(uint32_t entry_index) {
  if (entry_index >= entry_upload_pending_.size()) {
    entry_upload_pending_.resize(static_cast<std::size_t>(entry_index) + 1, 0);
  }
  if (entry_upload_pending_[entry_index] != 0) {
    return;
  }
  entry_upload_pending_[entry_index] = 1;
  pending_entry_uploads_.push_back(entry_index);
}

void InferenceCache::rebuild_launch_descriptors() {
  launch_descriptors_.resize(slot_to_entry_.size());
  for (std::size_t slot = 0; slot < slot_to_entry_.size(); ++slot) {
    launch_descriptors_[slot] = entries_[slot_to_entry_[slot]].descriptor;
  }
}

void InferenceCache::upload_full(cudaStream_t stream) {
  MOONAI_PROFILE_SCOPE("inference_cache_upload", stream);

  if (!h_eval_order_.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(d_eval_order_, h_eval_order_.data(), h_eval_order_.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }
  if (!h_conn_from_.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(d_conn_from_, h_conn_from_.data(), h_conn_from_.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_conn_weights_, h_conn_weights_.data(), h_conn_weights_.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  }
  if (!h_conn_ptr_.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(d_conn_ptr_, h_conn_ptr_.data(), h_conn_ptr_.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }
  if (!h_out_indices_.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(d_out_indices_, h_out_indices_.data(), h_out_indices_.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }
}

void InferenceCache::upload_pending(cudaStream_t stream) {
  if (pending_entry_uploads_.empty()) {
    return;
  }

  MOONAI_PROFILE_SCOPE("inference_cache_upload", stream);

  for (const uint32_t entry_index : pending_entry_uploads_) {
    const Entry &entry = entries_[entry_index];

    if (entry.descriptor.num_eval > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_eval_order_ + entry.descriptor.eval_off,
                                 h_eval_order_.data() + entry.descriptor.eval_off,
                                 static_cast<std::size_t>(entry.descriptor.num_eval) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
    }

    if (entry.conn_capacity > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_conn_from_ + entry.descriptor.conn_off, h_conn_from_.data() + entry.descriptor.conn_off,
                                 static_cast<std::size_t>(entry.conn_capacity) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_conn_weights_ + entry.descriptor.conn_off,
                                 h_conn_weights_.data() + entry.descriptor.conn_off,
                                 static_cast<std::size_t>(entry.conn_capacity) * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    }

    if (entry.ptr_capacity > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_conn_ptr_ + entry.descriptor.ptr_off, h_conn_ptr_.data() + entry.descriptor.ptr_off,
                                 static_cast<std::size_t>(entry.ptr_capacity) * sizeof(int), cudaMemcpyHostToDevice,
                                 stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_out_indices_ + entry.descriptor.out_off, h_out_indices_.data() + entry.descriptor.out_off,
                               kOutputSlots * sizeof(int), cudaMemcpyHostToDevice, stream));

    entry_upload_pending_[entry_index] = 0;
  }

  pending_entry_uploads_.clear();
}

void InferenceCache::build_from(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream) {
  MOONAI_PROFILE_SCOPE("inference_cache_build");

  entries_.clear();
  slot_to_entry_.clear();
  free_entries_.clear();
  launch_descriptors_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  pending_entry_uploads_.clear();
  entry_upload_pending_.clear();

  node_extent_ = 0;
  eval_extent_ = 0;
  conn_extent_ = 0;
  ptr_extent_ = 0;
  output_extent_ = 0;

  entries_.reserve(count);
  slot_to_entry_.reserve(count);
  entry_upload_pending_.reserve(count);

  for (uint32_t slot = 0; slot < count; ++slot) {
    const CompiledNetwork *compiled = network_cache.get_compiled(slot);
    if (!compiled) {
      spdlog::error("Missing compiled network for slot {}", slot);
      clear();
      return;
    }

    const uint32_t entry_index = acquire_entry(*compiled);
    slot_to_entry_.push_back(entry_index);
  }

  rebuild_launch_descriptors();

  if (needs_reallocation(node_extent_, eval_extent_, conn_extent_, ptr_extent_, output_extent_, slot_to_entry_.size())) {
    allocate_device_memory(grow_capacity(node_capacity_, node_extent_), grow_capacity(eval_capacity_, eval_extent_),
                           grow_capacity(conn_capacity_, conn_extent_), grow_capacity(ptr_capacity_, ptr_extent_),
                           grow_capacity(output_capacity_, output_extent_),
                           grow_capacity(entity_capacity_, slot_to_entry_.size()));
  }

  upload_full(stream);
  if (!launch_descriptors_.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(d_descriptors_, launch_descriptors_.data(),
                               launch_descriptors_.size() * sizeof(NetworkDescriptor), cudaMemcpyHostToDevice, stream));
  }

  pending_entry_uploads_.clear();
  std::fill(entry_upload_pending_.begin(), entry_upload_pending_.end(), 0);
  dirty_ = false;
  launch_dirty_ = false;
  full_upload_required_ = false;
}

void InferenceCache::add_entity(uint32_t slot, const CompiledNetwork &compiled) {
  if (dirty_) {
    return;
  }

  if (slot != slot_to_entry_.size()) {
    dirty_ = true;
    return;
  }

  const uint32_t entry_index = acquire_entry(compiled);
  slot_to_entry_.push_back(entry_index);
  launch_dirty_ = true;
}

void InferenceCache::swap_remove_entity(uint32_t removed_slot, uint32_t last_slot) {
  if (dirty_) {
    return;
  }

  if (slot_to_entry_.empty() || last_slot >= slot_to_entry_.size() || removed_slot >= slot_to_entry_.size()) {
    dirty_ = true;
    return;
  }

  free_entries_.push_back(slot_to_entry_[removed_slot]);
  if (removed_slot != last_slot) {
    slot_to_entry_[removed_slot] = slot_to_entry_[last_slot];
  }
  slot_to_entry_.pop_back();
  launch_dirty_ = true;
}

bool InferenceCache::prepare_for_launch(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream) {
  if (count == 0) {
    return true;
  }

  if (dirty_ || slot_to_entry_.size() != count) {
    build_from(network_cache, count, stream);
    return !dirty_;
  }

  if (needs_reallocation(node_extent_, eval_extent_, conn_extent_, ptr_extent_, output_extent_, count)) {
    allocate_device_memory(grow_capacity(node_capacity_, node_extent_), grow_capacity(eval_capacity_, eval_extent_),
                           grow_capacity(conn_capacity_, conn_extent_), grow_capacity(ptr_capacity_, ptr_extent_),
                           grow_capacity(output_capacity_, output_extent_), grow_capacity(entity_capacity_, count));
    full_upload_required_ = true;
  }

  if (full_upload_required_) {
    upload_full(stream);
    std::fill(entry_upload_pending_.begin(), entry_upload_pending_.end(), 0);
    pending_entry_uploads_.clear();
  } else {
    upload_pending(stream);
  }

  if (launch_dirty_ || full_upload_required_) {
    rebuild_launch_descriptors();
    if (!launch_descriptors_.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(d_descriptors_, launch_descriptors_.data(),
                                 launch_descriptors_.size() * sizeof(NetworkDescriptor), cudaMemcpyHostToDevice,
                                 stream));
    }
  }

  launch_dirty_ = false;
  full_upload_required_ = false;
  return true;
}

bool InferenceCache::launch_inference_async(const float *sensor_inputs, float *brain_outputs, std::size_t count,
                                            cudaStream_t stream) {
  if (count == 0 || !is_valid()) {
    return true;
  }

  if (count > slot_to_entry_.size()) {
    spdlog::error("Agent count exceeds cached network count");
    return false;
  }

  if (node_extent_ > 0) {
    CUDA_CHECK(cudaMemsetAsync(d_node_values_, 0, node_extent_ * sizeof(float), stream));
  }

  const int num_blocks = (static_cast<int>(count) + kBlockSize - 1) / kBlockSize;

  MOONAI_PROFILE_SCOPE("inference_kernel", stream);

  kernel_neural_inference<<<num_blocks, kBlockSize, 0, stream>>>(
      d_descriptors_, d_node_values_, d_eval_order_, d_conn_from_, d_conn_weights_, d_conn_ptr_, d_out_indices_,
      sensor_inputs, brain_outputs, static_cast<int>(count));

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("Neural inference kernel launch failed: {}", cudaGetErrorString(err));
    return false;
  }

  return true;
}

} // namespace moonai::evolution
