#include "evolution/network_cache.hpp"
namespace moonai {

void NetworkCache::assign(Entity e, const Genome &genome) {
  if (e.index >= networks_.size()) {
    networks_.resize(static_cast<std::size_t>(e.index) + 1);
  }
  networks_[e.index] = std::make_unique<NeuralNetwork>(genome);
}

NeuralNetwork *NetworkCache::get(Entity e) const {
  if (e == INVALID_ENTITY || e.index >= networks_.size()) {
    return nullptr;
  }
  return networks_[e.index].get();
}

void NetworkCache::remove(Entity e) {
  if (e == INVALID_ENTITY || e.index >= networks_.size()) {
    return;
  }
  if (e.index + 1 == networks_.size()) {
    networks_.pop_back();
    return;
  }
  networks_[e.index].reset();
}

void NetworkCache::move_entity(Entity from, Entity to) {
  if (from == to) {
    return;
  }
  if (from == INVALID_ENTITY || from.index >= networks_.size() ||
      !networks_[from.index]) {
    return;
  }
  if (to.index >= networks_.size()) {
    networks_.resize(static_cast<std::size_t>(to.index) + 1);
  }
  networks_[to.index] = std::move(networks_[from.index]);
}

bool NetworkCache::has(Entity e) const {
  return get(e) != nullptr;
}

std::vector<float>
NetworkCache::activate(Entity e, const std::vector<float> &inputs) const {
  auto *network = get(e);
  if (network) {
    return network->activate(inputs);
  }
  return {};
}

void NetworkCache::activate_batch(std::size_t entity_count,
                                  const std::vector<float> &all_inputs,
                                  std::vector<float> &all_outputs,
                                  int inputs_per_network,
                                  int outputs_per_network) {
  all_outputs.resize(entity_count * outputs_per_network);

  for (size_t i = 0; i < entity_count; ++i) {
    Entity e{static_cast<uint32_t>(i)};
    const float *input_ptr = &all_inputs[i * inputs_per_network];
    float *output_ptr = &all_outputs[i * outputs_per_network];

    auto *network = get(e);
    if (network) {
      network->activate_into(input_ptr, inputs_per_network, output_ptr,
                             outputs_per_network);
    } else {
      // No network found - output zeros
      std::fill(output_ptr, output_ptr + outputs_per_network, 0.0f);
    }
  }
}

void NetworkCache::clear() {
  networks_.clear();
}

} // namespace moonai
