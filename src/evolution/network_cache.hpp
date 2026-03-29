#pragma once
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "simulation/entity.hpp"
#include <memory>
#include <vector>

namespace moonai {

class NetworkCache {
public:
  void assign(Entity e, const Genome &genome);

  NeuralNetwork *get(Entity e) const;

  NeuralNetwork *get_network(Entity e) const {
    return get(e);
  }

  void remove(Entity e);
  void move_entity(Entity from, Entity to);

  bool has(Entity e) const;

  std::vector<float> activate(Entity e, const std::vector<float> &inputs) const;

  void activate_batch(std::size_t entity_count,
                      const std::vector<float> &all_inputs,
                      std::vector<float> &all_outputs, int inputs_per_network,
                      int outputs_per_network);

  void clear();

  size_t size() const {
    return networks_.size();
  }
  bool empty() const {
    return networks_.empty();
  }

private:
  std::vector<std::unique_ptr<NeuralNetwork>> networks_;
};

} // namespace moonai
