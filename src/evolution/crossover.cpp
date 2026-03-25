#include "evolution/crossover.hpp"

#include <map>
#include <unordered_set>

namespace moonai {

Genome Crossover::crossover(const Genome &parent_a, const Genome &parent_b,
                            Random &rng) {
  // parent_a should be the fitter parent
  const Genome &fitter =
      (parent_a.fitness() >= parent_b.fitness()) ? parent_a : parent_b;
  const Genome &other = (&fitter == &parent_a) ? parent_b : parent_a;

  Genome child(fitter.num_inputs(), fitter.num_outputs());

  // Build innovation maps
  std::map<std::uint32_t, const ConnectionGene *> map_fitter, map_other;
  for (const auto &c : fitter.connections())
    map_fitter[c.innovation] = &c;
  for (const auto &c : other.connections())
    map_other[c.innovation] = &c;

  // Collect all hidden node ids we'll need
  std::unordered_set<std::uint32_t> needed_nodes;

  // Matching genes: pick randomly, disjoint/excess: take from fitter
  for (const auto &[innov, conn] : map_fitter) {
    ConnectionGene gene;

    if (map_other.count(innov)) {
      // Matching gene - randomly pick from either parent
      if (rng.next_bool(0.5f)) {
        gene = *conn;
      } else {
        gene = *map_other.at(innov);
      }

      // If disabled in either parent, 75% chance of being disabled in child
      if (!conn->enabled || !map_other.at(innov)->enabled) {
        gene.enabled = !rng.next_bool(0.75f);
      }
    } else {
      // Disjoint or excess from fitter parent
      gene = *conn;
    }

    child.add_connection(gene);
    needed_nodes.insert(gene.in_node);
    needed_nodes.insert(gene.out_node);
  }

  // Add hidden nodes that are referenced by connections
  for (const auto &node : fitter.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id)) {
      child.add_node(node);
    }
  }
  // Also check other parent for hidden nodes (for matching gene connections)
  for (const auto &node : other.nodes()) {
    if (node.type == NodeType::Hidden && needed_nodes.count(node.id) &&
        !child.has_node(node.id)) {
      child.add_node(node);
    }
  }

  return child;
}

} // namespace moonai
