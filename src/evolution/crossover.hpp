#pragma once

#include "evolution/genome.hpp"
#include "core/random.hpp"

namespace moonai {

class Crossover {
public:
    static Genome crossover(const Genome& parent_a, const Genome& parent_b,
                            Random& rng);
};

} // namespace moonai
