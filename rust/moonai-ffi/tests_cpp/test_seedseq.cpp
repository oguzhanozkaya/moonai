#include <iostream>
#include <random>
#include <array>
#include <cstdint>

const int N = 312;

int main() {
    std::seed_seq seq{42};
    std::array<uint64_t, N> state;
    seq.generate(state.begin(), state.end());
    
    for (int i = 0; i < 5; i++) {
        std::cout << "seed_seq state[" << i << "]: " << state[i] << std::endl;
    }
    
    return 0;
}
