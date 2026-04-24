#include <iostream>
#include <random>
#include <array>
#include <cstdint>

int main() {
    // How does std::mt19937_64 seed with a single integer?
    // It uses seed_seq internally
    
    std::seed_seq seq{42};
    std::array<uint64_t, 312> state;
    seq.generate(reinterpret_cast<uint32_t*>(state.begin()), reinterpret_cast<uint32_t*>(state.end()));
    
    for (int i = 0; i < 10; i++) {
        std::cout << "state[" << i << "]: " << state[i] << std::endl;
    }
    
    return 0;
}
