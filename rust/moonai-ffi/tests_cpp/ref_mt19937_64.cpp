#include <iostream>
#include <cstdint>
#include <array>

using uint64 = uint64_t;

const int N = 312;
const int M = 156;
const uint64 MATRIX_A = 0xB5026F5AA96619E9ULL;
const int UM = 29;
const uint64 LMASK = (1ULL << UM) - 1;
const uint64 UMASK_NOT = ~LMASK;
const int SHIFT1 = 17;
const int SHIFT2 = 37;
const int L = 43;

class MT19937_64 {
public:
    void init(uint64 seed) {
        state[0] = seed;
        for (int i = 1; i < N; i++) {
            state[i] = 6364136223846793005ULL * (state[i-1] ^ (state[i-1] >> 62)) + i;
        }
        idx = N;
    }
    
    uint64 next() {
        if (idx >= N) twist();
        
        uint64 x = state[idx];
        x ^= (x >> UM);
        x ^= (x << SHIFT1) & 0x71D67FFFEDA60000ULL;
        x ^= (x << SHIFT2) & 0xFFF7EEE000000000ULL;
        x ^= (x >> L);
        
        idx++;
        return x;
    }
    
private:
    std::array<uint64, N> state;
    int idx;
    
    void twist() {
        uint64 x, y;
        for (int i = 0; i < N; i++) {
            x = (state[i] & UMASK_NOT) | (state[(i+1) % N] & LMASK);
            y = state[(i + M) % N] ^ (x >> 1) ^ ((x & 1) ? MATRIX_A : 0);
            state[i] = y;
        }
        idx = 0;
    }
};

int main() {
    MT19937_64 mt;
    mt.init(42);
    
    for (int i = 0; i < 5; i++) {
        std::cout << "raw " << i << ": " << mt.next() << std::endl;
    }
    
    return 0;
}
