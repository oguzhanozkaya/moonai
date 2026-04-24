#include <iostream>
#include <random>
#include <cstdint>

int main() {
    std::mt19937_64 engine(42);
    
    for (int i = 0; i < 5; i++) {
        uint64_t val = engine();
        std::cout << "raw " << i << ": " << val << std::endl;
    }
    
    std::uniform_int_distribution<int> dist(0, 1000);
    for (int i = 0; i < 5; i++) {
        int val = dist(engine);
        std::cout << "int " << i << ": " << val << std::endl;
    }
    
    return 0;
}
