#include <iostream>
#include <random>
#include <cstdint>

int main() {
    // Try different ways of seeding
    std::mt19937_64 engine1(42);
    
    // Print first 5 raw values
    for (int i = 0; i < 5; i++) {
        std::cout << "std mt19937_64 seed42 raw " << i << ": " << engine1() << std::endl;
    }
    
    return 0;
}
