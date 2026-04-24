#include <iostream>
#include <random>

int main() {
    std::mt19937_64 engine(42);
    std::uniform_int_distribution<int> dist(0, 1000);
    
    for (int i = 0; i < 20; i++) {
        std::cout << "int " << i << ": " << dist(engine) << "\n";
    }
    
    return 0;
}
