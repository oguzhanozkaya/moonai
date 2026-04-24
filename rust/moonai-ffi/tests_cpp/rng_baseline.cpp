#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

class Random {
public:
  explicit Random(int seed) : seed_(seed), engine_(seed) {}

  int next_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(engine_);
  }

  float next_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(engine_);
  }

  bool next_bool(float probability) {
    return next_float(0.0f, 1.0f) < probability;
  }

  int weighted_select(const std::vector<float> &weights) {
    if (weights.empty())
      return -1;

    float total = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (total <= 0.0f)
      return next_int(0, static_cast<int>(weights.size()) - 1);

    float r = next_float(0.0f, total);
    float cumulative = 0.0f;
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
      cumulative += weights[i];
      if (r <= cumulative)
        return i;
    }
    return static_cast<int>(weights.size()) - 1;
  }

private:
  int seed_;
  std::mt19937_64 engine_;
};

int main() {
  Random rng(42);

  // Output first 20 int values
  for (int i = 0; i < 20; ++i) {
    std::cout << "int " << i << ": " << rng.next_int(0, 1000) << "\n";
  }

  // Output first 10 float values
  for (int i = 0; i < 10; ++i) {
    std::cout << "float " << i << ": " << rng.next_float(0.0f, 1.0f) << "\n";
  }

  // Output first 10 bool values
  for (int i = 0; i < 10; ++i) {
    std::cout << "bool " << i << ": " << (rng.next_bool(0.5f) ? "true" : "false") << "\n";
  }

  // Output first 10 weighted select values
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  for (int i = 0; i < 10; ++i) {
    std::cout << "weighted " << i << ": " << rng.weighted_select(weights) << "\n";
  }

  return 0;
}
