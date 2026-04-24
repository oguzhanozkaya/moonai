#include "random.hpp"

CRandom::CRandom(uint64_t seed) : engine_(seed) {}

uint32_t CRandom::next_int32() {
    return static_cast<uint32_t>(engine_());
}

uint64_t CRandom::next_int64() {
    return engine_();
}

float CRandom::next_float() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(engine_);
}

bool CRandom::next_bool(float probability) {
    return next_float() < probability;
}

int CRandom::weighted_select(const std::vector<float>& weights) {
    if (weights.empty()) {
        return -1;
    }

    float total = 0.0f;
    for (float w : weights) {
        total += w;
    }

    if (total <= 0.0f) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(weights.size()) - 1);
        return dist(engine_);
    }

    std::uniform_real_distribution<float> real_dist(0.0f, total);
    float r = real_dist(engine_);
    float cumulative = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (r < cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(weights.size()) - 1;
}

extern "C" {

CRandom* c_random_create(int32_t seed) {
    return new CRandom(static_cast<uint64_t>(seed));
}

void c_random_destroy(CRandom* rng) {
    delete rng;
}

int32_t c_random_next_int(CRandom* rng, int32_t min, int32_t max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng->engine_);
}

float c_random_next_float(CRandom* rng, float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng->engine_);
}

bool c_random_next_bool(CRandom* rng, float probability) {
    return rng->next_bool(probability);
}

int32_t c_random_weighted_select(CRandom* rng, const float* weights, size_t len) {
    std::vector<float> w(weights, weights + len);
    return rng->weighted_select(w);
}

}
