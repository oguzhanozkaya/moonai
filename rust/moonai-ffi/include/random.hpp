#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct CRandom {
    std::mt19937_64 engine_;

    CRandom(uint64_t seed);
    uint32_t next_int32();
    uint64_t next_int64();
    float next_float();
    bool next_bool(float probability);
    int weighted_select(const std::vector<float>& weights);
};

extern "C" {

CRandom* c_random_create(int32_t seed);
void c_random_destroy(CRandom* rng);
int32_t c_random_next_int(CRandom* rng, int32_t min, int32_t max);
float c_random_next_float(CRandom* rng, float min, float max);
bool c_random_next_bool(CRandom* rng, float probability);
int32_t c_random_weighted_select(CRandom* rng, const float* weights, size_t len);

}
