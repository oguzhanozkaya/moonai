#include "simulation/environment.hpp"

#include <algorithm>
#include <cmath>

namespace moonai {

Environment::Environment(const SimulationConfig& config)
    : width_(config.grid_width)
    , height_(config.grid_height)
    , boundary_mode_(config.boundary_mode)
    , max_food_(config.food_count) {
}

Vec2 Environment::apply_boundary(Vec2 pos) const {
    if (boundary_mode_ == BoundaryMode::Wrap) {
        float w = static_cast<float>(width_);
        float h = static_cast<float>(height_);
        auto wrap = [](float v, float size) -> float {
            v = std::fmod(v, size);
            if (v < 0.0f) v += size;
            return v;
        };
        pos.x = wrap(pos.x, w);
        pos.y = wrap(pos.y, h);
    } else {
        pos.x = std::clamp(pos.x, 0.0f, static_cast<float>(width_));
        pos.y = std::clamp(pos.y, 0.0f, static_cast<float>(height_));
    }
    return pos;
}

void Environment::initialize_food(Random& rng, int count) {
    food_.clear();
    food_.reserve(count);
    for (int i = 0; i < count; ++i) {
        Food f;
        f.position = {rng.next_float(0, static_cast<float>(width_)),
                      rng.next_float(0, static_cast<float>(height_))};
        f.active = true;
        food_.push_back(f);
    }
}

void Environment::tick_food(Random& rng, float respawn_rate) {
    for (auto& f : food_) {
        if (!f.active && rng.next_bool(respawn_rate)) {
            f.position = {rng.next_float(0, static_cast<float>(width_)),
                          rng.next_float(0, static_cast<float>(height_))};
            f.active = true;
        }
    }
}

bool Environment::try_eat_food(Vec2 position, float range) {
    for (auto& f : food_) {
        if (f.active && position.distance_to(f.position) <= range) {
            f.active = false;
            return true;
        }
    }
    return false;
}

} // namespace moonai
