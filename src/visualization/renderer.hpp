#pragma once

#include "core/types.hpp"
#include "simulation/agent.hpp"
#include "simulation/environment.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Color.hpp>

#include <cstdint>

namespace moonai {

class Renderer {
public:
    Renderer();

    void draw_background(sf::RenderTarget& target, int width, int height);
    void draw_grid(sf::RenderTarget& target, int width, int height, float cell_size);
    void draw_boundaries(sf::RenderTarget& target, int width, int height);

    void draw_food(sf::RenderTarget& target, const std::vector<Food>& food);
    void draw_agent(sf::RenderTarget& target, const Agent& agent, bool selected = false);
    void draw_vision_range(sf::RenderTarget& target, const Agent& agent);
    void draw_sensor_lines(sf::RenderTarget& target, const Agent& agent,
                           const std::vector<std::unique_ptr<Agent>>& agents,
                           const std::vector<Food>& food);

    // Color helpers
    static sf::Color species_color(int species_id);

    bool show_grid = false;
    bool show_vision = false;
    bool show_sensors = false;
    float dead_fade_alpha = 60.0f;

private:
    sf::CircleShape circle_;
    sf::ConvexShape triangle_;
    sf::RectangleShape rect_;
};

} // namespace moonai
