#include "visualization/visualization_manager.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/simulation_manager.hpp"

#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <SFML/Window/WindowEnums.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Image.hpp>

#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

namespace moonai {

VisualizationManager::VisualizationManager(const SimulationConfig& config)
    : config_(config) {
}

VisualizationManager::~VisualizationManager() = default;

bool VisualizationManager::initialize() {
    window_ = std::make_unique<sf::RenderWindow>(
        sf::VideoMode({window_width_, window_height_}),
        "MoonAI - Predator-Prey Evolution"
    );
    window_->setFramerateLimit(static_cast<unsigned int>(config_.target_fps));

    // Set up camera to show the simulation world
    camera_view_ = sf::View(
        sf::Vector2f(static_cast<float>(config_.grid_width) / 2.0f,
                     static_cast<float>(config_.grid_height) / 2.0f),
        sf::Vector2f(static_cast<float>(config_.grid_width),
                     static_cast<float>(config_.grid_height))
    );

    // Adjust view to maintain aspect ratio
    float world_aspect = static_cast<float>(config_.grid_width) / config_.grid_height;
    float window_aspect = static_cast<float>(window_width_) / window_height_;

    if (window_aspect > world_aspect) {
        camera_view_.setSize(sf::Vector2f(
            static_cast<float>(config_.grid_height) * window_aspect,
            static_cast<float>(config_.grid_height)));
    } else {
        camera_view_.setSize(sf::Vector2f(
            static_cast<float>(config_.grid_width),
            static_cast<float>(config_.grid_width) / window_aspect));
    }

    window_->setView(camera_view_);

    overlay_.initialize();

    running_ = true;
    spdlog::info("Visualization initialized ({}x{} window, {}x{} world)",
                 window_width_, window_height_, config_.grid_width, config_.grid_height);
    return true;
}

void VisualizationManager::render(const SimulationManager& sim, const EvolutionManager& evolution) {
    if (!window_ || !running_) return;

    // Apply any pending click now that we have sim access
    if (pending_click_) {
        handle_mouse_click(pending_click_x_, pending_click_y_, sim);
        pending_click_ = false;
    }

    window_->clear(sf::Color::Black);
    window_->setView(camera_view_);

    // Draw world
    renderer_.draw_background(*window_, config_.grid_width, config_.grid_height);

    if (renderer_.show_grid) {
        renderer_.draw_grid(*window_, config_.grid_width, config_.grid_height, 50.0f);
    }

    renderer_.draw_boundaries(*window_, config_.grid_width, config_.grid_height);

    // Draw food
    renderer_.draw_food(*window_, sim.environment().food());

    // Draw agents
    const auto& agents = sim.agents();
    for (size_t i = 0; i < agents.size(); ++i) {
        if (!agents[i]->alive()) continue;

        bool is_selected = (static_cast<int>(i) == selected_agent_);
        renderer_.draw_agent(*window_, *agents[i], is_selected);

        // Draw debug visualization for selected agent
        if (is_selected) {
            if (renderer_.show_vision) {
                renderer_.draw_vision_range(*window_, *agents[i]);
            }
            if (renderer_.show_sensors) {
                renderer_.draw_sensor_lines(*window_, *agents[i],
                                             agents, sim.environment().food());
            }
        }
    }

    // Draw dead agents (faded)
    for (size_t i = 0; i < agents.size(); ++i) {
        if (agents[i]->alive()) continue;
        renderer_.draw_agent(*window_, *agents[i], false);
    }

    // Update FPS counter
    ++frame_count_;
    float fps_elapsed = fps_clock_.getElapsedTime().asSeconds();
    if (fps_elapsed >= 0.5f) {
        overlay_stats_.fps = static_cast<float>(frame_count_) / fps_elapsed;
        frame_count_ = 0;
        fps_clock_.restart();
    }

    // Update overlay stats from simulation
    overlay_stats_.tick = sim.current_tick();
    overlay_stats_.alive_predators = sim.alive_predators();
    overlay_stats_.alive_prey = sim.alive_prey();
    overlay_stats_.speed_multiplier = speed_multiplier_;
    overlay_stats_.paused = paused_;
    overlay_stats_.fast_forward = fast_forward_;
    overlay_stats_.selected_agent = selected_agent_;

    // Update selected agent info
    if (selected_agent_ >= 0 && selected_agent_ < static_cast<int>(agents.size())) {
        const auto& sel = agents[selected_agent_];
        overlay_stats_.selected_energy = sel->energy();
        overlay_stats_.selected_age = sel->age();
        overlay_stats_.selected_kills = sel->kills();
        overlay_stats_.selected_food_eaten = sel->food_eaten();
    }

    // Draw UI overlay (with selected genome for NN topology panel)
    const Genome* sel_genome = evolution.genome_at(selected_agent_);
    overlay_.set_activations(selected_node_activations_);
    overlay_.draw(*window_, overlay_stats_, sel_genome);

    window_->display();
}

bool VisualizationManager::should_close() const {
    return !running_;
}

void VisualizationManager::handle_events() {
    if (!window_) return;

    while (const auto event = window_->pollEvent()) {
        // Window close
        if (event->is<sf::Event::Closed>()) {
            running_ = false;
            return;
        }

        // Key pressed
        if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
            switch (key->code) {
                case sf::Keyboard::Key::Escape:
                    running_ = false;
                    break;

                case sf::Keyboard::Key::Space:
                    paused_ = !paused_;
                    break;

                case sf::Keyboard::Key::Period:  // > key (step forward)
                    if (paused_) step_requested_ = true;
                    break;

                case sf::Keyboard::Key::Equal:   // + key
                case sf::Keyboard::Key::Up:
                    speed_multiplier_ = std::min(speed_multiplier_ * 2, 64);
                    break;

                case sf::Keyboard::Key::Hyphen:  // - key
                case sf::Keyboard::Key::Down:
                    speed_multiplier_ = std::max(speed_multiplier_ / 2, 1);
                    break;

                case sf::Keyboard::Key::R:
                    reset_requested_ = true;
                    break;

                case sf::Keyboard::Key::G:
                    renderer_.show_grid = !renderer_.show_grid;
                    break;

                case sf::Keyboard::Key::V:
                    renderer_.show_vision = !renderer_.show_vision;
                    renderer_.show_sensors = renderer_.show_vision;
                    break;

                case sf::Keyboard::Key::H:
                    fast_forward_ = !fast_forward_;
                    break;

                case sf::Keyboard::Key::S:
                    if (window_) {
                        sf::Texture texture(window_->getSize());
                        texture.update(*window_);
                        sf::Image img = texture.copyToImage();
                        std::string fname = "screenshot_gen"
                            + std::to_string(overlay_stats_.generation)
                            + "_tick" + std::to_string(overlay_stats_.tick) + ".png";
                        (void)img.saveToFile(fname);
                        spdlog::info("Screenshot saved: {}", fname);
                    }
                    break;

                case sf::Keyboard::Key::Home:
                    // Reset camera to default
                    zoom_level_ = 1.0f;
                    camera_view_.setCenter(
                        sf::Vector2f(static_cast<float>(config_.grid_width) / 2.0f,
                                     static_cast<float>(config_.grid_height) / 2.0f)
                    );
                    update_camera();
                    break;

                default:
                    break;
            }
        }

        // Mouse wheel for zoom
        if (const auto* scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
            float delta = scroll->delta;
            float factor = (delta > 0) ? 0.9f : 1.1f;
            zoom_level_ *= factor;
            zoom_level_ = std::clamp(zoom_level_, 0.1f, 10.0f);
            update_camera();
        }

        // Mouse button pressed (drag start / click select)
        if (const auto* btn = event->getIf<sf::Event::MouseButtonPressed>()) {
            if (btn->button == sf::Mouse::Button::Middle ||
                btn->button == sf::Mouse::Button::Right) {
                dragging_ = true;
                drag_start_ = sf::Vector2f(static_cast<float>(btn->position.x),
                                           static_cast<float>(btn->position.y));
                view_start_ = camera_view_.getCenter();
            }
        }

        // Mouse button released
        if (const auto* btn = event->getIf<sf::Event::MouseButtonReleased>()) {
            if (btn->button == sf::Mouse::Button::Middle ||
                btn->button == sf::Mouse::Button::Right) {
                dragging_ = false;
            }
            // Left click = select agent; store coords, apply in render() with sim access
            if (btn->button == sf::Mouse::Button::Left && window_) {
                auto world_pos = window_->mapPixelToCoords(
                    btn->position, camera_view_);
                pending_click_ = true;
                pending_click_x_ = world_pos.x;
                pending_click_y_ = world_pos.y;
            }
        }

        // Mouse moved (for dragging)
        if (const auto* moved = event->getIf<sf::Event::MouseMoved>()) {
            if (dragging_ && window_) {
                sf::Vector2f current(static_cast<float>(moved->position.x),
                                     static_cast<float>(moved->position.y));
                sf::Vector2f delta = drag_start_ - current;
                // Scale by zoom level
                delta.x *= zoom_level_;
                delta.y *= zoom_level_;
                camera_view_.setCenter(view_start_ + delta);
            }
        }

        // Window resized
        if (const auto* resized = event->getIf<sf::Event::Resized>()) {
            window_width_ = resized->size.x;
            window_height_ = resized->size.y;
            update_camera();
        }
    }
}

void VisualizationManager::handle_mouse_click(float world_x, float world_y,
                                                const SimulationManager& sim) {
    // Find closest agent to click position
    float best_dist = 20.0f * zoom_level_;  // click threshold in world units
    int best_idx = -1;

    const auto& agents = sim.agents();
    for (size_t i = 0; i < agents.size(); ++i) {
        if (!agents[i]->alive()) continue;
        float dx = agents[i]->position().x - world_x;
        float dy = agents[i]->position().y - world_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<int>(i);
        }
    }

    selected_agent_ = best_idx;
}

void VisualizationManager::set_selected_activations(
    const std::vector<float>& vals,
    const std::unordered_map<std::uint32_t, int>& idx_map)
{
    selected_node_activations_.clear();
    // idx_map: node_id -> array index; invert to fill {node_id -> activation_value}
    for (const auto& [node_id, idx] : idx_map) {
        if (idx >= 0 && idx < static_cast<int>(vals.size())) {
            selected_node_activations_[node_id] = vals[idx];
        }
    }
}

void VisualizationManager::update_camera() {
    if (!window_) return;

    float world_w = static_cast<float>(config_.grid_width);
    float world_h = static_cast<float>(config_.grid_height);

    float window_aspect = static_cast<float>(window_width_) / window_height_;
    float world_aspect = world_w / world_h;

    float view_w, view_h;
    if (window_aspect > world_aspect) {
        view_h = world_h * zoom_level_;
        view_w = view_h * window_aspect;
    } else {
        view_w = world_w * zoom_level_;
        view_h = view_w / window_aspect;
    }

    camera_view_.setSize(sf::Vector2f(view_w, view_h));
}

} // namespace moonai
