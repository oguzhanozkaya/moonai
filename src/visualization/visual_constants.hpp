#pragma once

#include <cstdint>

namespace moonai {

// Entity colors (RGBA hex)
namespace colors {
constexpr uint32_t PREDATOR = 0xFF6B35FF; // Orange-Red
constexpr uint32_t PREY = 0x4ECDC4FF;     // Cyan
constexpr uint32_t FOOD = 0xFFE66DFF;     // Yellow
} // namespace colors

// Entity sizes
namespace sizes {
constexpr float PREDATOR_RADIUS = 4.0f;
constexpr float PREY_RADIUS = 3.0f;
constexpr float FOOD_RADIUS = 1.8f;
} // namespace sizes

// UI Chart colors (SFML RGB)
namespace chart_colors {
constexpr uint8_t PREDATOR_R = 0xFF;
constexpr uint8_t PREDATOR_G = 0x6B;
constexpr uint8_t PREDATOR_B = 0x35;

constexpr uint8_t PREY_R = 0x4E;
constexpr uint8_t PREY_G = 0xCD;
constexpr uint8_t PREY_B = 0xC4;

constexpr uint8_t FOOD_R = 0xFF;
constexpr uint8_t FOOD_G = 0xE6;
constexpr uint8_t FOOD_B = 0x6D;
} // namespace chart_colors

// Visualization constants
namespace visual {
// Grid and background
constexpr uint8_t GRID_R = 40;
constexpr uint8_t GRID_G = 40;
constexpr uint8_t GRID_B = 55;

constexpr uint8_t BORDER_R = 100;
constexpr uint8_t BORDER_G = 100;
constexpr uint8_t BORDER_B = 140;

constexpr uint8_t BG_R = 20;
constexpr uint8_t BG_G = 20;
constexpr uint8_t BG_B = 30;

// UI Panel
constexpr uint8_t PANEL_BG_R = 10;
constexpr uint8_t PANEL_BG_G = 10;
constexpr uint8_t PANEL_BG_B = 20;
constexpr uint8_t PANEL_ALPHA = 200;

constexpr uint8_t PANEL_OUTLINE_R = 60;
constexpr uint8_t PANEL_OUTLINE_G = 60;
constexpr uint8_t PANEL_OUTLINE_B = 80;

// Vision range
constexpr uint8_t VISION_FILL_ALPHA = 15;
constexpr uint8_t VISION_OUTLINE_ALPHA = 40;

// Sensor lines
constexpr uint8_t SENSOR_ALPHA = 80;
constexpr uint8_t FOOD_SENSOR_ALPHA = 60;
} // namespace visual

// Chart constants
namespace charts {
constexpr int CHART_MAX_POINTS = 300;
constexpr float LINE_WIDTH = 2.0f;
} // namespace charts

} // namespace moonai
