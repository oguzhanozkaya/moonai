#include "simulation/registry.hpp"
#include <chrono>
#include <gtest/gtest.h>

using namespace moonai;

TEST(ECSPerformanceTest, CreateManyEntities) {
  Registry registry;
  const int count = 10000;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < count; ++i) {
    registry.create();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_EQ(registry.size(), count);

  // Should be very fast - less than 100ms for 10k entities
  EXPECT_LT(duration.count(), 100000)
      << "Entity creation took too long: " << duration.count() << " us";
}

TEST(ECSPerformanceTest, IterateAllEntities) {
  Registry registry;
  const int count = 10000;

  // Create entities
  for (int i = 0; i < count; ++i) {
    registry.create();
  }

  // Initialize positions
  for (size_t i = 0; i < registry.positions().size(); ++i) {
    registry.positions().x[i] = static_cast<float>(i);
    registry.positions().y[i] = static_cast<float>(i * 2);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Iterate over all entities
  float total_x = 0.0f;
  float total_y = 0.0f;
  for (const auto &entity : registry.living_entities()) {
    size_t idx = registry.index_of(entity);
    total_x += registry.positions().x[idx];
    total_y += registry.positions().y[idx];
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Should be very fast due to cache-friendly SoA layout
  // Target: less than 1ms for 10k entities
  EXPECT_LT(duration.count(), 1000)
      << "Iteration took too long: " << duration.count() << " us";

  // Verify we actually processed all entities
  EXPECT_GT(total_x, 0.0f);
  EXPECT_GT(total_y, 0.0f);
}

TEST(ECSPerformanceTest, DestroyManyEntities) {
  Registry registry;
  const int count = 10000;
  std::vector<Entity> entities;
  entities.reserve(count);

  // Create entities
  for (int i = 0; i < count; ++i) {
    entities.push_back(registry.create());
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Destroy every other entity
  for (int i = 0; i < count; i += 2) {
    registry.destroy(entities[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_EQ(registry.size(), count / 2);

  // Destruction should be fast - less than 50ms for 10k
  EXPECT_LT(duration.count(), 50000)
      << "Entity destruction took too long: " << duration.count() << " us";
}

TEST(ECSPerformanceTest, RecycleSlots) {
  Registry registry;
  const int count = 10000;
  std::vector<Entity> entities;
  entities.reserve(count);

  // Create entities
  for (int i = 0; i < count; ++i) {
    entities.push_back(registry.create());
  }

  // Destroy all
  for (const auto &e : entities) {
    registry.destroy(e);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Recreate all - should use recycled slots
  for (int i = 0; i < count; ++i) {
    registry.create();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_EQ(registry.size(), count);

  // Should be fast even with slot recycling
  EXPECT_LT(duration.count(), 100000)
      << "Slot recycling took too long: " << duration.count() << " us";
}