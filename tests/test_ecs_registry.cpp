#include "simulation/registry.hpp"
#include <gtest/gtest.h>

using namespace moonai;

TEST(ECSRegistryTest, EntityCreation) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  EXPECT_NE(e1, e2);
  EXPECT_TRUE(registry.alive(e1));
  EXPECT_TRUE(registry.alive(e2));
  EXPECT_EQ(registry.size(), 2);
}

TEST(ECSRegistryTest, EntityDestruction) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  registry.destroy(e1);

  EXPECT_FALSE(registry.alive(e1));
  EXPECT_TRUE(registry.alive(e2));
  EXPECT_EQ(registry.size(), 1);
}

TEST(ECSRegistryTest, DestroyInvalidEntity) {
  Registry registry;

  registry.destroy(INVALID_ENTITY); // Should not crash

  auto e = registry.create();
  registry.destroy(e);
  registry.destroy(e); // Double destroy - should not crash

  EXPECT_FALSE(registry.alive(e));
}

TEST(ECSRegistryTest, SlotRecycling) {
  Registry registry;

  auto e1 = registry.create();
  uint32_t original_index = e1.index;

  registry.destroy(e1);

  auto e2 = registry.create();

  // Should recycle the same index but with new generation
  EXPECT_EQ(e2.index, original_index);
  EXPECT_GT(e2.generation, e1.generation);
  EXPECT_NE(e1, e2);
}

TEST(ECSRegistryTest, ComponentArraysResize) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  // Component arrays should have been resized
  EXPECT_GT(registry.positions().size(), 0);
  EXPECT_GT(registry.vitals().size(), 0);
  EXPECT_GT(registry.identity().size(), 0);
}

TEST(ECSRegistryTest, DirectComponentAccess) {
  Registry registry;

  auto e = registry.create();

  // Write components
  registry.pos_x(e) = 100.0f;
  registry.pos_y(e) = 200.0f;
  registry.energy(e) = 50.0f;

  // Read components back
  EXPECT_FLOAT_EQ(registry.pos_x(e), 100.0f);
  EXPECT_FLOAT_EQ(registry.pos_y(e), 200.0f);
  EXPECT_FLOAT_EQ(registry.energy(e), 50.0f);
}

TEST(ECSRegistryTest, LivingEntitiesList) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();
  auto e3 = registry.create();

  registry.destroy(e2);

  const auto &living = registry.living_entities();
  EXPECT_EQ(living.size(), 2);

  // e1 and e3 should be in the list
  bool has_e1 = false, has_e3 = false;
  for (const auto &e : living) {
    if (e == e1)
      has_e1 = true;
    if (e == e3)
      has_e3 = true;
  }
  EXPECT_TRUE(has_e1);
  EXPECT_TRUE(has_e3);
}

TEST(ECSRegistryTest, Clear) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  registry.clear();

  EXPECT_EQ(registry.size(), 0);
  EXPECT_FALSE(registry.alive(e1));
  EXPECT_FALSE(registry.alive(e2));
  EXPECT_TRUE(registry.empty());
}

TEST(ECSRegistryTest, SoAComponentsAccess) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  size_t idx1 = registry.index_of(e1);
  size_t idx2 = registry.index_of(e2);

  // Set position for e1
  registry.positions().x[idx1] = 10.0f;
  registry.positions().y[idx1] = 20.0f;

  // Set position for e2
  registry.positions().x[idx2] = 30.0f;
  registry.positions().y[idx2] = 40.0f;

  // Verify positions
  EXPECT_FLOAT_EQ(registry.positions().x[idx1], 10.0f);
  EXPECT_FLOAT_EQ(registry.positions().y[idx1], 20.0f);
  EXPECT_FLOAT_EQ(registry.positions().x[idx2], 30.0f);
  EXPECT_FLOAT_EQ(registry.positions().y[idx2], 40.0f);
}

TEST(ECSRegistryTest, VitalsSoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  registry.vitals().energy[idx] = 100.0f;
  registry.vitals().age[idx] = 5;
  registry.vitals().alive[idx] = 1;
  registry.vitals().reproduction_cooldown[idx] = 10;

  EXPECT_FLOAT_EQ(registry.vitals().energy[idx], 100.0f);
  EXPECT_EQ(registry.vitals().age[idx], 5);
  EXPECT_EQ(registry.vitals().alive[idx], 1);
  EXPECT_EQ(registry.vitals().reproduction_cooldown[idx], 10);
}

TEST(ECSRegistryTest, IdentitySoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  registry.identity().type[idx] = IdentitySoA::TYPE_PREDATOR;
  registry.identity().species_id[idx] = 42;
  registry.identity().entity_id[idx] = 123;

  EXPECT_EQ(registry.identity().type[idx], IdentitySoA::TYPE_PREDATOR);
  EXPECT_EQ(registry.identity().species_id[idx], 42);
  EXPECT_EQ(registry.identity().entity_id[idx], 123);
}

TEST(ECSRegistryTest, SensorSoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  // Test input pointer access
  float *inputs = registry.sensors().input_ptr(idx);
  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    inputs[i] = static_cast<float>(i);
  }

  // Verify inputs
  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    EXPECT_FLOAT_EQ(registry.sensors().inputs[idx * SensorSoA::INPUT_COUNT + i],
                    static_cast<float>(i));
  }

  // Test output pointer access
  float *outputs = registry.sensors().output_ptr(idx);
  outputs[0] = 0.5f;
  outputs[1] = -0.5f;

  EXPECT_FLOAT_EQ(registry.sensors().outputs[idx * SensorSoA::OUTPUT_COUNT + 0],
                  0.5f);
  EXPECT_FLOAT_EQ(registry.sensors().outputs[idx * SensorSoA::OUTPUT_COUNT + 1],
                  -0.5f);
}