#include "simulation/registry.hpp"
#include "simulation/system.hpp"
#include <gtest/gtest.h>

using namespace moonai;

class TestSystem : public System {
public:
  int update_count = 0;
  float last_dt = 0.0f;

  void update(Registry &registry, float dt) override {
    ++update_count;
    last_dt = dt;
  }

  const char *name() const override {
    return "TestSystem";
  }
};

TEST(SystemSchedulerTest, AddAndUpdate) {
  SystemScheduler scheduler;
  auto test_sys = std::make_unique<TestSystem>();
  auto *test_ptr = test_sys.get();

  scheduler.add_system(std::move(test_sys));

  Registry registry;
  scheduler.update(registry, 0.016f);

  EXPECT_EQ(test_ptr->update_count, 1);
  EXPECT_FLOAT_EQ(test_ptr->last_dt, 0.016f);
}

TEST(SystemSchedulerTest, MultipleSystems) {
  SystemScheduler scheduler;
  auto sys1 = std::make_unique<TestSystem>();
  auto sys2 = std::make_unique<TestSystem>();

  auto *ptr1 = sys1.get();
  auto *ptr2 = sys2.get();

  scheduler.add_system(std::move(sys1));
  scheduler.add_system(std::move(sys2));

  Registry registry;
  scheduler.update(registry, 0.016f);

  EXPECT_EQ(ptr1->update_count, 1);
  EXPECT_EQ(ptr2->update_count, 1);
  EXPECT_EQ(scheduler.system_count(), 2);
}

TEST(SystemSchedulerTest, ClearSystems) {
  SystemScheduler scheduler;
  scheduler.add_system(std::make_unique<TestSystem>());
  scheduler.add_system(std::make_unique<TestSystem>());

  EXPECT_EQ(scheduler.system_count(), 2);

  scheduler.clear();

  EXPECT_EQ(scheduler.system_count(), 0);
}