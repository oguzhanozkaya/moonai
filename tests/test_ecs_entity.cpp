#include "simulation/entity.hpp"
#include <gtest/gtest.h>

using namespace moonai;

TEST(EntityTest, DefaultConstruction) {
  // Entity is an aggregate, use value initialization
  Entity e{};
  EXPECT_EQ(e.index, 0);
  EXPECT_EQ(e.generation, 0);
}

TEST(EntityTest, ValidEntity) {
  Entity e{1, 1};
  EXPECT_TRUE(e.valid());

  Entity invalid{0, 0};
  EXPECT_FALSE(invalid.valid());

  Entity invalid2{1, 0};
  EXPECT_FALSE(invalid2.valid());
}

TEST(EntityTest, Equality) {
  Entity e1{1, 1};
  Entity e2{1, 1};
  Entity e3{1, 2};
  Entity e4{2, 1};

  EXPECT_EQ(e1, e2);
  EXPECT_NE(e1, e3);
  EXPECT_NE(e1, e4);
}

TEST(EntityTest, InvalidEntity) {
  EXPECT_EQ(INVALID_ENTITY.index, 0);
  EXPECT_EQ(INVALID_ENTITY.generation, 0);
  EXPECT_FALSE(INVALID_ENTITY.valid());
}

TEST(EntityTest, HashFunction) {
  EntityHash hasher;
  Entity e1{1, 1};
  Entity e2{1, 1};
  Entity e3{2, 3};

  EXPECT_EQ(hasher(e1), hasher(e2));
  EXPECT_NE(hasher(e1), hasher(e3));
}