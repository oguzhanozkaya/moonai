#include "simulation/sparse_set.hpp"
#include <gtest/gtest.h>

using namespace moonai;

TEST(SparseSetTest, InsertAndRetrieve) {
  SparseSet set;
  Entity e1{1, 1};

  size_t idx = set.insert(e1);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(set.size(), 1);
  EXPECT_TRUE(set.contains(e1));

  Entity retrieved = set.get_entity(idx);
  EXPECT_EQ(retrieved, e1);
}

TEST(SparseSetTest, MultipleInsertions) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  size_t idx1 = set.insert(e1);
  size_t idx2 = set.insert(e2);
  size_t idx3 = set.insert(e3);

  EXPECT_EQ(set.size(), 3);
  EXPECT_EQ(idx1, 0);
  EXPECT_EQ(idx2, 1);
  EXPECT_EQ(idx3, 2);

  EXPECT_TRUE(set.contains(e1));
  EXPECT_TRUE(set.contains(e2));
  EXPECT_TRUE(set.contains(e3));
}

TEST(SparseSetTest, DuplicateInsert) {
  SparseSet set;
  Entity e1{1, 1};

  size_t idx1 = set.insert(e1);
  size_t idx2 = set.insert(e1);

  EXPECT_EQ(idx1, idx2);
  EXPECT_EQ(set.size(), 1);
}

TEST(SparseSetTest, GetIndex) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);
  set.insert(e2);

  EXPECT_EQ(set.get_index(e1), 0);
  EXPECT_EQ(set.get_index(e2), 1);
}

TEST(SparseSetTest, GetIndexNotFound) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);

  size_t invalid = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(set.get_index(e2), invalid);
}

TEST(SparseSetTest, RemoveSingleEntity) {
  SparseSet set;
  Entity e1{1, 1};

  set.insert(e1);
  set.remove(e1);

  EXPECT_EQ(set.size(), 0);
  EXPECT_FALSE(set.contains(e1));
}

TEST(SparseSetTest, RemovePreservesOthers) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  set.remove(e2);

  EXPECT_EQ(set.size(), 2);
  EXPECT_TRUE(set.contains(e1));
  EXPECT_FALSE(set.contains(e2));
  EXPECT_TRUE(set.contains(e3));
}

TEST(SparseSetTest, RemoveMiddleEntity) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  set.remove(e2);

  // e3 should have swapped to e2's position
  EXPECT_EQ(set.get_index(e3), 1);
  EXPECT_EQ(set.get_entity(1), e3);
}

TEST(SparseSetTest, RemoveNonexistentEntity) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);
  set.remove(e2); // Should not crash

  EXPECT_EQ(set.size(), 1);
  EXPECT_TRUE(set.contains(e1));
}

TEST(SparseSetTest, DenseIteration) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  const auto &dense = set.dense();
  EXPECT_EQ(dense.size(), 3);

  // Check that all entities are in the dense array
  bool has_e1 = false, has_e2 = false, has_e3 = false;
  for (const auto &e : dense) {
    if (e == e1)
      has_e1 = true;
    if (e == e2)
      has_e2 = true;
    if (e == e3)
      has_e3 = true;
  }

  EXPECT_TRUE(has_e1);
  EXPECT_TRUE(has_e2);
  EXPECT_TRUE(has_e3);
}

TEST(SparseSetTest, GenerationMismatch) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e1_new_gen{1, 2};

  set.insert(e1);

  EXPECT_TRUE(set.contains(e1));
  EXPECT_FALSE(set.contains(e1_new_gen));
}