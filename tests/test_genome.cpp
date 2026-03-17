#include <gtest/gtest.h>
#include "evolution/genome.hpp"

using namespace moonai;

TEST(GenomeTest, ConstructorCreatesInputAndOutputNodes) {
    Genome g(3, 2);

    // 3 inputs + 1 bias + 2 outputs = 6 nodes
    EXPECT_EQ(g.nodes().size(), 6u);
    EXPECT_EQ(g.num_inputs(), 3);
    EXPECT_EQ(g.num_outputs(), 2);
}

TEST(GenomeTest, AddConnection) {
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});

    EXPECT_EQ(g.connections().size(), 1u);
    EXPECT_FLOAT_EQ(g.connections()[0].weight, 0.5f);
}

TEST(GenomeTest, HasConnection) {
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});

    EXPECT_TRUE(g.has_connection(0, 3));
    EXPECT_FALSE(g.has_connection(1, 3));
    EXPECT_FALSE(g.has_connection(3, 0));
}

TEST(GenomeTest, HasNode) {
    Genome g(2, 1);
    EXPECT_TRUE(g.has_node(0));
    EXPECT_TRUE(g.has_node(1));
    EXPECT_TRUE(g.has_node(2));  // bias
    EXPECT_TRUE(g.has_node(3));  // output
    EXPECT_FALSE(g.has_node(4));
}

TEST(GenomeTest, MaxNodeId) {
    Genome g(2, 1);
    EXPECT_EQ(g.max_node_id(), 3u);

    g.add_node({10, NodeType::Hidden});
    EXPECT_EQ(g.max_node_id(), 10u);
}

TEST(GenomeTest, Complexity) {
    Genome g(2, 1);
    EXPECT_EQ(g.complexity(), 4);  // 4 nodes, 0 connections

    g.add_connection({0, 3, 0.5f, true, 0});
    EXPECT_EQ(g.complexity(), 5);  // 4 nodes + 1 connection
}

TEST(GenomeTest, CompatibilityDistanceSameGenome) {
    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});

    float dist = Genome::compatibility_distance(g, g, 1.0f, 1.0f, 0.4f);
    EXPECT_FLOAT_EQ(dist, 0.0f);
}

TEST(GenomeTest, CompatibilityDistanceDifferentWeights) {
    Genome a(2, 1);
    a.add_connection({0, 3, 1.0f, true, 0});

    Genome b(2, 1);
    b.add_connection({0, 3, 0.0f, true, 0});

    float dist = Genome::compatibility_distance(a, b, 0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(dist, 1.0f);  // avg weight diff = 1.0
}

TEST(GenomeTest, CompatibilityDistanceWithExcess) {
    Genome a(2, 1);
    a.add_connection({0, 3, 0.5f, true, 0});
    a.add_connection({1, 3, 0.5f, true, 1});
    a.add_connection({2, 3, 0.5f, true, 2});

    Genome b(2, 1);
    b.add_connection({0, 3, 0.5f, true, 0});

    // b has max innov 0, a has max innov 2 -> 2 excess genes
    float dist = Genome::compatibility_distance(a, b, 1.0f, 0.0f, 0.0f);
    EXPECT_GT(dist, 0.0f);
}

TEST(GenomeTest, FitnessDefaultsToZero) {
    Genome g(2, 1);
    EXPECT_FLOAT_EQ(g.fitness(), 0.0f);
}

TEST(GenomeTest, AdjustedFitness) {
    Genome g(2, 1);
    g.set_adjusted_fitness(0.5f);
    EXPECT_FLOAT_EQ(g.adjusted_fitness(), 0.5f);
}

TEST(GenomeTest, JsonRoundTrip) {
    Genome g(3, 2);
    g.add_connection({0, 4, 0.5f, true, 0});
    g.add_connection({1, 5, -0.3f, false, 1});
    g.add_node({10, NodeType::Hidden});
    g.set_fitness(42.0f);

    std::string json = g.to_json();
    Genome restored = Genome::from_json(json);

    EXPECT_EQ(restored.num_inputs(), 3);
    EXPECT_EQ(restored.num_outputs(), 2);
    EXPECT_EQ(restored.nodes().size(), g.nodes().size());
    EXPECT_EQ(restored.connections().size(), 2u);
    EXPECT_FLOAT_EQ(restored.fitness(), 42.0f);
    EXPECT_FLOAT_EQ(restored.connections()[0].weight, 0.5f);
    EXPECT_FALSE(restored.connections()[1].enabled);
}
