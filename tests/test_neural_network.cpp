#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include <gtest/gtest.h>

using namespace moonai;

TEST(NeuralNetworkTest, ActivateReturnsCorrectOutputCount) {
  Genome g(3, 2);

  std::uint32_t innov = 0;
  for (int i = 0; i < 3; ++i) {
    for (int o = 4; o <= 5; ++o) {
      g.add_connection({static_cast<std::uint32_t>(i),
                        static_cast<std::uint32_t>(o), 0.5f, true, innov++});
    }
  }

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f, 0.5f, -1.0f});

  EXPECT_EQ(outputs.size(), 2u);
}

TEST(NeuralNetworkTest, OutputsAreBounded) {
  Genome g(2, 1);
  g.add_connection({0, 3, 100.0f, true, 0});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f, 0.0f});

  // Sigmoid output should be between 0 and 1
  EXPECT_GE(outputs[0], 0.0f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, BiasNodeIsActive) {
  Genome g(1, 1);
  // Node 0: input, Node 1: bias, Node 2: output
  // Only connect bias to output
  g.add_connection({1, 2, 1.0f, true, 0});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({0.0f}); // input is zero

  // Bias=1.0, weight=1.0 -> sigmoid(4.9 * 1.0) ≈ 0.9926
  EXPECT_GT(outputs[0], 0.9f);
}

TEST(NeuralNetworkTest, DisabledConnectionsIgnored) {
  Genome g(1, 1);
  g.add_connection({0, 2, 100.0f, false, 0}); // disabled
  g.add_connection({1, 2, 0.0f, true, 1});    // bias with zero weight

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  // Only the zero-weight bias connection is active
  // sigmoid(4.9 * 0) = sigmoid(0) = 0.5
  EXPECT_NEAR(outputs[0], 0.5f, 0.01f);
}

TEST(NeuralNetworkTest, HiddenNodeTopologicalSort) {
  Genome g(1, 1);
  // Node 0: input, Node 1: bias, Node 2: output
  g.add_node({3, NodeType::Hidden});

  // input -> hidden -> output
  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({3, 2, 1.0f, true, 1});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  // input=1.0 -> hidden=sigmoid(4.9*1.0) ≈ 0.9926
  // hidden ≈ 0.9926 -> output=sigmoid(4.9*0.9926) ≈ sigmoid(4.864) ≈ 0.9923
  EXPECT_GT(outputs[0], 0.9f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, MultipleHiddenLayers) {
  Genome g(1, 1);
  g.add_node({3, NodeType::Hidden});
  g.add_node({4, NodeType::Hidden});

  // input -> h1 -> h2 -> output
  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({3, 4, 1.0f, true, 1});
  g.add_connection({4, 2, 1.0f, true, 2});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({1.0f});

  EXPECT_GE(outputs[0], 0.0f);
  EXPECT_LE(outputs[0], 1.0f);
}

TEST(NeuralNetworkTest, ZeroInputsProduceDefaultOutput) {
  Genome g(2, 1);
  g.add_connection({0, 3, 1.0f, true, 0});
  g.add_connection({1, 3, 1.0f, true, 1});

  NeuralNetwork nn(g);
  auto outputs = nn.activate({0.0f, 0.0f});

  // With zero inputs and no bias connected, sigmoid(0) = 0.5
  EXPECT_NEAR(outputs[0], 0.5f, 0.01f);
}
