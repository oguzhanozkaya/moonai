#include <gtest/gtest.h>
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "gpu/gpu_batch.hpp"

using namespace moonai;
using namespace moonai::gpu;

// ── Helper: build GpuNetworkData from a vector of NeuralNetworks ────────────
// Mirrors the static build_gpu_network_data() in evolution_manager.cpp.
static GpuNetworkData build_test_network_data(
    const std::vector<std::unique_ptr<NeuralNetwork>>& nets,
    const std::string& activation_fn)
{
    GpuNetworkData data;

    if (activation_fn == "tanh")       data.activation_fn_id = 1;
    else if (activation_fn == "relu")  data.activation_fn_id = 2;
    else                               data.activation_fn_id = 0;

    int n = static_cast<int>(nets.size());
    data.descs.resize(n);

    int total_nodes = 0, total_eval = 0, total_conn = 0, total_out = 0;

    for (int i = 0; i < n; ++i) {
        const auto& net        = nets[i];
        const auto& nodes      = net->raw_nodes();
        const auto& eval_order = net->eval_order();
        const auto& incoming   = net->incoming();
        const auto& nidx       = net->node_index_map();

        int num_conn = 0;
        for (auto nid : eval_order) {
            num_conn += static_cast<int>(incoming[nidx.at(nid)].size());
        }
        int num_out = 0;
        for (const auto& node : nodes) {
            if (node.type == NodeType::Output) ++num_out;
        }

        data.descs[i].num_nodes   = static_cast<int>(nodes.size());
        data.descs[i].num_eval    = static_cast<int>(eval_order.size());
        data.descs[i].num_inputs  = net->num_inputs();
        data.descs[i].num_outputs = net->num_outputs();
        data.descs[i].node_off    = total_nodes;
        data.descs[i].eval_off    = total_eval;
        data.descs[i].conn_off    = total_conn;
        data.descs[i].out_off     = total_out;

        total_nodes += static_cast<int>(nodes.size());
        total_eval  += static_cast<int>(eval_order.size());
        total_conn  += num_conn;
        total_out   += num_out;
    }

    data.node_types.resize(total_nodes);
    data.eval_order.resize(total_eval);
    data.conn_ptr.resize(total_eval);
    data.in_count.resize(total_eval);
    data.conn_from.resize(total_conn);
    data.conn_w.resize(total_conn);
    data.out_indices.resize(total_out);

    for (int i = 0; i < n; ++i) {
        const auto& net        = nets[i];
        const auto& nodes      = net->raw_nodes();
        const auto& eval_order = net->eval_order();
        const auto& incoming   = net->incoming();
        const auto& nidx       = net->node_index_map();
        const auto& desc       = data.descs[i];

        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            uint8_t t;
            switch (nodes[j].type) {
                case NodeType::Input:  t = 0; break;
                case NodeType::Bias:   t = 1; break;
                case NodeType::Hidden: t = 2; break;
                case NodeType::Output: t = 3; break;
                default:               t = 2; break;
            }
            data.node_types[desc.node_off + j] = t;
        }

        int conn_running = 0;
        for (int j = 0; j < static_cast<int>(eval_order.size()); ++j) {
            std::uint32_t nid = eval_order[j];
            int ni = nidx.at(nid);

            data.eval_order[desc.eval_off + j] = ni;
            data.conn_ptr  [desc.eval_off + j] = conn_running;
            data.in_count  [desc.eval_off + j] = static_cast<int>(incoming[ni].size());

            for (const auto& [from_idx, w] : incoming[ni]) {
                data.conn_from[desc.conn_off + conn_running] = from_idx;
                data.conn_w   [desc.conn_off + conn_running] = w;
                ++conn_running;
            }
        }

        int out_running = 0;
        for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
            if (nodes[j].type == NodeType::Output) {
                data.out_indices[desc.out_off + out_running] = j;
                ++out_running;
            }
        }
    }

    return data;
}

// ── Helper: build a fully-connected genome with known weights ───────────────
static Genome make_genome(int inputs, int outputs, float weight) {
    Genome g(inputs, outputs);
    std::uint32_t innov = 0;
    for (const auto& in_node : g.nodes()) {
        if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias)
            continue;
        for (const auto& out_node : g.nodes()) {
            if (out_node.type != NodeType::Output) continue;
            g.add_connection({in_node.id, out_node.id, weight, true, innov++});
        }
    }
    return g;
}

class GpuTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        gpu_available_ = init_cuda();
        if (gpu_available_) {
            print_device_info();
        }
    }
    void SetUp() override {
        if (!gpu_available_) GTEST_SKIP() << "No CUDA device available";
    }
    static bool gpu_available_;
};
bool GpuTest::gpu_available_ = false;

// Test 1: CPU vs GPU output comparison with known topology
TEST_F(GpuTest, CpuGpuOutputsMatch) {
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;
    constexpr float kWeight = 0.5f;

    Genome g = make_genome(kInputs, kOutputs, kWeight);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    // CPU forward pass
    float cpu_in[kInputs] = {1.0f, 0.5f, -1.0f};
    float cpu_out[kOutputs] = {};
    nets[0]->activate_into(cpu_in, kInputs, cpu_out, kOutputs);

    // GPU forward pass
    GpuBatch batch(1, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);

    float gpu_out[kOutputs] = {};
    batch.pack_inputs_async(cpu_in, kInputs);
    batch.launch_inference_async();
    batch.start_unpack_async();
    batch.finish_unpack(gpu_out, kOutputs);

    for (int i = 0; i < kOutputs; ++i) {
        EXPECT_NEAR(cpu_out[i], gpu_out[i], 1e-5f)
            << "Output " << i << " mismatch: CPU=" << cpu_out[i]
            << " GPU=" << gpu_out[i];
    }
}

// Test 2: Batch consistency — identical networks with identical inputs
TEST_F(GpuTest, BatchConsistency) {
    constexpr int kBatch = 100;
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;

    Genome g = make_genome(kInputs, kOutputs, 0.7f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    for (int i = 0; i < kBatch; ++i) {
        nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
    }

    GpuBatch batch(kBatch, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);

    // All agents get same inputs
    std::vector<float> flat_in(kBatch * kInputs);
    for (int i = 0; i < kBatch; ++i) {
        flat_in[i * kInputs + 0] = 1.0f;
        flat_in[i * kInputs + 1] = 0.5f;
        flat_in[i * kInputs + 2] = -0.3f;
    }

    std::vector<float> flat_out(kBatch * kOutputs);
    batch.pack_inputs_async(flat_in.data(), kBatch * kInputs);
    batch.launch_inference_async();
    batch.start_unpack_async();
    batch.finish_unpack(flat_out.data(), kBatch * kOutputs);

    // All outputs should be identical
    for (int i = 1; i < kBatch; ++i) {
        for (int j = 0; j < kOutputs; ++j) {
            EXPECT_FLOAT_EQ(flat_out[j], flat_out[i * kOutputs + j])
                << "Agent " << i << " output " << j << " differs from agent 0";
        }
    }
}

// Test 3: Varied num_outputs — verify correct stride
TEST_F(GpuTest, VariedOutputCount) {
    for (int n_out : {1, 2, 3, 4}) {
        constexpr int kInputs = 2;
        constexpr int kBatch = 10;

        Genome g = make_genome(kInputs, n_out, 0.3f);
        std::vector<std::unique_ptr<NeuralNetwork>> nets;
        for (int i = 0; i < kBatch; ++i) {
            nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
        }

        GpuBatch batch(kBatch, kInputs, n_out);
        auto data = build_test_network_data(nets, "sigmoid");
        batch.upload_network_data(data);

        std::vector<float> flat_in(kBatch * kInputs, 0.5f);
        std::vector<float> flat_out(kBatch * n_out);

        batch.pack_inputs_async(flat_in.data(), kBatch * kInputs);
        batch.launch_inference_async();
        batch.start_unpack_async();
        batch.finish_unpack(flat_out.data(), kBatch * n_out);

        // CPU reference for agent 0
        float cpu_out[4] = {};
        float cpu_in[2] = {0.5f, 0.5f};
        nets[0]->activate_into(cpu_in, kInputs, cpu_out, n_out);

        for (int j = 0; j < n_out; ++j) {
            EXPECT_NEAR(cpu_out[j], flat_out[j], 1e-5f)
                << "n_out=" << n_out << " output " << j << " mismatch";
        }
    }
}

// Test 4: Zero-connection network (all connections disabled)
TEST_F(GpuTest, ZeroConnectionNetwork) {
    constexpr int kInputs = 3;
    constexpr int kOutputs = 2;

    // Create genome with no connections
    Genome g(kInputs, kOutputs);

    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));

    GpuBatch batch(1, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);

    float in_data[kInputs] = {1.0f, 0.5f, -1.0f};
    float out_data[kOutputs] = {};

    batch.pack_inputs_async(in_data, kInputs);
    batch.launch_inference_async();
    batch.start_unpack_async();
    batch.finish_unpack(out_data, kOutputs);

    // With no connections, outputs should be 0 (initialized) or sigmoid(0)
    // depending on whether output nodes are in eval_order with 0 incoming
    for (int i = 0; i < kOutputs; ++i) {
        EXPECT_TRUE(std::isfinite(out_data[i]))
            << "Output " << i << " is not finite: " << out_data[i];
    }
}

// Test 5: Large batch — verify no CUDA errors and finite outputs
TEST_F(GpuTest, LargeBatch) {
    constexpr int kBatch = 5000;
    constexpr int kInputs = 15;
    constexpr int kOutputs = 2;

    Genome g = make_genome(kInputs, kOutputs, 0.1f);
    std::vector<std::unique_ptr<NeuralNetwork>> nets;
    for (int i = 0; i < kBatch; ++i) {
        nets.push_back(std::make_unique<NeuralNetwork>(g, "sigmoid"));
    }

    GpuBatch batch(kBatch, kInputs, kOutputs);
    auto data = build_test_network_data(nets, "sigmoid");
    batch.upload_network_data(data);

    std::vector<float> flat_in(kBatch * kInputs);
    for (int i = 0; i < kBatch * kInputs; ++i) {
        flat_in[i] = static_cast<float>(i % 7) / 7.0f;  // deterministic pattern
    }

    std::vector<float> flat_out(kBatch * kOutputs);
    batch.pack_inputs_async(flat_in.data(), kBatch * kInputs);
    batch.launch_inference_async();
    batch.start_unpack_async();
    batch.finish_unpack(flat_out.data(), kBatch * kOutputs);

    for (int i = 0; i < kBatch * kOutputs; ++i) {
        EXPECT_TRUE(std::isfinite(flat_out[i]))
            << "Output at index " << i << " is not finite: " << flat_out[i];
    }
}

// Test 6: Activation function variants — tanh and relu produce correct results
TEST_F(GpuTest, ActivationFunctions) {
    for (const auto& fn : {"sigmoid", "tanh", "relu"}) {
        constexpr int kInputs = 3;
        constexpr int kOutputs = 2;

        Genome g = make_genome(kInputs, kOutputs, 0.5f);
        std::vector<std::unique_ptr<NeuralNetwork>> nets;
        nets.push_back(std::make_unique<NeuralNetwork>(g, fn));

        GpuBatch batch(1, kInputs, kOutputs);
        auto data = build_test_network_data(nets, fn);
        batch.upload_network_data(data);

        float in_data[kInputs] = {1.0f, -0.5f, 0.3f};
        float cpu_out[kOutputs] = {};
        nets[0]->activate_into(in_data, kInputs, cpu_out, kOutputs);

        float gpu_out[kOutputs] = {};
        batch.pack_inputs_async(in_data, kInputs);
        batch.launch_inference_async();
        batch.start_unpack_async();
        batch.finish_unpack(gpu_out, kOutputs);

        for (int i = 0; i < kOutputs; ++i) {
            EXPECT_NEAR(cpu_out[i], gpu_out[i], 1e-5f)
                << "Activation=" << fn << " output " << i
                << " mismatch: CPU=" << cpu_out[i] << " GPU=" << gpu_out[i];
        }
    }
}
