#include "common_connectors.h"
#include "forward_declare.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <limits>

using namespace snnl;

template <typename TElem>
void compRel(TElem a, TElem b, TElem rel_prec)
{
    if (std::abs(a - b) / (std::max(a, b)) > rel_prec) {
        std::cout << double(a) << " " << double(b) << " " << std::endl;
        FAIL();
    }
}

void test_node_grad(TNode<double>&                   node,
                    std::vector<TNodeShPtr<double>>& inputs,
                    TNodeShPtr<double>&              loss)
{

    node.values().forEach([&](const TIndex& index) {
        double eps        = 1e-4;
        double val_weight = node.value(index);

        node.value(index) = val_weight + eps;
        for (auto& input : inputs) {
            input->forward();
        }

        double val_loss_up = loss->value(0);
        node.value(index)  = val_weight - eps;

        for (auto& input : inputs) {
            input->forward();
        }

        double val_loss_down = loss->value(0);

        node.value(index) = val_weight;

        double numerical_grad = (val_loss_up - val_loss_down) / (2 * eps);

        double grad = node.grad(index);

        compRel(double(numerical_grad), double(grad), 1e-3);
    });
}

void test_grad(std::vector<TNodeShPtr<double>> inputs, TNodeShPtr<double> loss)
{
    loss->iterateWeights(
        [&](TNode<double>& weight) { test_node_grad(weight, inputs, loss); });
    for (auto& input : inputs) {
        test_node_grad(*input, inputs, loss);
    }
}

class LinearConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(LinearConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();

    TConnectorShPtr<double> dense =
        TConnector<double>::create<TDenseConnector>(shape.back(), 32ul);
    TConnectorShPtr<double> sigmoid =
        TConnector<double>::create<TSigmoidConnector>();
    TConnectorShPtr<double> sum = TConnector<double>::create<TSumConnector>();

    TNodeShPtr<double> tmp = dense->connect(input);
    tmp                    = sigmoid->connect(tmp);
    TNodeShPtr<double> out = sum->connect(tmp);

    dense->weight(0)->values().uniform();
    dense->weight(1)->values().uniform();

    input->forward();

    out->zeroGrad();
    out->computeGrad();

    // Multiple times to ensure that zeroGrad works correctly
    // out->zeroGrad();
    // out->computeGrad();

    test_grad({input}, out);
}

INSTANTIATE_TEST_SUITE_P(BackwardTests, LinearConnectorTest,
                         ::testing::Values(std::vector<size_t>{32},
                                           std::vector<size_t>{1, 32},
                                           std::vector<size_t>{2, 32},
                                           std::vector<size_t>{2, 3, 16}));

class SkipConnectorTest : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(SkipConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();

    TConnectorShPtr<double> dense_1 =
        TConnector<double>::create<TDenseConnector>(shape.back(), 32ul);

    TConnectorShPtr<double> dense_2 =
        TConnector<double>::create<TDenseConnector>(shape.back(), 32ul);

    TConnectorShPtr<double> sigmoid =
        TConnector<double>::create<TSigmoidConnector>();

    TConnectorShPtr<double> add = TConnector<double>::create<TAddConnector>();

    TConnectorShPtr<double> sum = TConnector<double>::create<TSumConnector>();

    TNodeShPtr<double> tmp_1 = dense_1->connect(input);
    tmp_1                    = sigmoid->connect(tmp_1);

    TNodeShPtr<double> tmp_2 = dense_2->connect(tmp_1);
    tmp_2                    = sigmoid->connect(tmp_2);

    TNodeShPtr<double> comb = add->connect(tmp_1, tmp_2);

    TNodeShPtr<double> out = sum->connect(comb);

    dense_1->weight(0)->values().uniform();
    dense_1->weight(1)->values().uniform();
    dense_2->weight(0)->values().uniform();
    dense_2->weight(1)->values().uniform();

    input->forward();

    out->zeroGrad();
    out->computeGrad();

    // Multiple times to ensure that zeroGrad works correctly
    out->zeroGrad();
    out->computeGrad();

    test_grad({input}, out);
}
INSTANTIATE_TEST_SUITE_P(BackwardTests, SkipConnectorTest,
                         ::testing::Values(std::vector<size_t>{32},
                                           std::vector<size_t>{1, 32},
                                           std::vector<size_t>{2, 32},
                                           std::vector<size_t>{2, 3, 32}));

TEST(BackwardTests, ComplexGraph)
{

    auto dense_1 = TConnector<double>::create<TDenseConnector>(16);

    TConnectorShPtr<double> sigmoid =
        TConnector<double>::create<TSigmoidConnector>();

    TConnectorShPtr<double> add = TConnector<double>::create<TAddConnector>();
    TConnectorShPtr<double> sum = TConnector<double>::create<TSumConnector>();

    // Two inputs
    TNodeShPtr<double> input_1 = TNode<double>::create({16, 16});
    TNodeShPtr<double> input_2 = TNode<double>::create({16, 16});

    // Dense connector
    auto tmp_1_0 = dense_1->connect(input_1);
    tmp_1_0      = sigmoid->connect(tmp_1_0);

    // Reuse same connector
    auto tmp_1_1 = dense_1->connect(tmp_1_0);
    tmp_1_1      = sigmoid->connect(tmp_1_1);

    // Reuse same connector on other input
    auto tmp_2_0 = dense_1->connect(input_2);
    tmp_2_0      = sigmoid->connect(tmp_2_0);

    // Skip connection by addition
    auto tmp_1_3 = add->connect(tmp_1_1, tmp_1_0);
    // Another skip connection
    auto tmp_1_4 = add->connect(tmp_1_3, tmp_1_0);

    // combine two inputs
    auto combined = add->connect(tmp_1_4, tmp_2_0);

    // Sum batches
    auto res = sum->connect(combined);

    input_1->values().uniform();
    input_2->values().uniform();

    dense_1->W().uniform();
    dense_1->B().uniform();

    input_1->forward();
    input_2->forward();

    // res->zeroGrad();
    res->computeGrad();
    res->zeroGrad();
    res->computeGrad();

    test_grad({input_1, input_2}, res);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}