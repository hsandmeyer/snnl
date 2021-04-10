#include "connector.h"
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

void test_node_grad(TNode<__float128>*                   node,
                    std::vector<TNodeShPtr<__float128>>& inputs,
                    TNodeShPtr<__float128>&              loss)
{

    node->values().forEach([&](const TIndex& index) {
        __float128 eps        = 1e-17;
        __float128 val_weight = node->value(index);

        node->value(index) = val_weight + eps;
        for (auto& input : inputs) {
            input->forward();
        }

        __float128 val_loss_up = loss->value(0);
        node->value(index)     = val_weight - eps;

        for (auto& input : inputs) {
            input->forward();
        }

        __float128 val_loss_down = loss->value(0);

        __float128 numerical_grad = (val_loss_up - val_loss_down) / (2 * eps);

        __float128 grad = node->grad(index);

        compRel(double(numerical_grad), double(grad), 1e-10);
    });
}

void test_grad(std::vector<TNodeShPtr<__float128>> inputs,
               TNodeShPtr<__float128>              loss)
{
    loss->iterateConnectorsBackwards([&](TConnector<__float128>& conn) {
        for (auto& weight_node : conn.weights()) {
            test_node_grad(weight_node, inputs, loss);
        }
    });
}

class LinearConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(LinearConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<__float128> input = TNode<__float128>::create(shape);

    input->values().uniform();

    TConnectorShPtr<__float128> encode =
        TConnector<__float128>::create<TDenseConnector>(shape.back(), 32ul);
    TConnectorShPtr<__float128> sigmoid =
        TConnector<__float128>::create<TSigmoidConnector>();
    TConnectorShPtr<__float128> sum =
        TConnector<__float128>::create<TSumConnector>();

    TNodeShPtr<__float128> tmp = encode->connect(input);
    tmp                        = sigmoid->connect(tmp);
    TNodeShPtr<__float128> out = sum->connect(tmp);

    //    encode->weight(0)->values().uniform();
    encode->weight(1)->values().uniform();

    input->forward();

    out->zeroGrad();
    out->computeGrad();

    // Multiple times to ensure that zeroGrad works correctly
    out->zeroGrad();
    out->computeGrad();

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

    TNodeShPtr<__float128> input = TNode<__float128>::create(shape);

    input->values().uniform();

    TConnectorShPtr<__float128> encode =
        TConnector<__float128>::create<TDenseConnector>(shape.back(), 32ul);

    TConnectorShPtr<__float128> sigmoid =
        TConnector<__float128>::create<TSigmoidConnector>();

    TConnectorShPtr<__float128> add =
        TConnector<__float128>::create<TAddConnector>();

    TConnectorShPtr<__float128> sum =
        TConnector<__float128>::create<TSumConnector>();

    TNodeShPtr<__float128> tmp_1 = encode->connect(input);
    tmp_1                        = sigmoid->connect(tmp_1);

    TNodeShPtr<__float128> tmp_2 = encode->connect(tmp_1);
    tmp_2                        = sigmoid->connect(tmp_2);

    TNodeShPtr<__float128> comb = add->connect(tmp_1, tmp_2);

    TNodeShPtr<__float128> out = sum->connect(comb);

    encode->weight(0)->values().uniform();
    encode->weight(1)->values().uniform();

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

    auto dense_1 = TConnector<__float128>::create<TDenseConnector>(16);

    TConnectorShPtr<__float128> sigmoid =
        TConnector<__float128>::create<TSigmoidConnector>();

    TConnectorShPtr<__float128> add =
        TConnector<__float128>::create<TAddConnector>();
    TConnectorShPtr<__float128> sum =
        TConnector<__float128>::create<TSumConnector>();

    // Two inputs
    TNodeShPtr<__float128> input_1 = TNode<__float128>::create({16, 16});
    TNodeShPtr<__float128> input_2 = TNode<__float128>::create({16, 16});

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

    res->backward();
    res->zeroGrad();

    res->computeGrad();

    test_grad({input_1, input_2}, res);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}