#include "common_modules.h"
#include "forward_declare.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

class OneDenseConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};
class MultiDenseConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(OneDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<float> input = TNode<float>::create(shape);

    input->values().arangeAlongAxis(-1, 0, input->values().shape(-1) * 2);

    TDenseModuleShPtr<float> encode =
        TModule<float>::create<TDenseModule>(shape.back(), 32ul);

    auto weights = encode->W();
    auto bias    = encode->B();

    weights->setAllValues(0);

    for (size_t i = 0; i < std::min(weights->shape(0), weights->shape(1));
         i++) {
        weights->value(i, i) = 1;
    }

    bias->setAllValues(1);

    TNodeShPtr<float> out = encode->call(input);

    out->values().forEach([&](const TIndex& index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(out->value(index), 1 + 2 * index[-1]);
        }
    });
}

TEST_P(MultiDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<float>        input = TNode<float>::create(shape);
    TDenseModuleShPtr<float> encode =
        TModule<float>::create<TDenseModule>(shape.back(), 32ul);
    TDenseModuleShPtr<float> decode =
        TModule<float>::create<TDenseModule>(32ul, 128ul);

    input->values().setAllValues(1);

    encode->W()->setAllValues(1);
    encode->B()->setAllValues(1);

    decode->W()->setAllValues(1);
    decode->B()->setAllValues(1);

    TNodeShPtr<float> out = encode->call(input);
    out                   = decode->call(out);

    out->values().forEach([&](const TIndex& index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(
                decode->W()->shape(1) * (encode->W()->shape(1) + 1) + 1,
                out->value(index));
        }
    });
}

INSTANTIATE_TEST_SUITE_P(OneDenseConnectorTestAllTests, OneDenseConnectorTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

INSTANTIATE_TEST_SUITE_P(MultiDenseConnectorTestAllTests,
                         MultiDenseConnectorTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

TEST(OwnershipTransfer, linear)
{
    // TODO: Proper deregistration

    TNodeShPtr<float> input = TNode<float>::create({1});
    input->values().setFlattenedValues({1});

    TNodeShPtr<float> out;
    {
        auto sum = TConnector<float>::create<TSumConnector>();
        out      = sum->call(input);
        out      = sum->call(out);
        out      = sum->call(out);
    }

    EXPECT_FLOAT_EQ(out->value(0), 1.0f);
}

TEST(ComplexGraph, complex_graph)
{

    auto dense_1 = TModule<float>::create<TDenseModule>(2, 2);

    TConnectorShPtr<float> sigmoid =
        TConnector<float>::create<TSigmoidConnector>();

    TConnectorShPtr<float> add = TConnector<float>::create<TAddConnector>();
    TConnectorShPtr<float> sum = TConnector<float>::create<TSumConnector>();

    // Two inputs
    TNodeShPtr<float> input_1 = TNode<float>::create({2, 2});
    TNodeShPtr<float> input_2 = TNode<float>::create({2, 2});

    input_1->values().setFlattenedValues({1, 2, 3, 4});
    input_2->values().setFlattenedValues({3.141, 1.414, 0., 42.});

    dense_1->W()->values().setFlattenedValues({1, -1, -1, 2});
    dense_1->B()->values().setFlattenedValues({-2.5, 2.5});

    // Dense connector
    auto tmp_1_0 = dense_1->call(input_1);
    tmp_1_0      = sigmoid->call(tmp_1_0);

    // Reuse same callor
    auto tmp_1_1 = dense_1->call(tmp_1_0);
    tmp_1_1      = sigmoid->call(tmp_1_1);

    // Reuse same callor on other input
    auto tmp_2_0 = dense_1->call(input_2);
    tmp_2_0      = sigmoid->call(tmp_2_0);

    // Skip callion by addition
    auto tmp_1_3 = add->call(tmp_1_1, tmp_1_0);
    // Another skip callion
    auto tmp_1_4 = add->call(tmp_1_3, tmp_1_0);

    // combine two inputs
    auto combined = add->call(tmp_1_4, tmp_2_0);

    // Sum batches
    auto res = sum->call(combined);

    // For check of correct result: See check.py
    EXPECT_FLOAT_EQ(res->value(0), 8.360636886487102);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}