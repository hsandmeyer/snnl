#include "connector.h"
#include "home/hauke/src/snnl/forward_declare.h"
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

    input->values().rangeAllDims(-1, 0, 2);

    TConnectorShPtr<float> encode =
        TConnector<float>::create<TDenseConnector>(32ul);

    TNodeShPtr<float> out = encode->connect(input);

    auto weights = encode->weight(0);
    weights->setAllValues(0);

    for (size_t i = 0; i < std::min(weights->shape(0), weights->shape(1));
         i++) {
        weights->value(i, i) = 1;
    }

    auto bias = encode->weight(1);
    bias->setAllValues(1);

    input->forward();

    out->values().forEach([&](const TIndex& index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(out->value(index), 1 + 2 * index[-1]);
        }
    });
}

TEST_P(MultiDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<float>      input = TNode<float>::create(shape);
    TConnectorShPtr<float> encode =
        TConnector<float>::create<TDenseConnector>(shape.back(), 32ul);
    TConnectorShPtr<float> decode =
        TConnector<float>::create<TDenseConnector>(32ul, 128ul);

    TNodeShPtr<float> out = encode->connect(input);
    out                   = decode->connect(out);

    input->values().setAllValues(1);

    encode->weight(0)->setAllValues(1);
    encode->weight(1)->setAllValues(1);

    decode->weight(0)->setAllValues(1);
    decode->weight(1)->setAllValues(1);

    input.get()->forward();

    out->values().forEach([&](const TIndex& index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(decode->weight(0)->shape(1) *
                                    (encode->weight(0)->shape(1) + 1) +
                                1,
                            out->value(index));
        }
    });
}

TEST(ComplexGraph, complex_graph)
{

    auto dense_1 = TConnector<float>::create<TDenseConnector>(2);

    TConnectorShPtr<float> sigmoid =
        TConnector<float>::create<TSigmoidConnector>();

    TConnectorShPtr<float> add = TConnector<float>::create<TAddConnector>();
    TConnectorShPtr<float> sum = TConnector<float>::create<TSumConnector>();

    // Two inputs
    TNodeShPtr<float> input_1 = TNode<float>::create({2, 2});
    TNodeShPtr<float> input_2 = TNode<float>::create({2, 2});

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

    input_1->values().setFlattenedValues({1, 2, 3, 4});
    input_2->values().setFlattenedValues({3.141, 1.414, 0., 42.});

    dense_1->W().setFlattenedValues({1, -1, -1, 2});
    dense_1->B().setFlattenedValues({-2.5, 2.5});

    input_1->forward();
    input_2->forward();

    // For check of correct result: See check.py
    EXPECT_FLOAT_EQ(res->value(0), 8.360636886487102);
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}