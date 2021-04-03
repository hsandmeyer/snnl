#include "connector.h"
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

    TNodePtr<float> input = TNode<float>::create(shape);

    input->values().rangeAllDims(-1, 0, 2);

    TConnectorPtr<float> encode =
        TConnector<float>::create<TDenseConnector>(shape.back(), 32ul);

    TNodePtr<float> out = encode->connect(input);

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

    TNodePtr<float>      input = TNode<float>::create(shape);
    TConnectorPtr<float> encode =
        TConnector<float>::create<TDenseConnector>(shape.back(), 32ul);
    TConnectorPtr<float> decode =
        TConnector<float>::create<TDenseConnector>(32ul, 128ul);

    TNodePtr<float> out = encode->connect(input);
    out                 = decode->connect(out);

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