#include "layer.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

class OneDenseLayerTest : public ::testing::TestWithParam<std::vector<size_t>> {
};
class MultiDenseLayerTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(OneDenseLayerTest, input_shape)
{
    auto shape = GetParam();

    TNode<float> input = TNode<float>::Default(shape);

    input->values().rangeAllDims(-1, 0, 2);

    TLayer<float> encode = TLayer<float>::TDenseLayer(32);
    TNode<float>  out    = encode(input);

    auto &weights = encode->weights(0);
    weights.setAllValues(0);

    for (size_t i = 0; i < std::min(weights.shape(0), weights.shape(1)); i++) {
        weights(i, i) = 1;
    }

    auto &bias = encode->weights(1);
    bias.setAllValues(1);

    input->call();

    out->values().forEach([&](TIndex &index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(out(index), 1 + 2 * index[-1]);
        }
    });
}

TEST_P(MultiDenseLayerTest, input_shape)
{
    auto shape = GetParam();

    TNode<float>  input  = TNode<float>::Default(shape);
    TLayer<float> encode = TLayer<float>::TDenseLayer(32);
    TLayer<float> decode = TLayer<float>::TDenseLayer(128);

    TNode<float> out = encode(input);
    out              = decode(out);

    input->values().setAllValues(1);

    encode->weights(0).setAllValues(1);
    encode->weights(1).setAllValues(1);

    decode->weights(0).setAllValues(1);
    decode->weights(1).setAllValues(1);

    input.get()->call();

    out->values().forEach([&](TIndex &index) {
        if (!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(decode->weights(0).shape(1) *
                                    (encode->weights(0).shape(1) + 1) +
                                1,
                            out(index));
        }
    });
}

INSTANTIATE_TEST_SUITE_P(OneDenseLayerTestAllTests, OneDenseLayerTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

INSTANTIATE_TEST_SUITE_P(MultiDenseLayerTestAllTests, MultiDenseLayerTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}