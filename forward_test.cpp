#include "layer.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

class OneDenseLayerTest : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(OneDenseLayerTest, input_shape)
{
    auto shape = GetParam();

    TNode<float> input = TNode<float>::Default(shape);

    input.getTensor().rangeAllDims(-1, 0, 2);

    TLayer<float> encode = TLayer<float>::TDenseLayer(32, input);

    auto &weights = encode.getWeights(0);
    weights.setAllValues(0);

    for (size_t i = 0; i < std::min(weights.shape(0), weights.shape(1)); i++) {
        weights(i, i) = 1;
    }

    auto &bias = encode.getWeights(1);
    bias.setAllValues(1);

    TNode<float> out = encode(input);

    input.call();

    out.getTensor().forEach([&out](std::vector<size_t> &index) {
        ASSERT_FLOAT_EQ(out(index), 1 + 2 * index[index.size() - 1]);
    });
}

INSTANTIATE_TEST_SUITE_P(OneDenseLayerTestAllTests, OneDenseLayerTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}