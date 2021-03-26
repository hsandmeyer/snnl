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

    input->values().rangeAllDims(-1, 0, 2);

    TLayer<float> encode = TLayer<float>::TDenseLayer(shape.back(), 32ul);
    TNode<float>  out    = encode(input);

    out->backCall();
}

INSTANTIATE_TEST_SUITE_P(OneDenseLayerTestAllTests, OneDenseLayerTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}