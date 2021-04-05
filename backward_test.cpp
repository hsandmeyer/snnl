#include "connector.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

class OneDenseConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(OneDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    TNodeShPtr<float> input = TNode<float>::create(shape);

    input->values().rangeAllDims(-1, 0, 2);

    TConnectorShPtr<float> encode =
        TConnector<float>::create<TDenseConnector>(shape.back(), 32ul);
    TNodeShPtr<float> out = encode->connect(input);

    out->backward();
}

INSTANTIATE_TEST_SUITE_P(OneDenseConnectorTestAllTests, OneDenseConnectorTest,
                         ::testing::Values(std::vector<size_t>{128},
                                           std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}