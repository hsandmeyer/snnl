#include "tensor.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace snnl;

class Tensor1DTest : public ::testing::TestWithParam<size_t> {
};

TEST_P(Tensor1DTest, size)
{
    size_t size = GetParam();

    TTensor<int> t{size};

    for (size_t i = 0; i < size; i++) {
        t(i) = i;
    }

    int i = 0;
    for (auto it = t.begin(); it < t.end(); it++) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
}

class Tensor2DTest
    : public ::testing::TestWithParam<std::pair<size_t, size_t>> {
};

TEST_P(Tensor2DTest, size)
{
    auto pair = GetParam();

    size_t dim1 = pair.first;
    size_t dim2 = pair.second;

    TTensor<int> t({dim1, dim2});

    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            t(i, j) = i * dim2 + j;
        }
    }

    int i = 0;

    for (auto it = t.begin(); it < t.end(); it++) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
}

class Tensor3DTest : public ::testing::TestWithParam<std::array<size_t, 3>> {
};

TEST_P(Tensor3DTest, size)
{
    auto dims = GetParam();

    TTensor<int> t(dims);

    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            for (size_t k = 0; k < dims[2]; k++) {
                t(i, j, k) = i * dims[2] * dims[1] + j * dims[2] + k;
            }
        }
    }

    int i = 0;
    for (auto it = t.begin(); it < t.end(); it++) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
}

INSTANTIATE_TEST_SUITE_P(Tensor1DTestAllTests, Tensor1DTest,
                         ::testing::Values(1, 2, 10));

INSTANTIATE_TEST_SUITE_P(Tensor2DTestAllTests, Tensor2DTest,
                         ::testing::Values(std::pair<size_t, size_t>(1, 1),
                                           std::pair<size_t, size_t>(1, 2),
                                           std::pair<size_t, size_t>(2, 1),
                                           std::pair<size_t, size_t>(2, 2),
                                           std::pair<size_t, size_t>(7, 8),
                                           std::pair<size_t, size_t>(10, 10)));

INSTANTIATE_TEST_SUITE_P(Tensor3DTestAllTests, Tensor3DTest,
                         ::testing::Values(std::array<size_t, 3>{1, 1, 1},
                                           std::array<size_t, 3>{1, 1, 2},
                                           std::array<size_t, 3>{1, 2, 1},
                                           std::array<size_t, 3>{2, 1, 1},
                                           std::array<size_t, 3>{2, 2, 2},
                                           std::array<size_t, 3>{7, 8, 9},
                                           std::array<size_t, 3>{10, 10, 10}));

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
