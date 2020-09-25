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
    auto shape = GetParam();

    TTensor<int> t(shape);

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                t(i, j, k) = i * shape[2] * shape[1] + j * shape[2] + k;
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
    TTensor<int> t2 = t;

    for (size_t i = 0; i < t2.shapeFlattened(-1); i++) {
        t2(i) *= 2;
    }

    i = 0;
    for (auto it = t2.begin(); it < t2.end(); it++) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t2(i), 2 * t(i));
        }
        i++;
    }

    TTensor<int> t3 = t;

    for (size_t i = 0; i < t3.shapeFlattened(-2); i++) {
        for (size_t j = 0; j < t3.shape(-1); j++) {
            t3(i, j) *= 2;
        }
    }

    i = 0;
    for (auto it = t3.begin(); it < t3.end(); it++) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t3(i), 2 * t(i));
        }
        i++;
    }
}

class Tensor4DTest : public ::testing::TestWithParam<std::array<size_t, 4>> {
};

TEST_P(Tensor4DTest, size)
{
    auto shape = GetParam();

    TTensor<int> t(shape);

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                for (size_t l = 0; l < shape[3]; l++) {
                    t(i, j, k, l) = i * shape[3] * shape[2] * shape[1] +
                                    j * shape[3] * shape[2] + k * shape[3] + l;
                }
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
    TTensor<int> t2 = t;

    for (size_t i = 0; i < t2.shapeFlattened(-1); i++) {
        t2(i) *= 2;
    }

    i = 0;
    for (auto it = t2.begin(); it < t2.end(); it++) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t2(i), 2 * t(i));
        }
        i++;
    }

    TTensor<int> t3 = t;

    for (size_t i = 0; i < t3.shapeFlattened(-2); i++) {
        for (size_t j = 0; j < t3.shape(-1); j++) {
            t3(i, j) *= 2;
        }
    }

    i = 0;
    for (auto it = t3.begin(); it < t3.end(); it++) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t3(i), 2 * t(i));
        }
        i++;
    }

    TTensor<int> t4 = t;

    for (size_t i = 0; i < t4.shapeFlattened(1); i++) {
        for (size_t j = 0; j < t4.shape(-2); j++) {
            for (size_t k = 0; k < t4.shape(-1); k++) {
                t4(i, j, k) *= 2;
            }
        }
    }

    i = 0;
    for (auto it = t4.begin(); it < t4.end(); it++) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t4(i), 2 * t(i));
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

INSTANTIATE_TEST_SUITE_P(Tensor4DTestAllTests, Tensor4DTest,
                         ::testing::Values(std::array<size_t, 4>{1, 1, 1, 1},
                                           std::array<size_t, 4>{1, 1, 1, 2},
                                           std::array<size_t, 4>{1, 1, 1, 2},
                                           std::array<size_t, 4>{1, 2, 2, 1},
                                           std::array<size_t, 4>{2, 1, 1, 1},
                                           std::array<size_t, 4>{2, 2, 2, 2},
                                           std::array<size_t, 4>{7, 8, 9, 10},
                                           std::array<size_t, 4>{10, 10, 10,
                                                                 10}));

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
