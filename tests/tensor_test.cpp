#include "tensor.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

using namespace snnl;

class Tensor1DTest : public ::testing::TestWithParam<size_t> {
};

TEST(Tensor0DTest, scalar)
{
    Tensor<float> t0;
    Tensor<float> t1({});
    Tensor<float> t2;
    Tensor<float> t3;

    t0() = 0;
    t1() = 1;
    t2() = 2;
    t3() = 3;

    Tensor<float> res = ((t0 + t1) * t2 - t3) / t2;
    EXPECT_EQ(res(), -0.5);

    Tensor<float> t_view = t1.viewAs({1, 1, 1});
    t_view(0, 0, 0)      = 123;

    EXPECT_EQ(t1(), 123);

    t1.modifyForEach([](auto&) { return -1; });
    EXPECT_EQ(t1(), -1);

    t3.arangeAlongAxis(0, 16, 16);
    EXPECT_EQ(t3(), 16);
}

TEST_P(Tensor1DTest, size)
{
    size_t size = GetParam();

    Tensor<int> t{size};

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

    Tensor<int> t({dim1, dim2});

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

    Tensor<int> t(shape);

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
    Tensor<int> t2 = t;

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

    Tensor<int> t3 = t;

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

    Tensor<int> t(shape);

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
    Tensor<int> t2 = t;

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

    Tensor<int> t3 = t;

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

    Tensor<int> t4 = t;

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

    // Steps of 2
    t.arangeAlongAxis(0, 1, t.shape(0) * 2 + 1);

    for (size_t i = 0; i < t.shape(0); i++) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                for (size_t l = 0; l < t.shape(3); l++) {
                    ASSERT_EQ(t(i, j, k, l), 1 + i * 2);
                }
            }
        }
    }

    // Steps of -2
    t.arangeAlongAxis(2, 5, static_cast<int>(t.shape(2)) * -2 + 5);

    for (size_t i = 0; i < t.shape(0); i++) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                for (size_t l = 0; l < t.shape(3); l++) {
                    ASSERT_EQ(t(i, j, k, l), 5 - k * 2);
                }
            }
        }
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

TEST(ViewTest, CompressAtEnd)
{
    Tensor<int> t({2, 2, 2});
    Tensor<int> t_view = t.viewAs({2, 4});
    for (size_t i = 0; i < t_view.shape(0); ++i) {
        for (size_t j = 0; j < t_view.shape(1); j++) {
            t_view(i, j) = i + j;
        }
    }

    for (size_t i = 0; i < t.shape(0); ++i) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                EXPECT_EQ(i + 2 * j + k, t(i, j, k));
            }
        }
    }
}

TEST(ViewTest, CompressAtFront)
{
    Tensor<int> t({2, 2, 2});
    Tensor<int> t_view = t.viewAs({4, 2});
    for (size_t i = 0; i < t_view.shape(0); ++i) {
        for (size_t j = 0; j < t_view.shape(1); j++) {
            t_view(i, j) = i + j;
        }
    }

    for (size_t i = 0; i < t.shape(0); ++i) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                EXPECT_EQ(2 * i + j + k, t(i, j, k));
            }
        }
    }
}

TEST(ViewTest, CompressAtMiddle)
{
    Tensor<int> t({2, 2, 2, 2});
    Tensor<int> t_view = t.viewAs({2, 4, 2});
    for (size_t i = 0; i < t_view.shape(0); ++i) {
        for (size_t j = 0; j < t_view.shape(1); j++) {
            for (size_t k = 0; k < t_view.shape(2); k++) {
                t_view(i, j, k) = i + j + k;
            }
        }
    }

    for (size_t i = 0; i < t.shape(0); ++i) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                for (size_t l = 0; l < t.shape(3); l++) {
                    EXPECT_EQ(i + 2 * j + k + l, t(i, j, k, l));
                }
            }
        }
    }
}

TEST(AppendAxis, AppendRight)
{
    Tensor<int> t({2});
    t.appendAxis(2);
    t.appendAxis();
    t.setFlattenedValues({0, 1, 2, 3});
    std::cout << t << std::endl;

    for (size_t i = 0; i < t.shape(0); ++i) {
        for (size_t j = 0; j < t.shape(1); j++) {
            for (size_t k = 0; k < t.shape(2); k++) {
                EXPECT_EQ(2 * i + j, t(i, j, k));
            }
        }
    }
}

TEST(AppendAxis, AppendLeft)
{
    Tensor<int> t({4});
    t.prependAxis();
    t.setFlattenedValues({0, 1, 2, 3});
    std::cout << t << std::endl;

    for (size_t i = 0; i < t.shape(0); ++i) {
        for (size_t j = 0; j < t.shape(1); j++) {
            EXPECT_EQ(i + j, t(i, j));
        }
    }
}

TEST(ViewTest, InvalidView)
{
    Tensor<int> t({2, 2, 3});
    ASSERT_THROW(t.viewAs({6, 2}), std::domain_error);
    ASSERT_THROW(t.viewAs({6, 3}), std::domain_error);
    ASSERT_NO_THROW(t.viewAs({2, 6}));
    auto t2 = t.viewAs({2, 6});
    ASSERT_THROW(t.setDims({1, 2, 3}), std::domain_error);
}

TEST(ViewTest, ShrinkTest)
{
    Tensor<int> t({2});
    Tensor<int> t_view = t.shrinkToNDimsFromLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 1}));
    t_view = t.shrinkToNDimsFromLeft(3);
    EXPECT_EQ(t_view.shape(), Index({2, 1, 1}));

    t_view = t.shrinkToNDimsFromRight(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 2}));

    t      = Tensor<int>({2, 2});
    t_view = t.shrinkToNDimsFromLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 2}));
    t_view = t.shrinkToNDimsFromRight(2);
    EXPECT_EQ(t_view.shape(), Index({2, 2}));

    t      = Tensor<int>({2, 2, 2});
    t_view = t.shrinkToNDimsFromLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 4}));

    t_view = t.shrinkToNDimsFromRight(2);
    EXPECT_EQ(t_view.shape(), Index({4, 2}));

    t      = Tensor<int>({2, 2, 2, 2});
    t_view = t.shrinkToNDimsFromLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 8}));

    t_view = t.shrinkToNDimsFromRight(2);
    EXPECT_EQ(t_view.shape(), Index({8, 2}));

    t      = Tensor<int>({});
    t_view = t.shrinkToNDimsFromRight(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 1}));

    t_view = t.shrinkToNDimsFromLeft(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 1}));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
