#include "tensor.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

using namespace snnl;

class Tensor1DTest : public ::testing::TestWithParam<size_t>
{};

TEST(Tensor0DTest, scalar)
{
    Tensor<float> t0;
    Tensor<float> t1;
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

    t1.modifyForEach([](auto&) {
        return -1;
    });
    EXPECT_EQ(t1(), -1);

    t3.arangeAlongAxis(0, 16, 16);
    EXPECT_EQ(t3(), 16);
}

TEST_P(Tensor1DTest, size)
{
    size_t size = GetParam();

    Tensor<int> t{size};

    for(size_t i = 0; i < size; i++) {
        t(i) = i;
    }

    int i = 0;
    for(auto it = t.begin(); it != t.end(); ++it) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
}

class Tensor2DTest : public ::testing::TestWithParam<std::pair<size_t, size_t>>
{};

TEST_P(Tensor2DTest, size)
{
    auto pair = GetParam();

    size_t dim1 = pair.first;
    size_t dim2 = pair.second;

    Tensor<int> t({dim1, dim2});

    for(size_t i = 0; i < dim1; i++) {
        for(size_t j = 0; j < dim2; j++) {
            t(i, j) = i * dim2 + j;
        }
    }

    int i = 0;

    for(auto it = t.begin(); it != t.end(); ++it) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
}

class Tensor3DTest : public ::testing::TestWithParam<std::array<size_t, 3>>
{};

TEST_P(Tensor3DTest, size)
{
    auto shape = GetParam();

    Tensor<int> t(shape);

    for(size_t i = 0; i < shape[0]; i++) {
        for(size_t j = 0; j < shape[1]; j++) {
            for(size_t k = 0; k < shape[2]; k++) {
                t(i, j, k) = i * shape[2] * shape[1] + j * shape[2] + k;
            }
        }
    }

    int i = 0;
    for(auto it = t.begin(); it != t.end(); ++it) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
    Tensor<int> t2      = t.copy();
    auto        t2_view = t2.flatten();

    for(size_t i = 0; i < t2_view.size(); i++) {
        t2_view(i) *= 2;
    }

    i = 0;
    for(auto it = t2.begin(); it != t2.end(); ++it) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t2(i), 2 * t(i));
        }
        i++;
    }

    Tensor<int> t3      = t.copy();
    auto        t3_view = t3.viewWithNDimsOnTheRight(2);

    for(size_t i = 0; i < t3_view.shape(-2); i++) {
        for(size_t j = 0; j < t3_view.shape(-1); j++) {
            t3_view(i, j) *= 2;
        }
    }

    i = 0;
    for(auto it = t3.begin(); it != t3.end(); ++it) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t3(i), 2 * t(i));
        }
        i++;
    }
}

class Tensor4DTest : public ::testing::TestWithParam<std::array<size_t, 4>>
{};

TEST_P(Tensor4DTest, size)
{
    auto shape = GetParam();

    Tensor<int> t(shape);

    for(size_t i = 0; i < shape[0]; i++) {
        for(size_t j = 0; j < shape[1]; j++) {
            for(size_t k = 0; k < shape[2]; k++) {
                for(size_t l = 0; l < shape[3]; l++) {
                    t(i, j, k, l) = i * shape[3] * shape[2] * shape[1] + j * shape[3] * shape[2] +
                                    k * shape[3] + l;
                }
            }
        }
    }

    int i = 0;
    for(auto it = t.begin(); it != t.end(); ++it) {
        {
            ASSERT_EQ(i, *it);
            ASSERT_EQ(i, t(i));
        }
        i++;
    }
    Tensor<int> t2      = t.copy();
    auto        t2_view = t2.flatten();

    for(size_t i = 0; i < t2.size(); i++) {
        t2_view(i) *= 2;
    }

    i = 0;
    for(auto it = t2.begin(); it != t2.end(); ++it) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t2(i), 2 * t(i));
        }
        i++;
    }

    Tensor<int> t3      = t.copy();
    auto        t3_view = t3.viewWithNDimsOnTheRight(2);

    for(size_t i = 0; i < t3_view.shape(-2); i++) {
        for(size_t j = 0; j < t3_view.shape(-1); j++) {
            t3_view(i, j) *= 2;
        }
    }

    i = 0;
    for(auto it = t3.begin(); it != t3.end(); ++it) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t3(i), 2 * t(i));
        }
        i++;
    }

    Tensor<int> t4      = t.copy();
    auto        t4_view = t4.viewWithNDimsOnTheRight(3);

    for(size_t i = 0; i < t4_view.shape(0); i++) {
        for(size_t j = 0; j < t4_view.shape(-2); j++) {
            for(size_t k = 0; k < t4_view.shape(-1); k++) {
                t4_view(i, j, k) *= 2;
            }
        }
    }

    i = 0;
    for(auto it = t4.begin(); it != t4.end(); ++it) {
        {
            ASSERT_EQ(2 * i, *it);
            ASSERT_EQ(t4(i), 2 * t(i));
        }
        i++;
    }

    // Steps of 2
    t.arangeAlongAxis(0, 1, t.shape(0) * 2 + 1);

    for(size_t i = 0; i < t.shape(0); i++) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
                for(size_t l = 0; l < t.shape(3); l++) {
                    ASSERT_EQ(t(i, j, k, l), 1 + i * 2);
                }
            }
        }
    }

    // Steps of -2
    t.arangeAlongAxis(2, 5, static_cast<int>(t.shape(2)) * -2 + 5);

    for(size_t i = 0; i < t.shape(0); i++) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
                for(size_t l = 0; l < t.shape(3); l++) {
                    ASSERT_EQ(t(i, j, k, l), 5 - k * 2);
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Tensor1DTestAllTests, Tensor1DTest, ::testing::Values(1, 2, 10));

INSTANTIATE_TEST_SUITE_P(
    Tensor2DTestAllTests, Tensor2DTest,
    ::testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(1, 2),
                      std::pair<size_t, size_t>(2, 1), std::pair<size_t, size_t>(2, 2),
                      std::pair<size_t, size_t>(7, 8), std::pair<size_t, size_t>(10, 10)));

INSTANTIATE_TEST_SUITE_P(
    Tensor3DTestAllTests, Tensor3DTest,
    ::testing::Values(std::array<size_t, 3>{1, 1, 1}, std::array<size_t, 3>{1, 1, 2},
                      std::array<size_t, 3>{1, 2, 1}, std::array<size_t, 3>{2, 1, 1},
                      std::array<size_t, 3>{2, 2, 2}, std::array<size_t, 3>{7, 8, 9},
                      std::array<size_t, 3>{10, 10, 10}));

INSTANTIATE_TEST_SUITE_P(
    Tensor4DTestAllTests, Tensor4DTest,
    ::testing::Values(std::array<size_t, 4>{1, 1, 1, 1}, std::array<size_t, 4>{1, 1, 1, 2},
                      std::array<size_t, 4>{1, 1, 1, 2}, std::array<size_t, 4>{1, 2, 2, 1},
                      std::array<size_t, 4>{2, 1, 1, 1}, std::array<size_t, 4>{2, 2, 2, 2},
                      std::array<size_t, 4>{7, 8, 9, 10}, std::array<size_t, 4>{10, 10, 10, 10}));

TEST(ViewTest, CompressAtEnd)
{
    Tensor<int> t({2, 2, 2});
    Tensor<int> t_view = t.viewAs({2, 4});
    for(size_t i = 0; i < t_view.shape(0); ++i) {
        for(size_t j = 0; j < t_view.shape(1); j++) {
            t_view(i, j) = i + j;
        }
    }

    for(size_t i = 0; i < t.shape(0); ++i) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
                EXPECT_EQ(i + 2 * j + k, t(i, j, k));
            }
        }
    }
}

TEST(ViewTest, CompressAtFront)
{
    Tensor<int> t({2, 2, 2});
    Tensor<int> t_view = t.viewAs({4, 2});
    for(size_t i = 0; i < t_view.shape(0); ++i) {
        for(size_t j = 0; j < t_view.shape(1); j++) {
            t_view(i, j) = i + j;
        }
    }

    for(size_t i = 0; i < t.shape(0); ++i) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
                EXPECT_EQ(2 * i + j + k, t(i, j, k));
            }
        }
    }
}

TEST(ViewTest, CompressAtMiddle)
{
    Tensor<int> t({2, 2, 2, 2});
    Tensor<int> t_view = t.viewAs({2, 4, 2});
    for(size_t i = 0; i < t_view.shape(0); ++i) {
        for(size_t j = 0; j < t_view.shape(1); j++) {
            for(size_t k = 0; k < t_view.shape(2); k++) {
                t_view(i, j, k) = i + j + k;
            }
        }
    }

    for(size_t i = 0; i < t.shape(0); ++i) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
                for(size_t l = 0; l < t.shape(3); l++) {
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

    for(size_t i = 0; i < t.shape(0); ++i) {
        for(size_t j = 0; j < t.shape(1); j++) {
            for(size_t k = 0; k < t.shape(2); k++) {
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

    for(size_t i = 0; i < t.shape(0); ++i) {
        for(size_t j = 0; j < t.shape(1); j++) {
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
    Tensor<int> t_view = t.viewWithNDimsOnTheLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 1}));

    t_view = t.viewWithNDimsOnTheLeft(3);
    EXPECT_EQ(t_view.shape(), Index({2, 1, 1}));

    t_view = t.viewWithNDimsOnTheRight(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 2}));

    t      = Tensor<int>({2, 2});
    t_view = t.viewWithNDimsOnTheLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 2}));
    t_view = t.viewWithNDimsOnTheRight(2);
    EXPECT_EQ(t_view.shape(), Index({2, 2}));

    t      = Tensor<int>({2, 2, 2});
    t_view = t.viewWithNDimsOnTheLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 4}));

    t_view = t.viewWithNDimsOnTheRight(2);
    EXPECT_EQ(t_view.shape(), Index({4, 2}));

    t      = Tensor<int>({2, 2, 2});
    t_view = t.viewWithNDimsOnTheLeft(1);
    EXPECT_EQ(t_view.shape(), Index({8}));

    t_view = t.viewWithNDimsOnTheRight(1);
    EXPECT_EQ(t_view.shape(), Index({8}));

    t      = Tensor<int>({2, 2, 2, 2});
    t_view = t.viewWithNDimsOnTheLeft(2);
    EXPECT_EQ(t_view.shape(), Index({2, 8}));

    t_view = t.viewWithNDimsOnTheRight(2);
    EXPECT_EQ(t_view.shape(), Index({8, 2}));

    t      = Tensor<int>();
    t_view = t.viewWithNDimsOnTheRight(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 1}));

    t_view = t.viewWithNDimsOnTheLeft(3);
    EXPECT_EQ(t_view.shape(), Index({1, 1, 1}));

    t = Tensor<int>({2, 2, 2});
    EXPECT_THROW(t.viewWithNDimsOnTheLeft(0), std::invalid_argument);
    EXPECT_THROW(t.viewWithNDimsOnTheRight(0), std::invalid_argument);
}

TEST(ViewTest, ShrinkToAxis)
{
    Tensor<int> t({2, 2});
    Tensor<int> t_view = t.viewFromIndices({0});
    EXPECT_EQ(t_view.shape(), Index({1, 4}));

    t_view = t.viewFromIndices({1});
    EXPECT_EQ(t_view.shape(), Index({2, 2}));

    t_view = t.viewFromIndices({2});
    EXPECT_EQ(t_view.shape(), Index({4, 1}));

    t      = Tensor<int>({2, 2, 2});
    t_view = t.viewFromIndices({1});
    EXPECT_EQ(t_view.shape(), Index({2, 4}));

    t_view = t.viewFromIndices({2});
    EXPECT_EQ(t_view.shape(), Index({4, 2}));

    t_view = t.viewFromIndices({-1, -2});
    EXPECT_EQ(t_view.shape(), Index({2, 2, 2}));

    t_view = t.viewFromIndices({0, 1});
    EXPECT_EQ(t_view.shape(), Index({1, 2, 4}));

    t_view = t.viewFromIndices({2, 3});
    EXPECT_EQ(t_view.shape(), Index({4, 2, 1}));

    t_view = t.viewFromIndices({1, 2, 2});
    EXPECT_EQ(t_view.shape(), Index({2, 2, 1, 2}));

    t      = Tensor<int>({2, 2, 2, 2});
    t_view = t.viewFromIndices({0});
    EXPECT_EQ(t_view.shape(), Index({1, 16}));

    t_view = t.viewFromIndices({1});
    EXPECT_EQ(t_view.shape(), Index({2, 8}));

    t_view = t.viewFromIndices({2});
    EXPECT_EQ(t_view.shape(), Index({4, 4}));

    t_view = t.viewFromIndices({3});
    EXPECT_EQ(t_view.shape(), Index({8, 2}));

    t_view = t.viewFromIndices({4});
    EXPECT_EQ(t_view.shape(), Index({16, 1}));

    t_view = t.viewFromIndices({1, 2});
    EXPECT_EQ(t_view.shape(), Index({2, 2, 4}));

    t_view = t.viewFromIndices({0, 2});
    EXPECT_EQ(t_view.shape(), Index({1, 4, 4}));

    t_view = t.viewFromIndices({0, -2});
    EXPECT_EQ(t_view.shape(), Index({1, 4, 4}));

    t_view = t.viewFromIndices({-2, 0});
    EXPECT_EQ(t_view.shape(), Index({1, 4, 4}));

    t_view = t.viewFromIndices({1, 2, 2, 2});
    EXPECT_EQ(t_view.shape(), Index({2, 2, 1, 1, 4}));
}

TEST(OperatorTest, BroadCasting)
{
    Tensor<int> a({2});

    a.setFlattenedValues({2, 3});

    Tensor<int> b({2, 2});
    b.setFlattenedValues({1, 2, 3, 4});
    Tensor tmp = b.copy();

    b *= a;
    for(size_t i = 0; i < b.shape(0); ++i) {
        for(size_t j = 0; j < b.shape(0); ++j) {
            std::cout << i << " " << j << " " << tmp.shape() << std::endl;
            EXPECT_EQ(b(i, j), tmp(i, j) * a(j));
        }
    }

    b.setFlattenedValues({1, 2, 3, 4});
    Tensor<int> c({2, 2, 2});

    c.setFlattenedValues({1, 2, 3, 4, 5, 6, 7, 8});
    tmp = c;
    c *= b;

    for(size_t i = 0; i < b.shape(0); ++i) {
        for(size_t j = 0; j < b.shape(0); ++j) {
            for(size_t k = 0; k < b.shape(0); ++k) {
                EXPECT_EQ(c(i, j, k), tmp(i, j, k) * b(j, k));
            }
        }
    }

    c.setFlattenedValues({1, 2, 3, 4, 5, 6, 7, 8});
    c *= a;
    for(size_t i = 0; i < b.shape(0); ++i) {
        for(size_t j = 0; j < b.shape(0); ++j) {
            for(size_t k = 0; k < b.shape(0); ++k) {
                EXPECT_EQ(c(i, j, k), tmp(i, j, k) * a(k));
            }
        }
    }

    c = tmp * b;
    for(size_t i = 0; i < b.shape(0); ++i) {
        for(size_t j = 0; j < b.shape(0); ++j) {
            for(size_t k = 0; k < b.shape(0); ++k) {
                EXPECT_EQ(c(i, j, k), tmp(i, j, k) * b(j, k));
            }
        }
    }

    c = tmp * a;
    for(size_t i = 0; i < b.shape(0); ++i) {
        for(size_t j = 0; j < b.shape(0); ++j) {
            for(size_t k = 0; k < b.shape(0); ++k) {
                EXPECT_EQ(c(i, j, k), tmp(i, j, k) * a(k));
            }
        }
    }

    {
        Tensor<int> t{2, 2, 2};
        auto        t_view = t.flatten();
        t_view.arangeAlongAxis(0, 0, 8);

        t *= 2;
        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), 2 * i);
        }

        t_view.arangeAlongAxis(0, 0, 8);
        t = 2 * t;

        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), 2 * i);
        }

        t_view.arangeAlongAxis(0, 0, 8);
        t = t * 2;

        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), i * 2);
        }
    }

    {
        Tensor<int> t{2, 2, 2};
        auto        t_view = t.flatten();
        t_view.arangeAlongAxis(0, 0, 8);

        t /= 2;
        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), i / 2);
        }

        t_view.arangeAlongAxis(0, 0, 8);
        t = t / 2;

        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), i / 2);
        }

        t_view.arangeAlongAxis(0, 1, 9);
        t = 2 / t;

        for(size_t i = 0; i < t.flatten().size(); i++) {
            EXPECT_EQ(t.flatten()(i), 2 / (i + 1));
        }
    }
}

TEST(PartialViewTest, TestDims)
{
    {
        Tensor<float> source{4, 3, 2};
        auto          view = source.viewAs(range(Full{}, 2), range(1, 3), ellipsis());
        Tensor<float> to_assign{2, 2, 2};
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        view.setAllValues(2);

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i < 2 && j >= 1 && j < 3) {
                        EXPECT_EQ(source(i, j, k), 2);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 2);
        }

        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        to_assign.setAllValues(3);
        view = to_assign;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i < 2 && j >= 1 && j < 3) {
                        EXPECT_EQ(source(i, j, k), 3);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 3);
        }
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
    }

    {
        Tensor<float> source{4, 3, 2};
        auto          view = source.viewAs(1, all(), ellipsis());
        Tensor<float> to_assign{3, 2};
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        view.setAllValues(2);

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i == 1) {
                        EXPECT_EQ(source(i, j, k), 2);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 2);
        }

        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        to_assign.setAllValues(3);
        view = to_assign;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i == 1) {
                        EXPECT_EQ(source(i, j, k), 3);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 3);
        }
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
    }

    {
        Tensor<float> source{4, 3, 2};
        auto          view = source.viewAs(1, ellipsis());
        Tensor<float> to_assign{3, 2};
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        view.setAllValues(2);

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i == 1) {
                        EXPECT_EQ(source(i, j, k), 2);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 2);
        }

        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        to_assign.setAllValues(3);
        view = to_assign;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(i == 1) {
                        EXPECT_EQ(source(i, j, k), 3);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 3);
        }
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
    }

    {
        Tensor<float> source{4, 3, 2};
        auto          view = source.viewAs(ellipsis(), range(0, 1));
        Tensor<float> to_assign{4, 3, 1};
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        view.setAllValues(2);

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(k == 0) {
                        EXPECT_EQ(source(i, j, k), 2);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 2);
        }

        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
        to_assign.setAllValues(3);
        view = to_assign;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                for(size_t k = 0; k < source.shape(2); k++) {
                    if(k == 0) {
                        EXPECT_EQ(source(i, j, k), 3);
                    }
                    else {
                        EXPECT_EQ(source(i, j, k), 0);
                    }
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 3);
        }
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
    }
    {
        Tensor<float> source{3, 3};
        auto          view = source.viewAs(newAxis(), newAxis(), ellipsis(), newAxis(), range(0, 2),
                                           newAxis(), NewAxis());
        EXPECT_EQ(view.shape(), Index({1, 1, 3, 1, 2, 1, 1}));
        view.setAllValues(2);

        Tensor<float> to_assign{1, 1, 3, 1, 2, 1, 1};
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                if(j < 2) {
                    EXPECT_EQ(source(i, j), 2);
                }
                else {
                    EXPECT_EQ(source(i, j), 0);
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 2);
        }

        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;

        to_assign.setAllValues(3);
        view = to_assign;

        for(size_t i = 0; i < source.shape(0); i++) {
            for(size_t j = 0; j < source.shape(1); j++) {
                if(j < 2) {
                    EXPECT_EQ(source(i, j), 3);
                }
                else {
                    EXPECT_EQ(source(i, j), 0);
                }
            }
        }
        for(auto& val : view) {
            EXPECT_EQ(val, 3);
        }
        // std::cout << "Source " << source << std::endl;
        // std::cout << "View " << view << std::endl;
    }
}

TEST(InputOutputTest, ToByteTest)
{
    Tensor<float> a({2, 7, 3});
    a.uniform();

    auto          array = a.toByteArray();
    Tensor<float> b;
    b.fromByteArray(array);

    EXPECT_EQ(a.shape(), b.shape());
    for(size_t i = 0; i < a.size(); i++) {
        EXPECT_EQ(a(i), b(i));
    }

    a = Tensor<float>();
    a.uniform();
    array = a.toByteArray();
    b.fromByteArray(array);

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a(), b());
}

TEST(InputOutputTest, BMPTest)
{
    Tensor<float> a({128, 128, 3});
    a.uniform();
    for(size_t i = 0; i < a.shape(0); i++) {
        a(i, 0, 0) = 1;
        a(i, 0, 1) = -1;
        a(i, 0, 2) = -1;
    }

    for(size_t i = 0; i < a.shape(1); i++) {
        a(0, i, 1) = 1;
        a(0, i, 0) = -1;
        a(0, i, 2) = -1;
    }

    for(size_t i = 0; i < std::min(a.shape(1), a.shape(0)); i++) {
        a(i, i, 0) = -1;
        a(i, i, 1) = -1;
        a(i, i, 2) = 1;
    }

    a.saveToBMP("test.bmp", -1, 1);

    Tensor<float> b({128, 128, 1});
    b.uniform();

    b.saveToBMP("test2.bmp", -1, 1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
