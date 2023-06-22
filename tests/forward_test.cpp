#include "common_modules.h"
#include "forward_declare.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

void compareTensor(Tensor<float>& a, Tensor<float>& b)
{
    auto it = b.begin();
    for(float& val : a) {
        EXPECT_FLOAT_EQ(val, *it);
        ++it;
    }
}

class OneDenseConnectorTest : public ::testing::TestWithParam<std::vector<size_t>>
{};
class MultiDenseConnectorTest : public ::testing::TestWithParam<std::vector<size_t>>
{};

TEST_P(OneDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    NodeShPtr<float> input = Node<float>::create(shape);

    input->values().arangeAlongAxis(-1, 0, input->values().shape(-1) * 2);

    DenseModuleShPtr<float> encode = Module<float>::create<DenseModule>(shape.back(), 32ul);

    auto weights = encode->W();
    auto bias    = encode->B();

    weights->setAllValues(0);

    for(size_t i = 0; i < std::min(weights->shape(0), weights->shape(1)); i++) {
        weights->value(i, i) = 1;
    }

    bias->setAllValues(1);

    NodeShPtr<float> out = encode->call(input);

    out->values().forEach([&](const Index& index) {
        if(!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(out->value(index), 1 + 2 * index[-1]);
        }
    });
}

TEST_P(MultiDenseConnectorTest, input_shape)
{
    auto shape = GetParam();

    NodeShPtr<float>        input  = Node<float>::create(shape);
    DenseModuleShPtr<float> encode = Module<float>::create<DenseModule>(shape.back(), 32ul);
    DenseModuleShPtr<float> decode = Module<float>::create<DenseModule>(32ul, 128ul);

    input->values().setAllValues(1);

    encode->W()->setAllValues(1);
    encode->B()->setAllValues(1);

    decode->W()->setAllValues(1);
    decode->B()->setAllValues(1);

    NodeShPtr<float> out = encode->call(input);
    out                  = decode->call(out);

    out->values().forEach([&](const Index& index) {
        if(!this->HasFatalFailure()) {
            ASSERT_FLOAT_EQ(decode->W()->shape(1) * (encode->W()->shape(1) + 1) + 1,
                            out->value(index));
        }
    });
}

INSTANTIATE_TEST_SUITE_P(OneDenseConnectorTestAllTests, OneDenseConnectorTest,
                         ::testing::Values(std::vector<size_t>{128}, std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

INSTANTIATE_TEST_SUITE_P(MultiDenseConnectorTestAllTests, MultiDenseConnectorTest,
                         ::testing::Values(std::vector<size_t>{128}, std::vector<size_t>{32, 128},
                                           std::vector<size_t>{128, 128},
                                           std::vector<size_t>{32, 128, 32}));

TEST(OwnershipTransfer, linear)
{
    // TODO: Proper deregistration

    NodeShPtr<float> input = Node<float>::create({1});
    input->values().setFlattenedValues({1});

    NodeShPtr<float> out;
    {
        auto sum = Connector<float>::create<SumConnector>();
        out      = sum->call(input);
        out      = sum->call(out);
        out      = sum->call(out);
    }

    EXPECT_FLOAT_EQ(out->value(), 1.0f);
}

TEST(ComplexGraph, complex_graph)
{

    auto dense_1 = Module<float>::create<DenseModule>(2, 2);

    ConnectorShPtr<float> sigmoid = Connector<float>::create<SigmoidConnector>();

    ConnectorShPtr<float> add = Connector<float>::create<AddConnector>();
    ConnectorShPtr<float> sum = Connector<float>::create<SumConnector>();

    // Two inputs
    NodeShPtr<float> input_1 = Node<float>::create({2, 2});
    NodeShPtr<float> input_2 = Node<float>::create({2, 2});

    input_1->values().setFlattenedValues({1, 2, 3, 4});
    input_2->values().setFlattenedValues({3.141, 1.414, 0., 42.});

    dense_1->W()->values().setFlattenedValues({1, -1, -1, 2});
    dense_1->B()->values().setFlattenedValues({-2.5, 2.5});

    // Dense connector
    auto tmp_1_0 = dense_1->call(input_1);
    tmp_1_0      = sigmoid->call(tmp_1_0);

    // Reuse same callor
    auto tmp_1_1 = dense_1->call(tmp_1_0);
    tmp_1_1      = sigmoid->call(tmp_1_1);

    // Reuse same callor on other input
    auto tmp_2_0 = dense_1->call(input_2);
    tmp_2_0      = sigmoid->call(tmp_2_0);

    // Skip callion by addition
    auto tmp_1_3 = add->call(tmp_1_1, tmp_1_0);
    // Another skip callion
    auto tmp_1_4 = add->call(tmp_1_3, tmp_1_0);

    // combine two inputs
    auto combined = add->call(tmp_1_4, tmp_2_0);

    // Sum batches
    auto res = sum->call(combined);

    // For check of correct result: See check.py
    EXPECT_FLOAT_EQ(res->value(), 8.360636886487102);
}

TEST(DotTest, MatrixTimesMatrix)
{
    NodeShPtr<float> a = Node<float>::create({2, 3});
    a->values().setFlattenedValues({1, 2, 3, 4, 5, 6});

    NodeShPtr<float> b = Node<float>::create({3, 2});
    b->values().setFlattenedValues({11, 12, 13, 14, 15, 16});

    Tensor<float> ref({2, 2});
    ref.setFlattenedValues({8.200000e+01, 8.800000e+01, 1.990000e+02, 2.140000e+02});

    compareTensor(Dot(a, b)->values(), ref);

    ref = Tensor<float>({3, 3});
    ref.setFlattenedValues({5.900000e+01, 8.200000e+01, 1.050000e+02, 6.900000e+01, 9.600000e+01,
                            1.230000e+02, 7.900000e+01, 1.100000e+02, 1.410000e+02});
    compareTensor(Dot(b, a)->values(), ref);
}

TEST(DotTest, TensorTimesTensor)
{
    NodeShPtr<float> a = Node<float>::create({2, 2, 3});
    a->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 2, 14);

    NodeShPtr<float> b = Node<float>::create({3, 3, 2});
    b->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 5, 23);

    Tensor<float> ref({2, 2, 3, 2});
    ref.setFlattenedValues({6.700000e+01, 7.600000e+01, 1.210000e+02, 1.300000e+02, 1.750000e+02,
                            1.840000e+02, 1.300000e+02, 1.480000e+02, 2.380000e+02, 2.560000e+02,
                            3.460000e+02, 3.640000e+02, 1.930000e+02, 2.200000e+02, 3.550000e+02,
                            3.820000e+02, 5.170000e+02, 5.440000e+02, 2.560000e+02, 2.920000e+02,
                            4.720000e+02, 5.080000e+02, 6.880000e+02, 7.240000e+02});

    compareTensor(Dot(a, b)->values(), ref);

    ref = Tensor<float>({3, 3, 2, 3});
    ref.setFlattenedValues(
        {4.000000e+01, 5.100000e+01, 6.200000e+01, 1.060000e+02, 1.170000e+02, 1.280000e+02,
         5.400000e+01, 6.900000e+01, 8.400000e+01, 1.440000e+02, 1.590000e+02, 1.740000e+02,
         6.800000e+01, 8.700000e+01, 1.060000e+02, 1.820000e+02, 2.010000e+02, 2.200000e+02,
         8.200000e+01, 1.050000e+02, 1.280000e+02, 2.200000e+02, 2.430000e+02, 2.660000e+02,
         9.600000e+01, 1.230000e+02, 1.500000e+02, 2.580000e+02, 2.850000e+02, 3.120000e+02,
         1.100000e+02, 1.410000e+02, 1.720000e+02, 2.960000e+02, 3.270000e+02, 3.580000e+02,
         1.240000e+02, 1.590000e+02, 1.940000e+02, 3.340000e+02, 3.690000e+02, 4.040000e+02,
         1.380000e+02, 1.770000e+02, 2.160000e+02, 3.720000e+02, 4.110000e+02, 4.500000e+02,
         1.520000e+02, 1.950000e+02, 2.380000e+02, 4.100000e+02, 4.530000e+02, 4.960000e+02});

    compareTensor(Dot(b, a)->values(), ref);
}

TEST(DotTest, TensorTimesTemsor2)
{
    NodeShPtr<float> a = Node<float>::create({2, 3});
    a->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 2, 8);

    NodeShPtr<float> b = Node<float>::create({3, 3, 2});
    b->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 5, 23);

    Tensor<float> ref({2, 3, 2});
    ref.setFlattenedValues({6.700000e+01, 7.600000e+01, 1.210000e+02, 1.300000e+02, 1.750000e+02,
                            1.840000e+02, 1.300000e+02, 1.480000e+02, 2.380000e+02, 2.560000e+02,
                            3.460000e+02, 3.640000e+02});

    compareTensor(Dot(a, b)->values(), ref);

    ref = Tensor<float>({3, 3, 3});
    ref.setFlattenedValues({4.000000e+01, 5.100000e+01, 6.200000e+01, 5.400000e+01, 6.900000e+01,
                            8.400000e+01, 6.800000e+01, 8.700000e+01, 1.060000e+02, 8.200000e+01,
                            1.050000e+02, 1.280000e+02, 9.600000e+01, 1.230000e+02, 1.500000e+02,
                            1.100000e+02, 1.410000e+02, 1.720000e+02, 1.240000e+02, 1.590000e+02,
                            1.940000e+02, 1.380000e+02, 1.770000e+02, 2.160000e+02, 1.520000e+02,
                            1.950000e+02, 2.380000e+02});
    compareTensor(Dot(b, a)->values(), ref);
}

TEST(DotTest, InnerProduct)
{
    NodeShPtr<float> a = Node<float>::create({2});
    a->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 2, 4);

    NodeShPtr<float> b = Node<float>::create({2});
    b->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 5, 7);

    EXPECT_FLOAT_EQ(Dot(a, b)->value(), 28.f);
    EXPECT_FLOAT_EQ(Dot(b, a)->value(), 28.f);
}

TEST(DotTest, MatrixTimesVector)
{
    NodeShPtr<float> a = Node<float>::create({2, 2});
    a->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 2, 6);

    NodeShPtr<float> b = Node<float>::create({2});
    b->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 5, 7);

    Tensor<float> ref({2});
    ref.setFlattenedValues({28, 50});
    compareTensor(Dot(a, b)->values(), ref);

    ref.setFlattenedValues({34, 45});
    compareTensor(Dot(b, a)->values(), ref);
}

TEST(DotTest, ScalarTimesVector)
{
    NodeShPtr<float> a = Node<float>::create();
    a->value()         = 3;
    std::cout << a->values() << std::endl;

    NodeShPtr<float> b = Node<float>::create({2});
    b->values().viewWithNDimsOnTheRight(1).arangeAlongAxis(0, 5, 7);

    Tensor<float> ref({2});
    ref.setFlattenedValues({15, 18});

    compareTensor(Dot(a, b)->values(), ref);
    compareTensor(Dot(b, a)->values(), ref);
}

TEST(DotTest, ScalarTimesScalar)
{
    NodeShPtr<float> a = Node<float>::create();
    a->value()         = 3;

    NodeShPtr<float> b = Node<float>::create();
    b->value()         = 2;

    EXPECT_FLOAT_EQ(Dot(a, b)->value(), 6);
    EXPECT_FLOAT_EQ(Dot(b, a)->value(), 6);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}