#include "common_connectors.h"
#include "connectors/connector_sigmoid.h"
#include "connectors/connector_sum.h"
#include "forward_declare.h"
#include "module.h"
#include "modules/module_dense.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <limits>

using namespace snnl;

template <typename TElem>
void compRel(TElem a, TElem b, TElem rel_prec)
{
    if (std::abs(a - b) / (std::max(a, b)) > rel_prec) {
        std::cout << double(a) << " " << double(b) << " " << std::endl;
        FAIL();
    }
}

void test_node_grad(TNode<double>& node, TModule<double>& model,
                    std::vector<TNodeShPtr<double>>& inputs)
{

    node.values().forEach([&](const TIndex& index) {
        double eps        = 1e-4;
        double val_weight = node.value(index);

        node.value(index) = val_weight + eps;

        auto loss = model.call(inputs);

        double val_loss_up = loss->value(0);
        node.value(index)  = val_weight - eps;

        loss = model.call(inputs);

        double val_loss_down = loss->value(0);

        node.value(index) = val_weight;

        double numerical_grad = (val_loss_up - val_loss_down) / (2 * eps);

        double grad = node.grad(index);

        compRel(double(numerical_grad), double(grad), 1e-5);
    });
}

void test_grad(TModule<double>& model, std::vector<TNodeShPtr<double>> inputs)
{
    auto loss = model.call(inputs);
    loss->iterateWeights(
        [&](TNode<double>& weight) { test_node_grad(weight, model, inputs); });
}

class LinearConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(LinearConnectorTest, input_shape)
{
    struct LinearModel : public TModule<double> {
        std::shared_ptr<TDenseModule<double>> dense;

        LinearModel(std::vector<size_t> shape)
        {
            dense = this->addModule<TDenseModule>(shape.back(), 8ul);
        }

        virtual TNodeShPtr<double>
        callHandler(std::vector<TNodeShPtr<double>> inputs) override
        {
            TNodeShPtr<double> tmp = dense->call(inputs.at(0));
            tmp                    = Sigmoid(tmp);
            return Sum(tmp);
        }
    };

    auto               shape = GetParam();
    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();

    LinearModel model(shape);

    model.dense->W()->values().uniform();
    model.dense->B()->values().uniform();

    TNodeShPtr<double> out = model.call(input);

    // Multiple times to ensure that zeroGrad works correctly
    out->computeGrad();
    out->computeGrad();

    test_grad(model, {input});
}

INSTANTIATE_TEST_SUITE_P(BackwardTests, LinearConnectorTest,
                         ::testing::Values(std::vector<size_t>{8},
                                           std::vector<size_t>{1, 8},
                                           std::vector<size_t>{2, 8},
                                           std::vector<size_t>{2, 3, 8}));

class SkipConnectorTest : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(SkipConnectorTest, input_shape)
{

    struct SkipModel : TModule<double> {
        TDenseModuleShPtr<double> dense_1;
        TDenseModuleShPtr<double> dense_2;

        SkipModel(std::vector<size_t> shape)
        {
            dense_1 = addModule<TDenseModule>(shape.back(), 8ul);
            dense_2 = addModule<TDenseModule>(shape.back(), 8ul);
        }

        virtual TNodeShPtr<double>
        callHandler(std::vector<TNodeShPtr<double>> inputs) override
        {
            TNodeShPtr<double> tmp_1 = dense_1->call(inputs[0]);
            tmp_1                    = Sigmoid(tmp_1);

            TNodeShPtr<double> tmp_2 = dense_2->call(tmp_1);
            tmp_2                    = Sigmoid(tmp_2);

            TNodeShPtr<double> comb = Add(tmp_1, tmp_2);

            return Sum(comb);
        }
    };

    auto               shape = GetParam();
    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();
    SkipModel model(shape);

    model.dense_1->W()->values().uniform();
    model.dense_1->B()->values().uniform();
    model.dense_2->W()->values().uniform();
    model.dense_2->B()->values().uniform();

    auto out = model.call(input);

    out->computeGrad();
    out->computeGrad();

    test_grad(model, {input});
}

INSTANTIATE_TEST_SUITE_P(BackwardTests, SkipConnectorTest,
                         ::testing::Values(std::vector<size_t>{8},
                                           std::vector<size_t>{1, 8},
                                           std::vector<size_t>{2, 8},
                                           std::vector<size_t>{2, 3, 8}));

TEST(BackwardTests, ComplexGraph)
{

    struct ComplexModel : TModule<double> {
        TDenseModuleShPtr<double> dense;

        ComplexModel() { dense = addModule<TDenseModule>(8ul, 8ul); }

        virtual TNodeShPtr<double>
        callHandler(std::vector<TNodeShPtr<double>> inputs) override
        {
            auto tmp_1_0 = dense->call(Sin(inputs[0]));
            tmp_1_0      = Sigmoid(tmp_1_0);

            // Reuse same callor
            auto tmp_1_1 = dense->call(tmp_1_0);
            tmp_1_1      = Sigmoid(tmp_1_1);

            // Reuse same callor on other input
            auto tmp_2_0 = dense->call(Sin(inputs[1]));
            tmp_2_0      = Sigmoid(tmp_2_0);

            // Skip callion by addition
            auto tmp_1_3 = Add(tmp_1_1, tmp_1_0);
            // Another skip callion
            auto tmp_1_4 = Add(tmp_1_3, tmp_1_0);

            // combine two inputs
            auto combined = Add(tmp_1_4, tmp_2_0);

            // Sum batches
            return Sum(combined);
        }
    };

    ComplexModel model;

    // Two inputs
    TNodeShPtr<double> input_1 = TNode<double>::create({4, 8});
    TNodeShPtr<double> input_2 = TNode<double>::create({4, 8});

    input_1->values().uniform();
    input_2->values().uniform();

    model.dense->W()->values().uniform();
    model.dense->B()->values().uniform();

    auto res = model.call(input_1, input_2);

    res->computeGrad();
    res->computeGrad();

    test_grad(model, {input_1, input_2});

    // Input is conectect via Sin. Sin does not involve any weights, nor are
    // there any weights above input_1 and input_2 -> Gradient should not have
    // be computed here
    for (auto& input : {input_1, input_2}) {
        for (auto& val : input->gradient()) {
            EXPECT_EQ(val, 0.f);
        }
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}