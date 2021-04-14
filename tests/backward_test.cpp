#include "common_connectors.h"
#include "connectors/connector_dense.h"
#include "forward_declare.h"
#include "model.h"
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

void test_node_grad(TNode<double>& node, TModel<double>& model,
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

        compRel(double(numerical_grad), double(grad), 1e-3);
    });
}

void test_grad(TModel<double>& model, std::vector<TNodeShPtr<double>> inputs)
{
    auto loss = model.call(inputs);
    loss->iterateWeights(
        [&](TNode<double>& weight) { test_node_grad(weight, model, inputs); });
    for (auto& input : inputs) {
        test_node_grad(*input, model, inputs);
    }
}

class LinearConnectorTest
    : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(LinearConnectorTest, input_shape)
{
    struct LinearModel : TModel<double> {
        TConnectorShPtr<double> dense;
        TConnectorShPtr<double> sigmoid;
        TConnectorShPtr<double> sum;

        LinearModel(std::vector<size_t> shape)
        {

            dense   = registerConnector<TDenseConnector>(shape.back(), 32ul);
            sigmoid = registerConnector<TSigmoidConnector>();
            sum     = registerConnector<TSumConnector>();
        }

        virtual TNodeShPtr<double>
        call(std::vector<TNodeShPtr<double>> inputs) override
        {
            TNodeShPtr<double> tmp = dense->call(inputs.at(0));
            tmp                    = sigmoid->call(tmp);

            return sum->call(tmp);
        }
    };

    auto               shape = GetParam();
    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();

    LinearModel model(shape);

    model.dense->weight(0)->values().uniform();
    model.dense->weight(1)->values().uniform();

    TNodeShPtr<double> out = model.call({input});

    out->zeroGrad();
    out->computeGrad();

    // Multiple times to ensure that zeroGrad works correctly
    // out->zeroGrad();
    // out->computeGrad();

    test_grad(model, {input});
}

INSTANTIATE_TEST_SUITE_P(BackwardTests, LinearConnectorTest,
                         ::testing::Values(std::vector<size_t>{32},
                                           std::vector<size_t>{1, 32},
                                           std::vector<size_t>{2, 32},
                                           std::vector<size_t>{2, 3, 16}));

class SkipConnectorTest : public ::testing::TestWithParam<std::vector<size_t>> {
};

TEST_P(SkipConnectorTest, input_shape)
{

    struct SkipModel : TModel<double> {
        TConnectorShPtr<double> dense_1;
        TConnectorShPtr<double> dense_2;
        TConnectorShPtr<double> sigmoid;
        TConnectorShPtr<double> add;
        TConnectorShPtr<double> sum;

        SkipModel(std::vector<size_t> shape)
        {

            dense_1 = registerConnector<TDenseConnector>(shape.back(), 32ul);
            dense_2 = registerConnector<TDenseConnector>(shape.back(), 32ul);
            sigmoid = registerConnector<TSigmoidConnector>();
            sum     = registerConnector<TSumConnector>();
            add     = registerConnector<TAddConnector>();
        }

        virtual TNodeShPtr<double>
        call(std::vector<TNodeShPtr<double>> inputs) override
        {
            TNodeShPtr<double> tmp_1 = dense_1->call(inputs[0]);
            tmp_1                    = sigmoid->call(tmp_1);

            TNodeShPtr<double> tmp_2 = dense_2->call(tmp_1);
            tmp_2                    = sigmoid->call(tmp_2);

            TNodeShPtr<double> comb = add->call(tmp_1, tmp_2);

            return sum->call(comb);
        }
    };

    auto               shape = GetParam();
    TNodeShPtr<double> input = TNode<double>::create(shape);

    input->values().uniform();
    SkipModel model(shape);

    model.dense_1->weight(0)->values().uniform();
    model.dense_1->weight(1)->values().uniform();
    model.dense_2->weight(0)->values().uniform();
    model.dense_2->weight(1)->values().uniform();

    auto out = model.call({input});

    out->zeroGrad();
    out->computeGrad();

    // Multiple times to ensure that zeroGrad works correctly
    out->zeroGrad();
    out->computeGrad();

    test_grad(model, {input});
}

INSTANTIATE_TEST_SUITE_P(BackwardTests, SkipConnectorTest,
                         ::testing::Values(std::vector<size_t>{32},
                                           std::vector<size_t>{1, 32},
                                           std::vector<size_t>{2, 32},
                                           std::vector<size_t>{2, 3, 32}));

TEST(BackwardTests, ComplexGraph)
{

    struct ComplexModel : TModel<double> {
        std::shared_ptr<TDenseConnector<double>> dense;
        TConnectorShPtr<double>                  sigmoid;
        TConnectorShPtr<double>                  add;
        TConnectorShPtr<double>                  sum;

        ComplexModel()
        {
            dense   = registerConnector<TDenseConnector>(16ul, 16ul);
            sigmoid = registerConnector<TSigmoidConnector>();
            sum     = registerConnector<TSumConnector>();
            add     = registerConnector<TAddConnector>();
        }

        virtual TNodeShPtr<double>
        call(std::vector<TNodeShPtr<double>> inputs) override
        {
            auto tmp_1_0 = dense->call(inputs[0]);
            tmp_1_0      = sigmoid->call(tmp_1_0);

            // Reuse same callor
            auto tmp_1_1 = dense->call(tmp_1_0);
            tmp_1_1      = sigmoid->call(tmp_1_1);

            // Reuse same callor on other input
            auto tmp_2_0 = dense->call(inputs[1]);
            tmp_2_0      = sigmoid->call(tmp_2_0);

            // Skip callion by addition
            auto tmp_1_3 = add->call(tmp_1_1, tmp_1_0);
            // Another skip callion
            auto tmp_1_4 = add->call(tmp_1_3, tmp_1_0);

            // combine two inputs
            auto combined = add->call(tmp_1_4, tmp_2_0);

            // Sum batches
            return sum->call(combined);
        }
    };

    ComplexModel model;

    // Two inputs
    TNodeShPtr<double> input_1 = TNode<double>::create({16, 16});
    TNodeShPtr<double> input_2 = TNode<double>::create({16, 16});

    input_1->values().uniform();
    input_2->values().uniform();

    model.dense->W().uniform();
    model.dense->B().uniform();

    auto res = model.call({input_1, input_2});

    res->zeroGrad();
    res->computeGrad();

    res->zeroGrad();
    res->computeGrad();

    test_grad(model, {input_1, input_2});
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}