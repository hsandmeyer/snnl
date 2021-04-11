#include "common_connectors.h"
#include "forward_declare.h"
#include "node.h"
#include <fstream>

using namespace snnl;

template <typename TElem>
void compRel(TElem a, TElem b, TElem rel_prec)
{
    if (std::abs(a - b) / (std::max(a, b)) > rel_prec) {
        std::cout << double(a) << " " << double(b) << " " << std::endl;
        std::cout << "FAIL" << std::endl;
    }
}

void test_node_grad(TNode<double>&                   node,
                    std::vector<TNodeShPtr<double>>& inputs,
                    TNodeShPtr<double>&              loss)
{

    node.values().forEach([&](const TIndex& index) {
        double eps        = 1e-4;
        double val_weight = node.value(index);

        node.value(index) = val_weight + eps;
        for (auto& input : inputs) {
            input->forward();
        }

        double val_loss_up = loss->value(0);
        node.value(index)  = val_weight - eps;

        for (auto& input : inputs) {
            input->forward();
        }

        double val_loss_down = loss->value(0);

        node.value(index) = val_weight;

        double numerical_grad = (val_loss_up - val_loss_down) / (2 * eps);

        double grad = node.grad(index);

        compRel(double(numerical_grad), double(grad), 1e-3);
    });
}

void test_grad(std::vector<TNodeShPtr<double>> inputs, TNodeShPtr<double> loss)
{
    loss->iterateWeights(
        [&](TNode<double>& weight) { test_node_grad(weight, inputs, loss); });
    // for (auto& input : inputs) {
    //    test_node_grad(*input, inputs, loss);
    //}
}

int main()
{
    TNodeShPtr<double> input = TNode<double>::create({4, 1});

    TConnectorShPtr<double> dense1 =
        TConnector<double>::create<TDenseConnector>(64);
    TConnectorShPtr<double> dense2 =
        TConnector<double>::create<TDenseConnector>(16);
    TConnectorShPtr<double> dense3 =
        TConnector<double>::create<TDenseConnector>(1);
    TConnectorShPtr<double> sigmoid =
        TConnector<double>::create<TSigmoidConnector>();

    TNodeShPtr<double> out = dense1->connect(input);
    out                    = sigmoid->connect(out);
    out                    = dense2->connect(out);
    out                    = sigmoid->connect(out);
    out                    = dense3->connect(out);
    // out                    = sigmoid->connect(out);

    TNodeShPtr<double> correct = TNode<double>::create({4, 1});
    correct->setConstant(true);

    TConnectorShPtr<double> mse =
        TDenseConnector<double>::create<TMSEConnector>();
    TNodeShPtr<double> loss = mse->connect(correct, out);

    dense1->weight(0)->values().uniform(-1, 1);
    dense1->weight(1)->values().uniform(-1, 1);
    dense2->weight(0)->values().uniform(-1, 1);
    dense2->weight(1)->values().uniform(-1, 1);

    for (size_t step = 0; step < 100000; step++) {
        input->values().uniform(-M_PI, M_PI);

        // std::cout << input->values() << std::endl;

        for (size_t ind = 0; ind < correct->shapeFlattened(-1); ind++) {
            correct->value(ind) = std::sin(input->value(ind));
            // correct->value(ind) = input->value(ind);
        }

        input->forward();

        if (step % 500 == 0) {
            std::cout << loss->value(0) << " ";
            for (size_t i = 0; i < correct->shapeFlattened(-1); i++) {
                std::cout << out->value(i) - correct->value(i) << " ";
            }
            std::cout << std::endl;
            std::ofstream fout("test.txt");

            // input->setDims({1, 1});
            // correct->setDims({1, 1});
            for (int i = 0; i < 100; i++) {
                double x = -M_PI + i * 2 * M_PI / 100.;
                ;
                fout << x << " " << std::sin(x) << " ";
                input->value(0) = x;
                input->forward();
                fout << out->value(0) << std::endl;
            }
            // input->setDims({4, 1});
            // correct->setDims({4, 1});
        }

        loss->zeroGrad();
        loss->computeGrad();
        // test_grad({input}, loss);

        double eps = 1e-1;

        loss->iterateWeights([&](TNode<double>& weight) {
            for (size_t ind = 0; ind < weight.shapeFlattened(-1); ind++) {
                weight.value(ind) = weight.value(ind) - eps * weight.grad(ind);
            }
        });
    }
}