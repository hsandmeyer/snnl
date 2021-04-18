#include "common_modules.h"
#include "forward_declare.h"
#include "node.h"
#include <fstream>

using namespace snnl;

int main()
{
    TNodeShPtr<double> input = TNode<double>::create({4, 1});

    TDenseModule<double> dense1(1, 64);
    TDenseModule<double> dense2(64, 16);
    TDenseModule<double> dense3(16, 1);

    TNodeShPtr<double> correct = TNode<double>::create({4, 1});
    correct->setConstant(true);

    TConnectorShPtr<double> mse =
        TDenseConnector<double>::create<TMSEConnector>();

    dense1.W()->values().uniform(-1, 1);
    dense1.B()->values().uniform(-1, 1);
    dense2.W()->values().uniform(-1, 1);
    dense2.B()->values().uniform(-1, 1);

    for (size_t step = 0; step < 10000; step++) {
        input->values().uniform(-M_PI, M_PI);

        // std::cout << input->values() << std::endl;

        for (size_t ind = 0; ind < correct->shapeFlattened(-1); ind++) {
            correct->value(ind) = std::sin(input->value(ind));
            // correct->value(ind) = input->value(ind);
        }

        TNodeShPtr<double> out  = dense1.call(input);
        out                     = Sigmoid(out);
        out                     = dense2.call(out);
        out                     = Sigmoid(out);
        out                     = dense3.call(out);
        TNodeShPtr<double> loss = mse->call(correct, out);

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
                fout << x << " " << std::sin(x) << " ";
                input->value(0) = x;
                // input->forward();
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