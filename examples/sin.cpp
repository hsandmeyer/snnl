#include "common_modules.h"
#include "forward_declare.h"
#include "modules/module_dense.h"
#include "node.h"
#include <fstream>

using namespace snnl;

struct SinModel : public TModule<float> {
    TDenseModuleShPtr<float> dense1;
    TDenseModuleShPtr<float> dense2;
    TDenseModuleShPtr<float> dense3;

    SinModel()
    {
        dense1 = addModule<TDenseModule>(1, 64);
        dense2 = addModule<TDenseModule>(64, 16);
        dense3 = addModule<TDenseModule>(16, 1);

        dense1->W()->values().uniform(-1, 1);
        dense1->B()->values().uniform(-1, 1);
        dense2->W()->values().uniform(-1, 1);
        dense2->B()->values().uniform(-1, 1);
        dense3->W()->values().uniform(-1, 1);
        dense3->B()->values().uniform(-1, 1);
    }

    virtual TNodeShPtr<float>
    callHandler(std::vector<TNodeShPtr<float>> input) override
    {
        TNodeShPtr<float> out = dense1->call(input);
        out                   = Sigmoid(out);
        out                   = dense2->call(out);
        out                   = Sigmoid(out);
        out                   = dense3->call(out);
        return out;
    }
};

int main()
{

    TNodeShPtr<float> input = TNode<float>::create({4, 1});

    TNodeShPtr<float> correct = TNode<float>::create({4, 1});
    correct->setConstant(true);

    TConnectorShPtr<float> mse =
        TDenseConnector<float>::create<TMSEConnector>();

    SinModel model;

    for (size_t step = 0; step < 100000; step++) {
        input->values().uniform(-M_PI, M_PI);

        for (size_t ind = 0; ind < correct->shapeFlattened(-1); ind++) {
            correct->value(ind) = std::sin(input->value(ind));
        }

        TNodeShPtr<float> out  = model.call(input);
        TNodeShPtr<float> loss = MSE(correct, out);

        loss->zeroGrad();
        loss->computeGrad();
        // test_grad({input}, loss);

        float eps = 1e-1;

        loss->iterateWeights([&](TNode<float>& weight) {
            for (size_t ind = 0; ind < weight.shapeFlattened(-1); ind++) {
                weight.value(ind) = weight.value(ind) - eps * weight.grad(ind);
            }
        });

        if (step % 500 == 0) {

            std::cout << loss->value(0) << " ";
            for (size_t i = 0; i < correct->shapeFlattened(-1); i++) {
                std::cout << out->value(i) - correct->value(i) << " ";
            }

            std::cout << std::endl;
            std::ofstream fout("test.txt");

            input->setDims({1, 1});
            correct->setDims({1, 1});
            for (int i = 0; i < 100; i++) {
                float x = -M_PI + i * 2 * M_PI / 100.;
                fout << x << " " << std::sin(x) << " ";
                input->value(0) = x;

                out = model.call(input);
                fout << out->value(0) << std::endl;
            }
            input->setDims({4, 1});
            correct->setDims({4, 1});
        }
    }
}