#include "common_modules.h"
#include "forward_declare.h"
#include "modules/module_dense.h"
#include "node.h"
#include "optimizer.h"
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
    size_t            batch_size = 4;
    TNodeShPtr<float> input      = TNode<float>::create({batch_size, 1});
    SinModel          model;

    TSGDOptimizer<float> optimizer(1e-1);

    for (size_t step = 0; step < 100000; step++) {
        input->values().uniform(-M_PI, M_PI);

        auto correct = Sin(input);
        correct->disconnect();

        TNodeShPtr<float> out  = model.call(input);
        TNodeShPtr<float> loss = MSE(correct, out);

        loss->computeGrad();

        optimizer.optimizeStep(loss);

        if (step % 500 == 0) {
            // std::cout << model.dense1->B()->values();

            std::cout << "Loss = " << loss->value(0) << std::endl;
            std::cout << "Diff =\n"
                      << out->values() - correct->values() << " " << std::endl;

            std::ofstream fout("test.txt");

            input->setDims({100, 1});
            input->values().arangeAlongAxis(0, -M_PI, M_PI);
            out = model.call(input);

            correct = Sin(input);
            correct->disconnect();

            for (size_t ind = 0; ind < input->values().shapeFlattened(-1);
                 ++ind) {
                fout << input->value(ind, 0) << " " << correct->value(ind, 0)
                     << " " << out->value(ind, 0) << std::endl;
            }

            input->setDims({batch_size, 1});
            correct->setDims({batch_size, 1});
        }
    }
}