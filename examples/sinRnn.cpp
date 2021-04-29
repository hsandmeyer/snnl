#include "common_modules.h"
#include "forward_declare.h"
#include "modules/module_dense.h"
#include "modules/module_simpleRNN.h"
#include "node.h"
#include "optimizer.h"
#include <fstream>

using namespace snnl;

struct SinRNNModel : public Module<float> {
    SimpleRNNModuleShPtr<float> rnn1;
    DenseModuleShPtr<float>     dense1;

    NodeShPtr<float> _h_stored;

    SinRNNModel()
    {
        rnn1   = addModule<SimpleRNNModule>(2, 32);
        dense1 = addModule<DenseModule>(32, 1);
    }

    virtual NodeShPtr<float>
    callHandler(std::vector<NodeShPtr<float>> input) override
    {
        NodeShPtr<float> out = rnn1->call(input);
        out                  = Sigmoid(out);
        out                  = dense1->call(out);
        return out;
    }

    void preserveState() { _h_stored = rnn1->hPrev(); }

    void loadState() { rnn1->hPrev() = _h_stored; }
};

int main()
{
    size_t           batch_size = 16;
    NodeShPtr<float> step       = Node<float>::create({batch_size, 1});
    NodeShPtr<float> x          = Node<float>::create({batch_size, 1});
    NodeShPtr<float> input      = Node<float>::create({batch_size, 2});

    NodeShPtr<float> sin = Node<float>::create({batch_size, 1});

    SinRNNModel model;

    SGDOptimizer<float> optimizer(1e-2);

    for (size_t i = 0; i < 1000000; i++) {
        step->values().uniform(0.5, 1.5);

        x = Add(x, step);

        for (size_t i = 0; i < batch_size; i++) {
            input->value(i, 0) = step->value(i, 0);
            input->value(i, 1) = sin->value(i, 0);
        }

        NodeShPtr<float> out = model.call(input);

        sin = Sin(x);
        sin->disconnect();

        NodeShPtr<float> loss = MSE(sin, out);

        loss->computeGrad();

        optimizer.optimizeStep(loss);

        if (i % 500 == 0) {
            // std::cout << model.dense1->B()->values();

            std::cout << "Loss = " << loss->value(0) << std::endl;
            std::cout << "Diff =\n"
                      << out->values() - sin->values() << " " << std::endl;

            NodeShPtr<float> x_fut = Node<float>::create({batch_size, 1});
            x_fut->values().setAllValues(0);

            std::ofstream fout("test.txt");
            model.preserveState();

            for (size_t future = 0; future < 100; future++) {
                step->values().uniform(0.5, 1.5);
                x_fut = Add(x_fut, step);
                x     = Add(x, step);

                for (size_t i = 0; i < batch_size; i++) {
                    input->value(i, 0) = step->value(i, 0);
                    input->value(i, 1) = out->value(i, 0);
                }
                sin = Sin(x);
                out = model.call(input);

                for (size_t batch = 0; batch < batch_size; batch++) {
                    fout << x_fut->value(batch, 0) << " "
                         << out->value(batch, 0) << " " << sin->value(batch, 0)
                         << " ";
                }
                fout << std::endl;
            }

            model.loadState();
        }
    }
}