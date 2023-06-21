#include "common_modules.h"
#include "modules/module_dense.h"
#include "node.h"
#include "optimizer.h"
#include "tensor.h"
#include <fstream>

using namespace snnl;

Tensor<float> read_mnist_images(std::string full_path)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    int number_of_images, image_size;

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) {
            throw std::runtime_error("Invalid MNIST image file!");
        }

        file.read((char*)&number_of_images, sizeof(number_of_images)),
            number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        Tensor<float> out = {number_of_images, n_rows, n_cols, 1};
        uchar         image[image_size];
        for(int i = 0; i < number_of_images; i++) {
            file.read((char*)image, image_size);
            for(size_t row = 0; row < size_t(n_rows); row++) {
                for(size_t col = 0; col < size_t(n_cols); col++) {
                    out(i, row, col, 0) = image[col + n_cols * row];
                }
            }
        }
        return out;
    }
    else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

Tensor<float> read_mnist_labels(std::string full_path)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;
    int                   number_of_labels;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) {
            throw std::runtime_error("Invalid MNIST label file!");
        }

        file.read((char*)&number_of_labels, sizeof(number_of_labels)),
            number_of_labels = reverseInt(number_of_labels);

        std::vector<uchar> data(number_of_labels);
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&data[i], 1);
        }
        Tensor<float> out{number_of_labels};
        for(int i = 0; i < number_of_labels; i++) {
            out(i) = data[i];
        }
        return out;
    }
    else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

struct MNistModel : public Module<float>
{
    DenseModuleShPtr<float> dense1;
    DenseModuleShPtr<float> dense2;
    DenseModuleShPtr<float> dense3;

    MNistModel()
    {
        dense1 = addModule<DenseModule>(1, 64);
        dense2 = addModule<DenseModule>(64, 16);
        dense3 = addModule<DenseModule>(16, 1);
    }

    virtual NodeShPtr<float> callHandler(std::vector<NodeShPtr<float>> input) override
    {
        NodeShPtr<float> out = dense1->call(input);
        out                  = Sigmoid(out);
        out                  = dense2->call(out);
        out                  = Sigmoid(out);
        out                  = dense3->call(out);
        return out;
    }
};

int main()
{
    auto images = read_mnist_images("../train-images.idx3-ubyte");
    images.saveToBMP("test.bmp");

    auto labels = read_mnist_labels("../train-labels.idx1-ubyte");
    std::cout << labels(2) << std::endl;

    /*
    size_t           batch_size = 4;
    NodeShPtr<float> input      = Node<float>::create({batch_size, 1});
    MNistModel       model;

    SGDOptimizer<float> optimizer(1e-1);

    for(size_t step = 0; step < 100000; step++) {
        input->values().uniform(-M_PI, M_PI);

        auto correct = Sin(input);
        correct->disconnect();

        NodeShPtr<float> out  = model.call(input);
        NodeShPtr<float> loss = MSE(correct, out);

        loss->computeGrad();

        optimizer.optimizeStep(loss);

        if(step % 500 == 0) {
            // std::cout << model.dense1->B()->values();

            std::cout << "Loss = " << loss->value(0) << std::endl;
            std::cout << "Diff =\n" << out->values() - correct->values() << " " << std::endl;

            std::ofstream fout("test.txt");

            input->setDims({100, 1});
            input->values().arangeAlongAxis(0, -M_PI, M_PI);
            out = model.call(input);

            correct = Sin(input);
            correct->disconnect();

            for(size_t ind = 0; ind < input->values().shapeFlattened(-1); ++ind) {
                fout << input->value(ind, 0) << " " << correct->value(ind, 0) << " "
                     << out->value(ind, 0) << std::endl;
            }

            input->setDims({batch_size, 1});
            correct->setDims({batch_size, 1});
        }
    }
    */
}