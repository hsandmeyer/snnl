#include "batch_generator.h"
#include "common_modules.h"
#include "connectors/connector_cross_entropy.h"
#include "connectors/connector_softmax.h"
#include "forward_declare.h"
#include "modules/module_dense.h"
#include "node.h"
#include "optimizer.h"
#include "statistics.h"
#include "tensor.h"
#include <cmath>
#include <fstream>
#include <stdexcept>

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

    size_t _image_height;
    size_t _image_width;

    std::shared_ptr<Conv2DModule<float>> conv2d_1;
    std::shared_ptr<Conv2DModule<float>> conv2d_2;
    std::shared_ptr<Conv2DModule<float>> conv2d_3;
    std::shared_ptr<Conv2DModule<float>> conv2d_4;
    DenseModuleShPtr<float>              dense_1;

    MNistModel(size_t image_height, size_t image_width)
        : _image_height(image_height)
        , _image_width(image_width)
    {
        conv2d_1 = this->addModule<Conv2DModule>(3, 3, 1, 16, "he_normal");
        conv2d_2 = this->addModule<Conv2DModule>(3, 3, 16, 32, "he_normal");
        conv2d_3 = this->addModule<Conv2DModule>(3, 3, 32, 64, "he_normal");
        dense_1  = addModule<DenseModule>(64 * image_height / 4 * image_width / 4, 10, "xavier");
    }

    virtual NodeShPtr<float> callHandler(std::vector<NodeShPtr<float>> inputs) override
    {

        auto& images = inputs.at(0);

        auto layer1 = conv2d_1->call(images);
        layer1      = ReLU(layer1);
        auto layer2 = AveragePooling(layer1, 2, 2);
        layer2      = conv2d_2->call(layer2);
        layer2      = ReLU(layer2);

        auto layer3 = AveragePooling(layer2, 2, 2);
        layer3      = conv2d_3->call(layer3);
        layer3      = ReLU(layer3);

        layer3 = Flatten(layer3);

        auto logits = dense_1->call(layer3);

        auto encoding = SoftMax(logits);

        return encoding;
    }
};

int main()
{
    // You need to extract the mnist data and put them into the root folder of the repo
    // (Assuming you work at snnl/build)
    auto train_images = read_mnist_images("../train-images.idx3-ubyte");
    train_images /= 255.f;
    train_images.saveToBMP("train.bmp", 0, 1);

    auto train_labels = read_mnist_labels("../train-labels.idx1-ubyte");

    auto test_images = read_mnist_images("../t10k-images.idx3-ubyte");
    test_images /= 255.f;
    test_images.saveToBMP("test.bmp", 0, 1);

    auto test_labels = read_mnist_labels("../t10k-labels.idx1-ubyte");

    size_t image_height = train_images.shape(1);
    size_t image_width  = train_images.shape(2);

    size_t batch_size = 32;
    size_t epoch_size = 2048;

    MNistModel model(image_height, image_width);

    AdamOptimizer<float> optimizer;

    BatchGenerator train_generator(train_images, train_labels);
    BatchGenerator test_generator(test_images, test_labels);
    test_generator.mute();

    float loss_sum = 0;

    train_generator.setEpochSize(epoch_size);
    train_generator.setEpochCallBack([&](size_t epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        std::cout << "Mean loss = " << loss_sum / epoch_size << std::endl;

        auto [single_image, single_label] = test_generator.generateBatch(1);

        auto   single_encoding = model.call(single_image);
        size_t predicted       = single_encoding->values().argMax()(0);

        std::cout << "Random example:" << std::endl
                  << "Correct label = " << single_label->value(0) << std::endl
                  << "Chosen        = " << predicted << " with "
                  << single_encoding->value(predicted) << std::endl
                  << "Encoding      = " << single_encoding->values() << std::endl;

        if(epoch >= 3 && size_t(single_label->value(0)) != predicted) {
            // Save wrong classifications
            single_image->values().saveToBMP(std::to_string(predicted) + ".bmp", 0, 1);
        }

        auto   test_encodings = model.call(test_images);
        double test_accuracy  = sparseAccuracy(test_encodings, test_labels);
        std::cout << "Test accuracy = " << test_accuracy << std::endl;

        // auto   train_encodings = model.call(train_images);
        // double train_accuracy  = sparseAccuracy(train_encodings, train_labels);
        // std::cout << "Training accuracy = " << train_accuracy << std::endl;

        model.saveToFile("mnist.snnl");
        loss_sum = 0;
    });

    for(size_t step = 0; step < 100000; step++) {

        auto [input_images, input_labels] = train_generator.generateBatch(batch_size);

        NodeShPtr<float> predicted_encodings = model.call(input_images, input_labels);

        auto loss = SparseCategoricalCrosseEntropy(predicted_encodings, input_labels);

        loss_sum += loss->value();

        loss->computeGrad();

        optimizer.optimizeStep(loss);
    }
}
