#include "common_modules.h"
#include "connectors/connector_cross_entropy.h"
#include "connectors/connector_softmax.h"
#include "modules/module_dense.h"
#include "node.h"
#include "optimizer.h"
#include "tensor.h"
#include <fstream>

using namespace snnl;

Tensor<double> read_mnist_images(std::string full_path)
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

        Tensor<double> out = {number_of_images, n_rows, n_cols, 1};
        uchar          image[image_size];
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

Tensor<double> read_mnist_labels(std::string full_path)
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
        Tensor<double> out{number_of_labels};
        for(int i = 0; i < number_of_labels; i++) {
            out(i) = data[i];
        }
        return out;
    }
    else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

struct MNistModel : public Module<double>
{

    size_t _image_height;
    size_t _image_width;

    std::shared_ptr<Conv2DModule<double>> conv2d_1;
    std::shared_ptr<Conv2DModule<double>> conv2d_2;
    std::shared_ptr<Conv2DModule<double>> conv2d_3;
    std::shared_ptr<Conv2DModule<double>> conv2d_4;
    DenseModuleShPtr<double>              dense_1;

    MNistModel(size_t image_height, size_t image_width)
        : _image_height(image_height)
        , _image_width(image_width)
    {
        conv2d_1 = this->addModule<Conv2DModule>(3, 3, 1, 16);
        conv2d_2 = this->addModule<Conv2DModule>(3, 3, 16, 32);
        conv2d_3 = this->addModule<Conv2DModule>(3, 3, 32, 64);
        dense_1  = addModule<DenseModule>(64 * image_height / 4 * image_width / 4, 10);
    }

    virtual NodeShPtr<double> callHandler(std::vector<NodeShPtr<double>> inputs) override
    {

        auto& images = inputs.at(0);

        // std::cout << "images " << images->shape() << std::endl;

        auto layer1 = conv2d_1->call(images);
        layer1      = ReLU(layer1);

        // std::cout << "layer1 " << layer1->shape() << std::endl;

        auto layer2 = AveragePooling(layer1, 2, 2);
        layer2      = conv2d_2->call(layer2);
        layer2      = ReLU(layer2);

        // std::cout << "layer2 " << layer2->shape() << std::endl;

        auto layer3 = AveragePooling(layer2, 2, 2);
        layer3      = conv2d_3->call(layer3);
        layer3      = ReLU(layer3);
        // std::cout << "layer3 " << layer3->shape() << std::endl;

        layer3 = Flatten(layer3);
        // std::cout << "layer3 flattened " << layer3->shape() << std::endl;

        auto logits = dense_1->call(layer3);
        // std::cout << "logits " << logits->shape() << std::endl;

        auto encoding = SoftMax(logits);
        // std::cout << "encoding " << encoding->shape() << std::endl;

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

    size_t            batch_size = 4;
    NodeShPtr<double> input_images =
        Node<double>::create({batch_size, image_width, image_height, 1});
    NodeShPtr<double> input_labels = Node<double>::create({batch_size});

    MNistModel model(image_height, image_width);

    std::random_device dev;
    std::mt19937       rng(dev());

    std::uniform_int_distribution<std::mt19937::result_type> chooser_train(
        0, train_images.shape(0) - 1);
    std::uniform_int_distribution<std::mt19937::result_type> chooser_test(0,
                                                                          test_images.shape(0) - 1);

    SGDOptimizer<double> optimizer(1e-1);

    for(size_t step = 0; step < 100000; step++) {

        for(size_t i = 0; i < batch_size; i++) {
            auto random_index = chooser_train(rng);

            input_images->values().viewAs(i, ellipsis()) =
                train_images.viewAs(random_index, ellipsis());

            input_labels->value(i) = train_labels(random_index);
        }

        NodeShPtr<double> encoding = model.call(input_images, input_labels);

        auto loss = SparseCategoricalCrosseEntropy(encoding, input_labels);

        loss->computeGrad();

        optimizer.optimizeStep(loss);

        if(step % 50 == 0) {
            std::cout << "Loss = " << loss->value(0) << std::endl;

            NodeShPtr<double> test_image = Node<double>::create({1, image_width, image_height, 1});
            NodeShPtr<double> test_label = Node<double>::create({1, 1});

            auto test_index = chooser_test(rng);

            test_image->values().viewAs(0, ellipsis()) = test_images.viewAs(test_index, ellipsis());
            test_label->value(0, 0)                    = test_labels(test_index);

            auto test_encoding = model.call(test_image);

            auto encoding_view = test_encoding->values().viewAs(0, all());

            double max       = 0;
            size_t max_index = 0;
            for(size_t i = 0; i < encoding_view.shape(0); i++) {
                if(max < encoding_view(i)) {
                    max_index = i;
                    max       = encoding_view(i);
                }
            }

            std::cout << "Correct label\t= " << test_label->value(0)
                      << "\nChosen \t\t= " << max_index << " with "
                      << test_encoding->value(max_index)
                      << "\nEncoding = " << test_encoding->values() << std::endl;
            if(size_t(test_label->value(0)) != max_index) {
                test_image->values().saveToBMP(std::to_string(max_index) + ".bmp", 0, 1);
            }

            model.saveToFile("mnist.snnl");
        }
    }
}