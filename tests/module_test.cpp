
#include "common_modules.h"
#include "forward_declare.h"
#include "node.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace snnl;

void compareTensor(Tensor<float>& a, Tensor<float>& b)
{
    auto it = b.begin();
    for(float& val : a) {
        EXPECT_FLOAT_EQ(val, *it);
        ++it;
    }
}

struct TestModel : public Module<float>
{
    DenseModuleShPtr<float> dense1;
    DenseModuleShPtr<float> dense2;
    DenseModuleShPtr<float> dense3;

    TestModel()
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

TEST(InputOutputTest, ToByteTest)
{
    TestModel model;

    NodeShPtr<float> input = Node<float>::create({16, 1});

    for(auto weight : model.weights()) {
        weight->values().uniform();
    }

    auto array = model.toByteArray();

    TestModel model2;
    model2.fromByteArray(array);
    auto result  = model.call(input);
    auto result2 = model2.call(input);

    EXPECT_EQ(result->value(), result2->value());
}

TEST(InputOutputTest, DiskTest)
{
    TestModel model;

    NodeShPtr<float> input = Node<float>::create({16, 1});

    for(auto weight : model.weights()) {
        weight->values().uniform();
    }

    model.saveToFile("test.snnl");

    TestModel model2;

    model2.loadFromFile("test");

    auto result  = model.call(input);
    auto result2 = model2.call(input);

    EXPECT_EQ(result->value(), result2->value());
}