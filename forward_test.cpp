#include "layer.h"

using namespace snnl;

int main()
{
    TNode<float> input = TNode<float>::Default(32, 128);

    TLayer<float> encode = TLayer<float>::TDenseLayer(32, input);
    TNode<float>  out    = encode(input);

    TLayer<float> decode = TLayer<float>::TDenseLayer(128, out);

    out = decode(out);

    input.call();
};