#include "layer.h"

using namespace snnl;

int main()
{
    TLayer<float> input = {10, 1};

    TConnector<float> conn = TConnector<float>::TDenseConnector({1, 2});
    TLayer<float>     out  = conn(input);
};