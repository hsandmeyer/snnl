#pragma once
#include "forward_declare.h"
#include "tensor.h"

namespace snnl
{

template<typename TElem>
double sparseAccuracy(const Tensor<TElem>& encodings, const Tensor<TElem>& labels)
{

    auto label_view = labels.flatten();

    auto encoding_results = encodings.viewWithNDimsOnTheRight(2).argMax();

    double accuracy = 0;
    for(size_t i = 0; i < labels.shape(-1); i++) {
        if(encoding_results(i) == size_t(label_view(i))) {
            accuracy++;
        }
    }
    return accuracy / label_view.size();
}

template<typename TElem>
double sparseAccuracy(const NodeShPtr<TElem>& encodings, const Tensor<TElem>& labels)
{
    return sparseAccuracy(encodings->values(), labels);
}

} // namespace snnl