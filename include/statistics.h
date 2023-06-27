#pragma once
#include "forward_declare.h"
#include "tensor.h"

namespace snnl
{

template<typename TElem>
double sparseAccuracy(const Tensor<TElem>& encodings, const Tensor<TElem>& labels)
{

    auto label_view = labels.flatten();

    auto encoding_view = encodings.viewWithNDimsOnTheRight(2);

    double accuracy = 0;
    for(size_t i = 0; i < labels.shape(-1); i++) {
        TElem  max     = encoding_view(i, 0);
        size_t encoded = 0;
        for(size_t j = 1; j < encoding_view.shape(-1); j++) {
            if(encoding_view(i, j) > max) {
                encoded = j;
                max     = encoding_view(i, j);
            }
        }
        if(encoded == size_t(label_view(i))) {
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