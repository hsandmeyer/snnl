#pragma once
#include "connectors/connector_conv2d.h"
#include "forward_declare.h"
#include "module.h"
#include <stdexcept>

namespace snnl
{

template<class TElem>
class Conv2DModule : public Module<TElem>
{

    NodeShPtr<TElem> _kernel;

public:
    size_t _kernel_width;
    size_t _kernel_height;
    size_t _input_dim;
    size_t _output_dim;

    Conv2DModule(size_t kernel_width, size_t kernel_height, size_t input_dim, size_t output_dim,
                 std::string weight_initialization = "he_normal")
        : _kernel_width(kernel_width)
        , _kernel_height(kernel_height)
        , _input_dim(input_dim)
        , _output_dim(output_dim)
    {
        _kernel = this->addWeight({_kernel_width, _kernel_height, _input_dim, _output_dim});

        init_weight(_kernel->values(), weight_initialization,
                    _input_dim * _kernel_width * _kernel_height, _output_dim);
    }

    virtual NodeShPtr<TElem> callHandler(std::vector<NodeShPtr<TElem>> inputs) override
    {
        if(inputs.size() != 1) {
            throw std::invalid_argument("Maximal one node per call for conv2d module");
        }

        return Conv2D(_kernel, inputs.at(0));
    }

    NodeShPtr<TElem>& Kernel() { return _kernel; }
};

template<typename TElem>
using Conv2DModuleShPtr = std::shared_ptr<Conv2DModule<TElem>>;

} // namespace snnl
