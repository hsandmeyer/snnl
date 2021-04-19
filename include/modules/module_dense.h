#pragma once
#include "connectors/connector_dense.h"
#include "forward_declare.h"
#include "module.h"
#include <stdexcept>

namespace snnl {

template <class TElem>
class TDenseModule : public TModule<TElem> {

    TNodeShPtr<TElem> _W;
    TNodeShPtr<TElem> _B;

public:
    size_t _input_units;
    size_t _output_units;

    TDenseModule(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {
        _W = this->addWeight({_output_units, _input_units});
        _B = this->addWeight({_output_units});

        _W->values().xavier(_input_units, _output_units);
        _B->setAllValues(0);
    }

    virtual TNodeShPtr<TElem>
    callHandler(std::vector<TNodeShPtr<TElem>> inputs) override
    {
        if (inputs.size() != 1) {
            throw std::invalid_argument(
                "Maximal one node per call for dense module");
        }

        return Dense(_W, _B, inputs.at(0));
    }

    TNodeShPtr<TElem>& W() { return _W; }

    TNodeShPtr<TElem>& B() { return _B; }
};

template <typename TElem>
using TDenseModuleShPtr = std::shared_ptr<TDenseModule<TElem>>;

} // namespace snnl