
#pragma once
#include "connectors/connector_dense.h"
#include "forward_declare.h"
#include "module.h"
#include <stdexcept>

namespace snnl
{

template<class TElem>
class SimpleRNNModule : public Module<TElem>
{

    NodeShPtr<TElem> _W_h;
    NodeShPtr<TElem> _W_x;
    NodeShPtr<TElem> _B;
    NodeShPtr<TElem> _h_prev;

public:
    size_t _input_units;
    size_t _output_units;

    SimpleRNNModule(size_t input_dim, size_t output_dim)
        : _input_units(input_dim)
        , _output_units(output_dim)
    {
        _W_h = this->addWeight({_output_units, _output_units});
        _W_x = this->addWeight({_input_units, _output_units});
        _B   = this->addWeight({_output_units});

        _W_x->values().xavier(_input_units, _output_units);
        _W_h->setAllValues(0);
        _B->setAllValues(0);

        _h_prev = Node<TElem>::create({_output_units});
        _h_prev->setAllValues(0);
    }

    virtual NodeShPtr<TElem> callHandler(std::vector<NodeShPtr<TElem>> inputs) override
    {
        if(inputs.size() != 1) {
            throw std::invalid_argument("Maximal one node per call for simpleRNN module");
        }

        _h_prev->disconnect();
        auto h  = Add(Add(Dot(_h_prev, _W_h), Dot(inputs[0], _W_x)), _B);
        _h_prev = h;
        return h;
    }

    NodeShPtr<TElem>& W_h() { return _W_h; }
    NodeShPtr<TElem>& W_x() { return _W_x; }

    NodeShPtr<TElem>& B() { return _B; }
    NodeShPtr<TElem>& hPrev() { return _h_prev; }
};

template<typename TElem>
using SimpleRNNModuleShPtr = std::shared_ptr<SimpleRNNModule<TElem>>;

} // namespace snnl