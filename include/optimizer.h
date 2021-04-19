#pragma once
#include "forward_declare.h"
#include "module.h"

namespace snnl {

template <typename TElem>
class TOptimizer {

    int _num_states_per_weight;

    std::map<TNodeShPtr<TElem>, std::vector<TTensor<TElem>>> _states;

    virtual void optimizeGrad(TNode<TElem>&                weight,
                              std::vector<TTensor<TElem>>& states) = 0;

public:
    TOptimizer(int num_states_per_weight)
        : _num_states_per_weight(num_states_per_weight)
    {
    }

    void optimizeStep(TNodeShPtr<TElem> loss)
    {
        loss->iterateWeights([&](TNode<TElem>& weight) {
            TNodeShPtr<TElem> weight_ptr = weight.getPtr();

            auto& states = _states[weight_ptr];

            if (states.empty()) {
                states.resize(_num_states_per_weight);
                for (auto& t : states) {
                    t.setDims(weight.shape());
                    t.setAllValues(0);
                }
            }

            optimizeGrad(weight, states);
        });
    }
};

template <typename TElem>
class TSGDOptimizer : public TOptimizer<TElem> {

    TElem _learning_rate;

    virtual void optimizeGrad(TNode<TElem>& weight,
                              std::vector<TTensor<TElem>>&) override
    {
        for (size_t ind = 0; ind < weight.shapeFlattened(-1); ind++) {
            weight.value(ind) =
                weight.value(ind) - _learning_rate * weight.grad(ind);
        }
    }

public:
    TSGDOptimizer(TElem learning_rate)
        : TOptimizer<TElem>::TOptimizer(0), _learning_rate(learning_rate)
    {
    }
};

} // namespace snnl