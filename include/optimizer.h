#pragma once
#include "forward_declare.h"
#include "module.h"

namespace snnl
{

template<typename TElem>
class Optimizer
{

    int _num_states_per_weight;

    std::map<NodeShPtr<TElem>, std::vector<Tensor<TElem>>> _states;

    virtual void optimizeGrad(Node<TElem>& weight, std::vector<Tensor<TElem>>& states) = 0;

public:
    Optimizer(int num_states_per_weight)
        : _num_states_per_weight(num_states_per_weight)
    {
    }

    void optimizeStep(NodeShPtr<TElem> loss)
    {
        loss->iterateWeights([&](Node<TElem>& weight) {
            NodeShPtr<TElem> weight_ptr = weight.getPtr();

            auto& states = _states[weight_ptr];

            if(states.empty()) {
                states.resize(_num_states_per_weight);
                for(auto& t : states) {
                    t.setDims(weight.shape());
                    t.setAllValues(0);
                }
            }

            optimizeGrad(weight, states);
        });
    }
};

template<typename TElem>
class SGDOptimizer : public Optimizer<TElem>
{

    TElem _learning_rate;

    virtual void optimizeGrad(Node<TElem>& weight, std::vector<Tensor<TElem>>&) override
    {
        auto weight_vals = weight.values().flatten();
        auto weight_grad = weight.gradient().flatten();

        for(size_t ind = 0; ind < weight_vals.size(); ind++) {
            weight_vals(ind) = weight_vals(ind) - _learning_rate * weight_grad(ind);
        }
    }

public:
    SGDOptimizer(TElem learning_rate)
        : Optimizer<TElem>::Optimizer(0)
        , _learning_rate(learning_rate)
    {
    }
};

} // namespace snnl