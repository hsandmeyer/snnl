#pragma once
#include "forward_declare.h"
#include "module.h"

namespace snnl
{

template<typename TElem>
class Optimizer
{

    int _num_states_per_weight;

    // momenta etc
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

            if(states.empty() && _num_states_per_weight > 0) {
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

template<typename TElem>
class AdamOptimizer : public Optimizer<TElem>
{

    TElem _alpha;
    TElem _beta_1;
    TElem _beta_2;

    size_t _t = 0;

    virtual void optimizeGrad(Node<TElem>& weight, std::vector<Tensor<TElem>>& states) override
    {

        _t++;

        auto theta_t = weight.values().flatten();
        auto g_t     = weight.gradient().flatten();

        auto m_t = states[0].flatten();
        auto v_t = states[1].flatten();

        // Save memory by reallocating every step. Better preallocate for performance?
        Tensor<TElem> v_t_hat(v_t.shape());
        Tensor<TElem> m_t_hat(m_t.shape());

        for(size_t ind = 0; ind < theta_t.size(); ind++) {
            m_t(ind)     = _beta_1 * m_t(ind) + (1 - _beta_1) * g_t(ind);
            v_t(ind)     = _beta_2 * v_t(ind) + (1 - _beta_2) * g_t(ind) * g_t(ind);
            m_t_hat(ind) = m_t(ind) / (1 - std::pow(_beta_1, _t));
            v_t_hat(ind) = v_t(ind) / (1 - std::pow(_beta_2, _t));
            theta_t(ind) =
                theta_t(ind) -
                _alpha * m_t_hat(ind) / (sqrt(std::max(v_t_hat(ind), TElem(0))) + TElem(1.e-8));
        }
    }

public:
    AdamOptimizer(TElem alpha = 0.001, TElem beta_1 = 0.9, TElem beta_2 = 0.999)
        : Optimizer<TElem>::Optimizer(2)
        , _alpha(alpha)
        , _beta_1(beta_1)
        , _beta_2(beta_2)
    {
    }
};

} // namespace snnl