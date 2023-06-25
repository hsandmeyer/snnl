#pragma once

#include "connector.h"
namespace snnl
{

template<class TElem, template<class> class Functor>
class ElementWiseConnector : public Connector<TElem>
{
public:
    virtual ~ElementWiseConnector() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if(input_nodes.size() > 1) {
            throw std::invalid_argument("Maximal one node per call for element wise connectors");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {
        auto input_vals  = input_nodes.front()->values().flatten();
        auto output_vals = output_node->values().flatten();

        for(size_t ind = 0; ind < output_vals.size(); ind++) {
            output_vals(ind) = Functor<TElem>::forward(input_vals(ind));
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        auto input_vals  = input_nodes.front()->values().flatten();
        auto input_grad  = input_nodes.front()->gradient().flatten();
        auto output_grad = output_node->gradient().flatten();

        for(size_t ind = 0; ind < output_grad.size(); ind++) {
            TElem input_value     = input_vals(ind);
            TElem output_gradient = output_grad(ind);

            input_grad(ind) += Functor<TElem>::backward(input_value) * output_gradient;
        }
    }
};
} // namespace snnl