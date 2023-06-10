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
        NodeShPtr<TElem> input_node = input_nodes.front();
        for(size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) = Functor<TElem>::forward(input_node->value(ind));
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();

        for(size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            TElem input_value = input_node->value(ind);
            TElem output_grad = output_node->grad(ind);

            input_node->grad(ind) += Functor<TElem>::backward(input_value) * output_grad;
        }
    }
};
} // namespace snnl