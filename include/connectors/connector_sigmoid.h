#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class SigmoidConnector : public Connector<TElem> {
public:
    virtual ~SigmoidConnector() {}

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Sigmoid connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();

        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) =
                static_cast<TElem>(1) /
                (static_cast<TElem>(1) + std::exp(-input_node->value(ind)));
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();

        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            TElem input_value = input_node->value(ind);
            TElem output_grad = output_node->grad(ind);
            TElem tmp         = std::exp(-input_value) + 1;
            input_node->grad(ind) +=
                std::exp(-input_value) / (tmp * tmp) * output_grad;
        }
    }
};

template <class TElem>
NodeShPtr<TElem> Sigmoid(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<SigmoidConnector>(node);
}

} // namespace snnl