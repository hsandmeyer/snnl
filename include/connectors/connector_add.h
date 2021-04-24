#pragma once
#include "connector.h"

namespace snnl {
template <class TElem>
class AddConnector : public Connector<TElem> {
public:
    virtual ~AddConnector() {}

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        Index out_shape = input_nodes.front()->shape();

        for (auto node_ptr : input_nodes) {
            if (node_ptr->shape() != out_shape) {
                throw("Add connector can only work on tensors of exakt same "
                      "shape");
            }
        }
        return out_shape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {
        output_node->setAllValues(0);
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                output_node->value(ind) += input_node_ptr->value(ind);
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                input_node_ptr->grad(ind) += output_node->grad(ind);
            }
        }
    }
};

template <class TElem, class... TArgs>
NodeShPtr<TElem> Add(const NodeShPtr<TElem>& node, const TArgs&... args)
{
    return Connector<TElem>::template apply<AddConnector>(node, args...);
}

} // namespace snnl