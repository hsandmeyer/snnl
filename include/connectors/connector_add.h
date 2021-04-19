#pragma once
#include "connector.h"

namespace snnl {
template <class TElem>
class TAddConnector : public TConnector<TElem> {
public:
    virtual ~TAddConnector() {}

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        TIndex out_shape = input_nodes.front()->shape();

        for (auto node_ptr : input_nodes) {
            if (node_ptr->shape() != out_shape) {
                throw("Add connector can only work on tensors of exakt same "
                      "shape");
            }
        }
        return out_shape;
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {
        output_node->setAllValues(0);
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                output_node->value(ind) += input_node_ptr->value(ind);
            }
        }
    }

    void backwardHandler(const TNode<TElem>*             output_node,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
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
TNodeShPtr<TElem> Add(const TNodeShPtr<TElem>& node, const TArgs&... args)
{
    return TConnector<TElem>::template apply<TAddConnector>(node, args...);
}

} // namespace snnl