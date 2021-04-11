#pragma once
#include "connector.h"

namespace snnl {
template <class TElem>
class TAddConnector : public TConnector<TElem> {
public:
    virtual ~TAddConnector()
    {
        std::cout << "Destroying Add connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const override
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

    void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                        const std::vector<TNode<TElem>*>&,
                        TNode<TElem>* output_node) override
    {
        // std::cout << "FORWARD on Add layer" << std::endl;
        output_node->setAllValues(0);
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                output_node->value(ind) += input_node_ptr->value(ind);
            }
        }
    }

    void backwardHandler(const TNode<TElem>* output_node,
                         std::vector<TNode<TElem>*>&,
                         std::vector<TNode<TElem>*>& input_nodes) override
    {
        // std::cout << "BACKWARD on add layer" << std::endl;
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                input_node_ptr->grad(ind) += output_node->grad(ind);
            }
        }
    }
};

} // namespace snnl