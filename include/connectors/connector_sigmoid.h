#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class TSigmoidConnector : public TConnector<TElem> {
public:
    virtual ~TSigmoidConnector()
    {
        std::cout << "Destroying Sigmoid connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Sigmoid connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {
        // std::cout << "FORWARD on Sigmoid layer" << std::endl;
        TNodeShPtr<TElem> input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) =
                static_cast<TElem>(1) /
                (static_cast<TElem>(1) + std::exp(-input_node->value(ind)));
        }
    }

    void backwardHandler(const TNode<TElem>*             output_node,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
    {
        // std::cout << "BACKWARD on sigmoid layer" << std::endl;
        TNodeShPtr<TElem> input_node = input_nodes.front();

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
TNodeShPtr<TElem> Sigmoid(const TNodeShPtr<TElem>& node)
{
    return TConnector<TElem>::template apply<TSigmoidConnector>(node);
}

} // namespace snnl