#pragma once
#include "connector.h"
#include <limits>

namespace snnl
{

template<class TElem>
class SparseCategoricalCrossEntropyConnector : public Connector<TElem>
{
public:
    virtual ~SparseCategoricalCrossEntropyConnector() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if(input_nodes.size() != 2) {
            throw std::invalid_argument("Exactly two nodes needed for SparseCrossEntropyConnector");
        }

        if(input_nodes[0]->NDims() != input_nodes[1]->NDims() + 1) {
            throw std::invalid_argument("SparseCrossEntropyConnector: Dimensions of the input "
                                        "nodes do not match correctly");
        }

        if(input_nodes[0]->shape(-2) != input_nodes[1]->shape(-1)) {
            throw std::invalid_argument("SparseCrossEntropyConnector: Dimensions of the input "
                                        "nodes do not match correctly. " +
                                        input_nodes[0]->shape() + " " + input_nodes[1]->shape());
        }
        return Index{1};
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {

        auto distributions = input_nodes[0]->values().viewWithNDimsOnTheRight(2);
        auto labels        = input_nodes[1]->values().viewWithNDimsOnTheRight(1);

        for(size_t i = 0; i < labels.shape(-1); i++) {
            output_node->value() -=
                log(distributions(i, labels(i)) + std::numeric_limits<TElem>::min());
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        TElem out_grad           = output_node->grad(0);
        auto  distributions_grad = input_nodes[0]->gradient().viewWithNDimsOnTheRight(2);
        auto  distributions_val  = input_nodes[0]->values().viewWithNDimsOnTheRight(2);
        auto  labels_val         = input_nodes[1]->values().viewWithNDimsOnTheRight(1);
        auto  labels_grad        = input_nodes[1]->gradient().viewWithNDimsOnTheRight(1);

        // No gradient for labels
        labels_grad.setAllValues(0);

        for(size_t i = 0; i < labels_val.shape(-1); i++) {
            distributions_grad(i, labels_val(i)) +=
                -1 / (distributions_val(i, labels_val(i)) + std::numeric_limits<TElem>::min()) *
                out_grad;
        }
    }
};

template<class TElem>
NodeShPtr<TElem> SparseCategoricalCrosseEntropy(const NodeShPtr<TElem>& model_output,
                                                const NodeShPtr<TElem>& correct)
{
    return Connector<TElem>::template apply<SparseCategoricalCrossEntropyConnector>(model_output,
                                                                                    correct);
}

} // namespace snnl