#pragma once
#include "connector.h"
#include "forward_declare.h"

namespace snnl {

template <class TElem>
class SumConnector : public Connector<TElem> {
public:
    virtual ~SumConnector() {}

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for sum connector");
        }
        return Index{};
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {
        output_node->value() = 0;
        output_node->value() += std::accumulate(
            input_nodes.front()->values().begin(),
            input_nodes.front()->values().end(), static_cast<TElem>(0));
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        TElem output_grad = output_node->grad();
        for (auto& val : input_nodes.front()->gradient()) {
            val += output_grad;
        }
    }
};

template <class TElem>
NodeShPtr<TElem> Sum(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<SumConnector>(node);
}

} // namespace snnl