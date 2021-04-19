#pragma once
#include "connector.h"
#include "forward_declare.h"

namespace snnl {

template <class TElem>
class TSumConnector : public TConnector<TElem> {
public:
    virtual ~TSumConnector() {}

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for sum connector");
        }
        return TIndex{1};
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {
        output_node->value(0) = 0;
        output_node->value(0) += std::accumulate(
            input_nodes.front()->values().begin(),
            input_nodes.front()->values().end(), static_cast<TElem>(0));
    }

    void backwardHandler(const TNode<TElem>*             output_node,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
    {
        TElem output_grad = output_node->grad(0);
        for (auto& val : input_nodes.front()->gradient()) {
            val += output_grad;
        }
    }
};

template <class TElem>
TNodeShPtr<TElem> Sum(const TNodeShPtr<TElem>& node)
{
    return TConnector<TElem>::template apply<TSumConnector>(node);
}

} // namespace snnl