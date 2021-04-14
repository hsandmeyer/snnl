#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class TSumConnector : public TConnector<TElem> {
public:
    virtual ~TSumConnector()
    {
        std::cout << "Destroying Sum connector" << std::endl;
    }

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
                        const std::vector<TNodeShPtr<TElem>>&,
                        TNode<TElem>* output_node) override
    {
        // std::cout << "FORWARD on Sum layer" << std::endl;

        output_node->value(0) = 0;
        output_node->value(0) += std::accumulate(
            input_nodes.front()->values().begin(),
            input_nodes.front()->values().end(), static_cast<TElem>(0));
    }

    void backwardHandler(const TNode<TElem>* output_node,
                         std::vector<TNodeShPtr<TElem>>&,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
    {
        // std::cout << "BACKWARD on sum layer" << std::endl;
        TElem output_grad = output_node->grad(0);
        for (auto& val : input_nodes.front()->gradient()) {
            val += output_grad;
        }
    }
};

} // namespace snnl