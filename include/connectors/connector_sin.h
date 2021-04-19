
#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class TSinConnector : public TConnector<TElem> {
public:
    virtual ~TSinConnector()
    {
        std::cout << "Destroying Sin connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Sin connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {
        TNodeShPtr<TElem> input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) = std::sin(input_node->value(ind));
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

            input_node->grad(ind) += std::cos(input_value) * output_grad;
        }
    }
};

template <class TElem>
TNodeShPtr<TElem> Sin(const TNodeShPtr<TElem>& node)
{
    return TConnector<TElem>::template apply<TSinConnector>(node);
}

template <class TElem>
class TCosConnector : public TConnector<TElem> {
public:
    virtual ~TCosConnector()
    {
        std::cout << "Destroying Cos connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Cos connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {
        TNodeShPtr<TElem> input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) = std::cos(input_node->value(ind));
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

            input_node->grad(ind) -= std::sin(input_value) * output_grad;
        }
    }
};

template <class TElem>
TNodeShPtr<TElem> Cos(const TNodeShPtr<TElem>& node)
{
    return TConnector<TElem>::template apply<TCosConnector>(node);
}

} // namespace snnl