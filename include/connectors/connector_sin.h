
#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class SinConnector : public Connector<TElem> {
public:
    virtual ~SinConnector() {}

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Sin connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) = std::sin(input_node->value(ind));
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();

        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            TElem input_value = input_node->value(ind);
            TElem output_grad = output_node->grad(ind);

            input_node->grad(ind) += std::cos(input_value) * output_grad;
        }
    }
};

template <class TElem>
NodeShPtr<TElem> Sin(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<SinConnector>(node);
}

template <class TElem>
class TCosConnector : public Connector<TElem> {
public:
    virtual ~TCosConnector() {}

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Cos connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) = std::cos(input_node->value(ind));
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        NodeShPtr<TElem> input_node = input_nodes.front();

        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            TElem input_value = input_node->value(ind);
            TElem output_grad = output_node->grad(ind);

            input_node->grad(ind) -= std::sin(input_value) * output_grad;
        }
    }
};

template <class TElem>
NodeShPtr<TElem> Cos(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<TCosConnector>(node);
}

} // namespace snnl