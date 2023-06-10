#pragma once
#include "connector.h"

namespace snnl
{

template<class TElem>
class MSEConnector : public Connector<TElem>
{
public:
    virtual ~MSEConnector() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if(input_nodes.size() != 2) {
            throw std::invalid_argument("Exact two nodes needed for mse connector");
        }
        if(input_nodes[0]->shape() != input_nodes[1]->shape()) {
            throw std::invalid_argument("Input nodes for MSE layer need to have exact same size");
        }
        return Index{1};
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {

        auto input_0 = input_nodes[0]->values();
        auto input_1 = input_nodes[1]->values();

        output_node->value(0) = 0;
        for(size_t ind = 0; ind < input_0.shapeFlattened(-1); ind++) {
            TElem diff = (input_0(ind) - input_1(ind));
            output_node->value(0) += diff * diff;
        }
        output_node->value(0) /= input_0.shapeFlattened(-1);
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        auto  input_0  = input_nodes[0];
        auto  input_1  = input_nodes[1];
        TElem size     = input_0->shapeFlattened(-1);
        TElem out_grad = output_node->grad(0) / static_cast<TElem>(size);

        for(size_t ind = 0; ind < size; ind++) {
            input_0->grad(ind) += 2. * (input_0->value(ind) - input_1->value(ind)) * out_grad;
            input_1->grad(ind) += 2. * (input_1->value(ind) - input_0->value(ind)) * out_grad;
        }
    }
};

template<class TElem>
NodeShPtr<TElem> MSE(const NodeShPtr<TElem>& model_output, const NodeShPtr<TElem>& correct)
{
    return Connector<TElem>::template apply<MSEConnector>(model_output, correct);
}

} // namespace snnl