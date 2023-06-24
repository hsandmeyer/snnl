#pragma once
#include "connector.h"

namespace snnl
{

template<class TElem>
class FlattenConnector : public Connector<TElem>
{
public:
    virtual ~FlattenConnector() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if(input_nodes.size() != 1) {
            throw std::invalid_argument("Exactly one input node needed Flatten");
        }

        Index shape = {input_nodes.at(0)->shape(0),
                       input_nodes.at(0)->NElems() / input_nodes.at(0)->shape(0)};
        return shape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {

        auto  input_vals  = input_nodes[0]->values().viewWithNDimsOnTheLeft(2);
        auto& output_vals = output_node->values();

        for(size_t batch = 0; batch < input_vals.shape(0); batch++) {
            for(size_t flat_index = 0; flat_index < input_vals.shape(-1); flat_index++) {
                output_vals(batch, flat_index) = input_vals(batch, flat_index);
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {

        auto  input_grad  = input_nodes[0]->gradient().viewWithNDimsOnTheLeft(2);
        auto& output_grad = output_node->gradient();

        for(size_t batch = 0; batch < input_grad.shape(0); batch++) {
            for(size_t flat_index = 0; flat_index < input_grad.shape(-1); flat_index++) {
                input_grad(batch, flat_index) += output_grad(batch, flat_index);
            }
        }
    }
};

template<class TElem>
NodeShPtr<TElem> Flatten(const NodeShPtr<TElem>& input)
{
    return Connector<TElem>::template apply<FlattenConnector>(input);
}

} // namespace snnl