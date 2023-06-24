
#pragma once
#include "connector.h"
#include <stdexcept>

namespace snnl
{

template<class TElem>
class ConcatenateConnector : public Connector<TElem>
{

    /* Fully analog implementation of np.dot*/

    friend class Connector<TElem>;
    long _axis = 0;

    void dimChecks(const std::vector<NodeShPtr<TElem>>& input_nodes) const
    {

        auto shape = input_nodes.at(0)->shape();
        for(size_t i = 0; i < input_nodes.size(); i++) {
            if(input_nodes.at(i)->NDims() != shape.NDims()) {
                throw std::invalid_argument("Concatenate: Dimension mismatch");
            }
            for(long dim_ind = 0; dim_ind < static_cast<long>(shape.NDims()); dim_ind++) {
                if(dim_ind != _axis && shape[dim_ind] != input_nodes.at(i)->shape(dim_ind)) {
                    throw std::invalid_argument(
                        "Concatenate: shape mismatch:" + input_nodes.at(0)->shape() + " vs " +
                        input_nodes.at(1)->shape());
                }
            }
        }
    }

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        dimChecks(input_nodes);

        auto shape = input_nodes.at(0)->shape();

        shape[_axis] = 0;

        for(auto& node : input_nodes) {
            shape[_axis] += node->shape(_axis);
        }

        return shape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {
        auto out_view = output_node->values().viewFromIndices({_axis, _axis + 1});

        size_t offset = 0;
        for(auto& node : input_nodes) {

            auto node_view = node->values().viewFromIndices({_axis, _axis + 1});

            for(size_t i = 0; i < node_view.shape(0); i++) {
                for(size_t j = 0; j < node_view.shape(1); j++) {
                    for(size_t k = 0; k < node_view.shape(2); k++) {
                        out_view(i, j + offset, k) = node_view(i, j, k);
                    }
                }
            }

            offset += node->shape(_axis);
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        auto out_grad_view = output_node->gradient().viewFromIndices({_axis, _axis + 1});

        size_t offset = 0;
        for(auto& node : input_nodes) {

            auto node_grad_view = node->gradient().viewFromIndices({_axis, _axis + 1});

            for(size_t i = 0; i < node_grad_view.shape(0); i++) {
                for(size_t j = 0; j < node_grad_view.shape(1); j++) {
                    for(size_t k = 0; k < node_grad_view.shape(2); k++) {
                        node_grad_view(i, j, k) += out_grad_view(i, j + offset, k);
                    }
                }
            }

            offset += node->shape(_axis);
        }
    }

public:
    ConcatenateConnector(long axis = 0)
        : _axis(axis)
    {
    }

    virtual ~ConcatenateConnector() {}
};

template<class TElem>
NodeShPtr<TElem> Concatenate(const NodeShPtr<TElem>& a, const NodeShPtr<TElem>& b, long axis = 0)
{
    auto conn = Connector<TElem>::template create<ConcatenateConnector>(axis);
    return conn->call(a, b);
}

} // namespace snnl