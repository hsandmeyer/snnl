#pragma once
#include "connector.h"
#include <stdexcept>

namespace snnl
{

template<class TElem>
class SoftMaxConnector : public Connector<TElem>
{
public:
    virtual ~SoftMaxConnector() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        if(input_nodes.size() != 1) {
            throw std::invalid_argument("Exactly one input node needed SoftMax");
        }

        return input_nodes.at(0)->shape();
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {

        auto input_vals  = input_nodes[0]->values().viewWithNDimsOnTheRight(2);
        auto output_vals = output_node->values().viewWithNDimsOnTheRight(2);

        for(size_t higherDim = 0; higherDim < input_vals.shape(-2); higherDim++) {
            TElem norm = 0;
            TElem max  = 0;
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                max = std::max(input_vals(higherDim, i), max);
            }
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                norm += exp(input_vals(higherDim, i) - max);
            }
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                output_vals(higherDim, i) = exp(input_vals(higherDim, i) - max) / norm;
                /*
                if(output_vals(higherDim, i) == 0) {
                    throw(std::domain_error(
                        "0 in softmax: val = " + std::to_string(input_vals(higherDim, i)) +
                        " max = " + std::to_string(max) + " norm = " + std::to_string(norm)));
                }
                if(output_vals(higherDim, i) == 1) {
                    throw(std::domain_error(
                        "1 in softmax: val = " + std::to_string(input_vals(higherDim, i)) +
                        " max = " + std::to_string(max) + " norm = " + std::to_string(norm)));
                }
                */
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {

        auto          input_vals  = input_nodes[0]->values().viewWithNDimsOnTheRight(2);
        auto          input_grad  = input_nodes[0]->gradient().viewWithNDimsOnTheRight(2);
        auto          output_grad = output_node->gradient().viewWithNDimsOnTheRight(2);
        auto          output_vals = output_node->values().viewWithNDimsOnTheRight(2);
        Tensor<TElem> tmp(input_grad.shape());
        tmp.setAllValues(0);

        for(size_t higherDim = 0; higherDim < input_vals.shape(-2); higherDim++) {
            TElem norm = 0;
            TElem max  = 0;
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                max = std::max(input_vals(higherDim, i), max);
            }
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                norm += exp(input_vals(higherDim, i) - max);
            }
            // Be careful. There are non-diagonal elements in df_j/dz_i
            for(size_t i = 0; i < input_vals.shape(-1); i++) {
                for(size_t j = 0; j < output_grad.shape(-1); j++) {
                    if(i == j) {
                        TElem normWithoutXi = norm - exp(input_vals(higherDim, i) - max);
                        input_grad(higherDim, i) += exp(input_vals(higherDim, i) - max) *
                                                    normWithoutXi / (norm * norm) *
                                                    output_grad(higherDim, j);
                    }
                    else {
                        input_grad(higherDim, i) -= exp(input_vals(higherDim, j) - max) *
                                                    exp(input_vals(higherDim, i) - max) /
                                                    (norm * norm) * output_grad(higherDim, j);
                    }
                }
            }
        }
    }
};

template<class TElem>
NodeShPtr<TElem> SoftMax(const NodeShPtr<TElem>& input)
{
    return Connector<TElem>::template apply<SoftMaxConnector>(input);
}

} // namespace snnl