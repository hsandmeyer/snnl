#pragma once
#include "connector.h"
#include <stdexcept>

namespace snnl
{
template<class TElem, template<class> class Functor>
class ElementWiseCombination : public Connector<TElem>
{
public:
    virtual ~ElementWiseCombination() {}

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {

        if(input_nodes.size() != 2) {
            throw std::invalid_argument("Exactly two nodes needed for element wise combination");
        }
        auto& a = input_nodes.front();
        auto& b = input_nodes.back();

        long NDims_smaller = std::min(a->NDims(), b->NDims());
        for(long i = 1; i <= NDims_smaller; i++) {
            if(a->shape(-i) != b->shape(-i)) {
                throw std::invalid_argument("Operatr *=: Dimension mismatch. Shape at " +
                                            std::to_string(-i) +
                                            " is unequal: " + std::to_string(a->shape(-i)) +
                                            " vs " + std::to_string(b->shape(-i)));
            }
        }
        if(a->NDims() >= b->NDims()) {
            return a->shape();
        }
        else {
            return b->shape();
        }
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {

        auto& a = input_nodes.front()->values();
        auto& b = input_nodes.back()->values();

        if(a.NDims() > b.NDims()) {
            Tensor<TElem> out_view = output_node->values().viewFromIndices({-b.NDims()});
            Tensor<TElem> a_view   = a.viewFromIndices({-b.NDims()});
            Tensor<TElem> b_view   = b.viewWithNDimsOnTheLeft(1);

            for(size_t i = 0; i < a_view.shape(0); i++) {
                for(size_t j = 0; j < a_view.shape(1); j++) {
                    out_view(i, j) = Functor<TElem>::forward(a_view(i, j), b_view(j));
                }
            }
        }
        else {
            Tensor<TElem> out_view = output_node->values().viewFromIndices({-a.NDims()});
            Tensor<TElem> b_view   = b.viewFromIndices({-a.NDims()});
            Tensor<TElem> a_view   = a.viewWithNDimsOnTheLeft(1);

            for(size_t i = 0; i < b_view.shape(0); i++) {
                for(size_t j = 0; j < b_view.shape(1); j++) {
                    out_view(i, j) = Functor<TElem>::forward(a_view(j), b_view(i, j));
                }
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        auto& val_a = input_nodes.front()->values();
        auto& val_b = input_nodes.back()->values();

        auto& grad_a = input_nodes.front()->gradient();
        auto& grad_b = input_nodes.back()->gradient();

        if(val_a.NDims() > val_b.NDims()) {
            Tensor<TElem> grad_out_view =
                output_node->gradient().viewFromIndices({-val_b.NDims()});

            Tensor<TElem> a_val_view = val_a.viewFromIndices({-val_b.NDims()});
            Tensor<TElem> b_val_view = val_b.viewWithNDimsOnTheLeft(1);

            Tensor<TElem> a_grad_view = grad_a.viewFromIndices({-grad_b.NDims()});
            Tensor<TElem> b_grad_view = grad_b.viewWithNDimsOnTheLeft(1);

            for(size_t i = 0; i < a_val_view.shape(0); i++) {
                for(size_t j = 0; j < a_val_view.shape(1); j++) {
                    auto [deriv_a, deriv_b] =
                        Functor<TElem>::backward(a_val_view(i, j), b_val_view(j));

                    a_grad_view(i, j) += deriv_a * grad_out_view(i, j);
                    b_grad_view(j) += deriv_b * grad_out_view(i, j);
                }
            }
        }
        else {
            Tensor<TElem> grad_out_view =
                output_node->gradient().viewFromIndices({-val_a.NDims()});

            Tensor<TElem> b_val_view = val_b.viewFromIndices({-val_a.NDims()});
            Tensor<TElem> a_val_view = val_a.viewWithNDimsOnTheLeft(1);

            Tensor<TElem> b_grad_view = grad_b.viewFromIndices({-grad_a.NDims()});
            Tensor<TElem> a_grad_view = grad_a.viewWithNDimsOnTheLeft(1);

            for(size_t i = 0; i < b_val_view.shape(0); i++) {
                for(size_t j = 0; j < b_val_view.shape(1); j++) {
                    auto [deriv_a, deriv_b] =
                        Functor<TElem>::backward(a_val_view(j), b_val_view(i, j));

                    a_grad_view(j) += deriv_a * grad_out_view(i, j);
                    b_grad_view(i, j) += deriv_b * grad_out_view(i, j);
                }
            }
        }
    }
};
} // namespace snnl