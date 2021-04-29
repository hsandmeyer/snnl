#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class DotConnector : public Connector<TElem> {

    /* Fully analog implementation of np.dot*/

    friend class Connector<TElem>;

    void dimChecks(const std::vector<NodeShPtr<TElem>>& input_nodes) const
    {
        if (input_nodes.size() != 2) {
            throw std::invalid_argument(
                "Need exactly two inputs for matrix multiplicatoin");
        }

        auto& a = input_nodes.at(0);
        auto& b = input_nodes.at(1);

        size_t output_units;
        if (a->NDims() > 1) {
            // Matrix
            output_units = a->shape(-1);
        }
        else {
            // a is scalar: Always okay
            return;
        }

        size_t input_units;
        if (b->NDims() > 1) {
            // Matrix
            input_units = b->shape(-2);
        }
        else if (b->NDims() > 0) {
            // Vector
            input_units = b->shape(-1);
        }
        else {
            // b is scalar: Always okay
            return;
        }

        if (output_units != input_units) {
            throw std::invalid_argument(
                "Mismatch of output dimension: " + std::to_string(input_units) +
                "!=" + std::to_string(output_units));
        }
    }

    Index
    outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        dimChecks(input_nodes);
        auto& a       = *input_nodes.at(0);
        auto& b       = *input_nodes.at(1);
        long  OutDims = a.NDims() + b.NDims() - 2;

        if (a.isScalar()) {
            return b.shape();
        }
        if (b.isScalar()) {
            return a.shape();
        }

        Index outShape(OutDims);

        for (long i = 0; i < static_cast<long>(a.NDims() - 1l); i++) {
            outShape[i] = a.shape(i);
        }
        for (long i = 0; i < static_cast<long>(b.NDims() - 2l); i++) {
            outShape[i + a.NDims() - 1] = b.shape(i);
        }

        // outShape has dim 0 if we have a scalar product
        if (outShape.size() > 0) {
            outShape[-1] = b.shape(-1);
        }
        return outShape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>* output_node) override
    {

        auto& output = output_node->values();

        auto& a = input_nodes.at(0)->values();
        auto& b = input_nodes.at(1)->values();

        // Special cases for products with scalars
        if (a.isScalar()) {
            auto b_itt = b.begin();
            for (auto& val : output) {
                val = a() * *b_itt;
                ++b_itt;
            }
            return;
        }
        if (b.isScalar()) {
            auto a_itt = a.begin();
            for (auto& val : output) {
                val = *a_itt * b();
                ++a_itt;
            }
            return;
        }

        // Matrix product for arbitrary dimensions a la numpy.dot
        auto a_view = a.view();
        auto b_view = b.view();

        if (a_view.NDims() <= 1) {
            // a is a vector. Make it a matrix
            a_view.prependAxis();
        }
        if (b_view.NDims() <= 1) {
            // b is a vector. Make it a matrix
            b_view.appendAxis();
        }

        a_view = a_view.shrinkToNDimsFromRight(2);
        b_view = b_view.shrinkToNDimsFromRight(3);

        Index out_view_shape(3);

        out_view_shape[0] = a_view.shape(0);
        out_view_shape[1] = b_view.shape(0);
        out_view_shape[2] = b_view.shape(-1);

        auto out_view = output.viewAs(out_view_shape);

        for (size_t i = 0; i < a_view.shape(0); i++) {
            for (size_t j = 0; j < b_view.shape(0); j++) {
                for (size_t k = 0; k < a_view.shape(-1); k++) {
                    for (size_t l = 0; l < b_view.shape(-1); l++) {
                        out_view(i, j, l) += a_view(i, k) * b_view(j, k, l);
                    }
                }
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {
        auto& output_grad = output_node->gradient();

        auto& a = input_nodes.at(0)->values();
        auto& b = input_nodes.at(1)->values();

        auto& grad_a = input_nodes.at(0)->gradient();
        auto& grad_b = input_nodes.at(1)->gradient();

        if (a.isScalar() && !b.isScalar()) {
            // scalar times vector/matrix
            for (size_t i = 0; i < output_grad.shapeFlattened(-1); i++) {
                grad_a() += b(i) * output_grad(i);
                grad_b(i) += a() * output_grad(i);
            }
            return;
        }

        if (b.isScalar() && !a.isScalar()) {
            // vector times scalar/matrix
            for (size_t i = 0; i < output_grad.shapeFlattened(-1); i++) {
                grad_a(i) += b() * output_grad(i);
                grad_b() += a(i) * output_grad(i);
            }
            return;
        }

        if (b.isScalar() && a.isScalar()) {
            // scalar times scalar
            grad_a() += b() * output_grad();
            grad_b() += a() * output_grad();
            return;
        }

        auto a_view      = a.view();
        auto b_view      = b.view();
        auto a_grad_view = grad_a.view();
        auto b_grad_view = grad_b.view();

        if (a_view.NDims() <= 1) {
            a_view.prependAxis();
            a_grad_view.prependAxis();
        }
        if (b_view.NDims() <= 1) {
            b_view.appendAxis();
            b_grad_view.appendAxis();
        }

        a_view = a_view.shrinkToNDimsFromRight(2);
        b_view = b_view.shrinkToNDimsFromRight(3);

        a_grad_view = a_grad_view.shrinkToNDimsFromRight(2);
        b_grad_view = b_grad_view.shrinkToNDimsFromRight(3);

        Index out_view_shape(3);

        out_view_shape[0] = a_view.shape(0);
        out_view_shape[1] = b_view.shape(0);
        out_view_shape[2] = b_view.shape(-1);

        auto out_grad_view = output_grad.viewAs(out_view_shape);

        for (size_t i = 0; i < a_view.shape(0); i++) {
            for (size_t j = 0; j < b_view.shape(0); j++) {
                for (size_t k = 0; k < a_view.shape(-1); k++) {
                    for (size_t l = 0; l < b_view.shape(-1); l++) {
                        a_grad_view(i, k) +=
                            b_view(j, k, l) * out_grad_view(i, j, l);
                        b_grad_view(j, k, l) +=
                            a_view(i, k) * out_grad_view(i, j, l);
                    }
                }
            }
        }
    }

public:
    virtual ~DotConnector() {}
};

template <class TElem>
NodeShPtr<TElem> Dot(const NodeShPtr<TElem>& a, const NodeShPtr<TElem>& b)
{
    return Connector<TElem>::template apply<DotConnector>(std::move(a),
                                                          std::move(b));
}

} // namespace snnl