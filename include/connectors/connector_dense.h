#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class TDenseConnector : public TConnector<TElem> {

    friend class TConnector<TElem>;

    void dimChecks(const std::vector<TNodeShPtr<TElem>>& input_nodes) const
    {
        size_t input_units  = input_nodes.at(0)->shape(-1);
        size_t output_units = input_nodes.at(0)->shape(-2);
        if (input_nodes.size() != 3) {
            throw std::invalid_argument(
                "Need weight, bias and input node for dense layer");
        }
        if (output_units != input_nodes.at(1)->shape(0)) {
            throw std::invalid_argument(
                "Mismatch of output dimension of b (second input)" +
                std::to_string(input_nodes.at(1)->shape(0)) +
                "!=" + std::to_string(output_units));
        }
        if (input_units != input_nodes.at(2)->shape(-1)) {
            throw std::invalid_argument(
                "Mismatch of output dimension of x (third input)" +
                std::to_string(input_nodes.at(2)->shape(-1)) +
                "!=" + std::to_string(input_units));
        }
    }

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        dimChecks(input_nodes);

        TIndex out_shape = input_nodes.back()->shape();
        out_shape[-1]    = input_nodes.front()->shape(-2);
        return out_shape;
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        TNode<TElem>* output_node) override
    {

        // std::cout << "FORWARD on dense layer" << std::endl;
        dimChecks(input_nodes);

        auto& output = output_node->values();

        TNode<TElem>& W = *input_nodes.at(0);
        TNode<TElem>& B = *input_nodes.at(1);
        TNode<TElem>& x = *input_nodes.at(2);

        if (x.NDims() > 1) {
            // TODO Generalize this using a better tensor loop with some
            // kind of ellipsis object
            for (size_t higherDim = 0; higherDim < x.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = B.value(i);
                    for (size_t j = 0; j < x.shape(-1); j++) {
                        output(higherDim, i) +=
                            W.value(i, j) * x.value(higherDim, j);
                    }
                }
            }
            /*
            output.forEach([&](const TIndex& ind_in) {
                int i          = ind_in[-1];
                output(ind_in) = _B->value(i);
                auto ind_out   = ind_in;
                for (size_t j = 0; j < x.shape(-1); j++) {
                    ind_out[-1] = j;
                    output(ind_in) += _W->value(i, j) * x(ind_out);
                }
            });
            */
        }
        else {
            for (size_t i = 0; i < output.shape(-1); i++) {
                output(i) = B.value(i);
                for (size_t j = 0; j < x.shape(-1); j++) {
                    output(i) += W.value(i, j) * x.value(j);
                }
            }
        }
    }

    void backwardHandler(const TNode<TElem>*             output,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
    {
        // std::cout << "BACKWARD on dense layer" << std::endl;
        dimChecks(input_nodes);

        TNode<TElem>& W = *input_nodes.at(0);
        TNode<TElem>& B = *input_nodes.at(1);
        TNode<TElem>& x = *input_nodes.at(2);

        if (x.NDims() > 1) {
            for (size_t higherDim = 0; higherDim < x.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output->shape(-1); i++) {
                    B.grad(i) += output->grad(higherDim, i);
                    for (size_t j = 0; j < x.shape(-1); j++) {
                        x.grad(higherDim, j) +=
                            W.value(i, j) * output->grad(higherDim, i);
                        W.grad(i, j) +=
                            x.value(higherDim, j) * output->grad(higherDim, i);
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < output->shape(-1); i++) {
                B.grad(i) += output->grad(i);
                for (size_t j = 0; j < x.shape(-1); j++) {
                    x.grad(j) += W.value(i, j) * output->grad(i);
                    W.grad(i, j) += x.value(j) * output->grad(i);
                }
            }
        }
    }

public:
    virtual ~TDenseConnector()
    {
        std::cout << "Destroying Dense Connector" << std::endl;
    }
};

template <class TElem>
TNodeShPtr<TElem> Dense(const TNodeShPtr<TElem>& W, const TNodeShPtr<TElem>& b,
                        const TNodeShPtr<TElem>& x)
{
    return TConnector<TElem>::template apply<TDenseConnector>(
        std::move(W), std::move(b), std::move(x));
}

} // namespace snnl