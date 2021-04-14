#pragma once
#include "connector.h"

namespace snnl {

template <class TElem>
class TDenseConnector : public TConnector<TElem> {

    friend class TConnector<TElem>;

    TNode<TElem>* _W;
    TNode<TElem>* _B;

    std::vector<TNode<TElem>*> _inputs;
    std::vector<TNode<TElem>*> _outputs;

    size_t _input_units = -1;
    size_t _output_units;

    void dimChecks(const std::vector<TNodeShPtr<TElem>>& input_nodes) const
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one input node per call for dense layer");
        }
        if (input_nodes.empty()) {
            throw std::invalid_argument("No input nodes provided");
        }
        if (_input_units != input_nodes.front()->shape(-1)) {
            throw std::invalid_argument(
                "Output dimsion of previous node (" +
                std::to_string(input_nodes.front()->shape(-1)) +
                ") != " + std::to_string(_input_units));
        }
    }

    TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const override
    {
        dimChecks(input_nodes);

        TIndex out_shape = input_nodes.front()->shape();
        out_shape[-1]    = _output_units;
        return out_shape;
    }

    void forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                        const std::vector<TNodeShPtr<TElem>>& weights,
                        TNode<TElem>* output_node) override
    {

        // std::cout << "FORWARD on dense layer" << std::endl;
        dimChecks(input_nodes);

        auto& input  = input_nodes.front()->values();
        auto& output = output_node->values();

        TNode<TElem>& W = *weights.at(0);
        TNode<TElem>& B = *weights.at(1);

        if (input.NDims() > 1) {
            // TODO Generalize this using a better tensor loop with some
            // kind of ellipsis object
            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = B.value(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(higherDim, i) +=
                            W.value(i, j) * input(higherDim, j);
                    }
                }
            }
            /*
            output.forEach([&](const TIndex& ind_in) {
                int i          = ind_in[-1];
                output(ind_in) = _B->value(i);
                auto ind_out   = ind_in;
                for (size_t j = 0; j < input.shape(-1); j++) {
                    ind_out[-1] = j;
                    output(ind_in) += _W->value(i, j) * input(ind_out);
                }
            });
            */
        }
        else {
            for (size_t i = 0; i < output.shape(-1); i++) {
                output(i) = B.value(i);
                for (size_t j = 0; j < input.shape(-1); j++) {
                    output(i) += W.value(i, j) * input(j);
                }
            }
        }
    }

    void backwardHandler(const TNode<TElem>*             output,
                         std::vector<TNodeShPtr<TElem>>& weights,
                         std::vector<TNodeShPtr<TElem>>& input_nodes) override
    {
        // std::cout << "BACKWARD on dense layer" << std::endl;
        dimChecks(input_nodes);

        auto& input = input_nodes.front();

        TNode<TElem>& W = *weights.at(0);
        TNode<TElem>& B = *weights.at(1);

        if (input->NDims() > 1) {
            for (size_t higherDim = 0; higherDim < input->shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output->shape(-1); i++) {
                    B.grad(i) += output->grad(higherDim, i);
                    for (size_t j = 0; j < input->shape(-1); j++) {
                        input->grad(higherDim, j) +=
                            W.value(i, j) * output->grad(higherDim, i);
                        W.grad(i, j) += input->value(higherDim, j) *
                                        output->grad(higherDim, i);
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < output->shape(-1); i++) {
                B.grad(i) += output->grad(i);
                for (size_t j = 0; j < input->shape(-1); j++) {
                    input->grad(j) += W.value(i, j) * output->grad(i);
                    W.grad(i, j) += input->value(j) * output->grad(i);
                }
            }
        }
    }

    TDenseConnector(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {

        this->addWeightTensor({_output_units, _input_units});
        this->addWeightTensor({_output_units});

        _W = this->weight(0).get();
        _B = this->weight(1).get();
    }

public:
    TTensor<TElem>& W()
    {
        if (!_W) {
            throw std::runtime_error(
                "Weights for dense layer not initialized. Connect layer first");
        }
        return _W->values();
    }

    TTensor<TElem>& B()
    {
        if (!_B) {
            throw std::runtime_error(
                "Weights for dense layer not initialized. Connect layer first");
        }
        return _B->values();
    }
    virtual ~TDenseConnector()
    {
        std::cout << "Destroying Dense Connector" << std::endl;
    }
};

} // namespace snnl