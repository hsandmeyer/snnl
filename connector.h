#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <algorithm>
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace snnl {

template <class TElem>
class TConnectorBaseImpl {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TConnector<TElem>;

protected:
    std::vector<TNodeBaseImpl<TElem>*>                 _next_nodes;
    std::vector<std::shared_ptr<TNodeBaseImpl<TElem>>> _prev_nodes;
    std::vector<TNodeBaseImpl<TElem>*>                 _weights;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    virtual void connectNextNodeHandler(TNodeBaseImpl<TElem>&){};

    virtual void connectPrevNodeHandler(TNodeBaseImpl<TElem>&){};

    virtual void connectWeightNodeHandler(TNodeBaseImpl<TElem>&){};

public:
    virtual TNode<TElem> createOutputNode(std::vector<TNode<TElem>>&)
    {
        return TNode<TElem>();
    };

    virtual void forwardHandler() = 0;

    virtual void buildHandler(bool was_build_before) = 0;

    virtual void backwardHandler() = 0;

    virtual ~TConnectorBaseImpl() {}

    auto& prevNodes() { return _prev_nodes; }

    auto& nextNodes() { return _next_nodes; }

    std::shared_ptr<TNodeBaseImpl<TElem>>
    addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        return addWeightTensor(TIndex{shape});
    }

    template <typename TArray>
    std::shared_ptr<TNodeBaseImpl<TElem>> addWeightTensor(const TArray& shape)
    {
        std::shared_ptr<TNodeBaseImpl<TElem>> weight =
            std::make_shared<TNodeBaseImpl<TElem>>(shape);

        _prev_nodes.push_back(weight);
        connectWeightNodeHandler(*weight);

        _weights.push_back(weight.get());
        return weight;
    }

    void forward()
    {
        _prev_ready++;
        if (_prev_ready == _prev_nodes.size() - _weights.size()) {

            forwardHandler();

            for (auto& node : _next_nodes) {
                node->forward();
            }
            _prev_ready = 0;
        }
    }

    void backward()
    {
        _next_ready++;
        if (_next_ready == _next_nodes.size()) {

            backwardHandler();

            for (auto& node : _prev_nodes) {
                node->backward();
            }

            _next_ready = 0;
        }
    }

    void build()
    {
        if (!_was_build) {
            buildHandler(_was_build);
        }
        _was_build = true;
    }

    void connectNextNode(std::shared_ptr<TNodeBaseImpl<TElem>> next_node)
    {
        _next_nodes.push_back(next_node.get());
        connectNextNodeHandler(*next_node);
    }

    void connectPrevNode(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node)
    {
        _prev_nodes.push_back(prev_node);
        connectPrevNodeHandler(*prev_node);
    }

    TNodeBaseImpl<TElem>& nextNode(const size_t i)
    {
        return *_next_nodes.at(i);
    }

    TNodeBaseImpl<TElem>& prevNode(const size_t i)
    {
        return *_prev_nodes.at(i);
    }

    TNodeBaseImpl<TElem>* weight(size_t i) { return _weights.at(i); }

    bool isWeight(TNodeBaseImpl<TElem>* node)
    {
        for (TNodeBaseImpl<TElem>*& prev : _weights) {
            if (prev == node) {
                return true;
            }
        }
        return false;
    }
};

template <class TElem>
class TDenseConnectorImpl : public TConnectorBaseImpl<TElem> {

    TNodeBaseImpl<TElem>* _W;
    TNodeBaseImpl<TElem>* _B;

    std::vector<TNodeBaseImpl<TElem>*> _inputs;
    std::vector<TNodeBaseImpl<TElem>*> _outputs;

    size_t _input_units = -1;
    size_t _output_units;

    virtual void connectPrevNodeHandler(TNodeBaseImpl<TElem>& prev) override
    {
        _inputs.push_back(&prev);
    }

    virtual void connectNextNodeHandler(TNodeBaseImpl<TElem>& prev) override
    {
        _outputs.push_back(&prev);
    }

    virtual TNode<TElem>
    createOutputNode(std::vector<TNode<TElem>>& nodes) override
    {
        if (nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Dense Connector");
        }

        return TNode<TElem>();
    }

    virtual void buildHandler(bool was_build_before) override
    {

        if (!was_build_before) {
            if (_input_units == -1ul) {
                _input_units = this->_prev_nodes.at(0)->shape(-1);
            }
            this->addWeightTensor({_output_units, _input_units});
            this->addWeightTensor({_output_units});
        }

        _W = this->weight(0);
        _B = this->weight(1);
    }

    void dimChecks(TTensor<TElem>& input)
    {
        if (_input_units != input.shape(-1)) {
            throw std::invalid_argument("Output dimsion of previous node (" +
                                        std::to_string(input.shape(-1)) +
                                        ") != " + std::to_string(_input_units));
        }
    }

    void forwardHandler() override
    {

        for (size_t node_ind = 0; node_ind < this->_inputs.size(); node_ind++) {

            // if (this->isWeight(this->_prev_nodes[node_ind].get())) {
            //    continue;
            //}

            auto& output = _outputs[node_ind]->values();
            auto& input  = _inputs[node_ind]->values();

            dimChecks(input);

            TIndex out_shape = input.shape();
            out_shape[-1]    = _output_units;
            output.setDims(out_shape);

            if (out_shape.NDims() > 1) {
                // TODO Generalize this using a better tensor loop with some
                // kind of ellipsis object
                for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                     higherDim++) {
                    for (size_t i = 0; i < output.shape(-1); i++) {
                        output(higherDim, i) = _B->value(i);
                        for (size_t j = 0; j < input.shape(-1); j++) {
                            output(higherDim, i) +=
                                _W->value(i, j) * input(higherDim, j);
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
                    output(i) = _B->value(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(i) += _W->value(i, j) * input(j);
                    }
                }
            }
        }
    }

    void backwardHandler() override
    {
        // TODO: //Calculate dfdw and dfdz here
    }

public:
    TDenseConnectorImpl(size_t output_dim) : _output_units(output_dim) {}

    TDenseConnectorImpl(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {
    }

    virtual ~TDenseConnectorImpl()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

template <class TElem>
class TConnector {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TConnectorBaseImpl<TElem>;

    std::shared_ptr<TConnectorBaseImpl<TElem>> _impl;

    TConnector(std::shared_ptr<TConnectorBaseImpl<TElem>> impl) : _impl(impl) {}

public:
    void connectNextNode(const TNode<TElem>& next)
    {
        _impl->connectNextNode(next._impl);
        next._impl->connectPrevConnector(_impl);
    }

    void connectPrevNode(TNode<TElem>& prev)
    {
        _impl->connectPrevNode(prev._impl);
        prev._impl->connectNextConnector(_impl);
    }

    template <typename... TNodes>
    TNode<TElem> operator()(TNodes&... prev)
    {
        std::vector<TNode<TElem>> nodes{prev...};

        for (auto& node : nodes) {
            connectPrevNode(node);
        }

        TNode<TElem> out(_impl->createOutputNode(nodes));
        connectNextNode(out);

        _impl->build();

        return out;
    }

    static TConnector<TElem> TDenseConnector(size_t input_dim,
                                             size_t output_dim)
    {
        return TConnector<TElem>(std::make_shared<TDenseConnectorImpl<TElem>>(
            input_dim, output_dim));
    }

    TConnectorBaseImpl<TElem>* get() { return _impl.get(); }

    TConnectorBaseImpl<TElem>* operator->() { return _impl.get(); }
};

} // namespace snnl