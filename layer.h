#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace snnl {

template <class TElem>
class TLayerBaseImpl {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TLayer<TElem>;

protected:
    std::vector<std::shared_ptr<TNodeBaseImpl<TElem>>> _next_nodes;
    std::vector<TNodeBaseImpl<TElem>*>                 _prev_nodes;

    std::vector<TTensor<TElem>> _all_weights;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    virtual void connectNextNodeHandler(TNodeBaseImpl<TElem>& next_node) = 0;

    virtual void connectPrevNodeHandler(TNodeBaseImpl<TElem>& prev_node) = 0;

public:
    virtual TNode<TElem> createOutputNode(std::vector<TNode<TElem>>&) const = 0;

    virtual void callHandler() = 0;

    virtual void buildHandler(bool was_build_before) = 0;

    virtual void backCallHandler() = 0;

    virtual ~TLayerBaseImpl() {}

    auto& prevNodes() { return _prev_nodes; }

    auto& nextNodes() { return _next_nodes; }

    void addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        _all_weights.push_back(TTensor<TElem>(shape));
    }

    template <typename TArray>
    void addWeightTensor(const TArray& shape)
    {
        _all_weights.push_back(TTensor<TElem>(shape));
    }

    void call()
    {
        _prev_ready++;
        if (_prev_ready == _prev_nodes.size()) {

            callHandler();

            for (auto& node : _next_nodes) {
                node->call();
            }
            _prev_ready = 0;
        }
    }

    void backCall()
    {
        _next_ready++;
        if (_next_ready == _next_nodes.size()) {

            backCallHandler();

            for (auto& node : _prev_nodes) {
                node->backCall();
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
        _next_nodes.push_back(next_node);
        connectNextNodeHandler(*next_node);
    }

    void connectPrevNode(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node)
    {
        _prev_nodes.push_back(prev_node.get());
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

    std::vector<TTensor<TElem>*> weights() { return _all_weights; }

    TTensor<TElem>& weights(const int i)
    {
        if (_all_weights.size() == 0) {
            throw std::out_of_range(
                "No weights registered yet. Call Layer first");
        }
        return _all_weights.at(i);
    }
};

template <class TElem>
class TDenseLayerImpl : public TLayerBaseImpl<TElem> {

    TTensor<TElem>* _weights;
    TTensor<TElem>* _bias;
    size_t          _input_units = -1;
    size_t          _output_units;

    TNode<TElem> createOutputNode(std::vector<TNode<TElem>>& nodes) const
    {
        if (nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Dense layer");
        }

        return TNode<TElem>::Default();
    }

    virtual void buildHandler(bool was_build_before)
    {

        if (!was_build_before) {
            if (_input_units == -1ul) {
                _input_units = this->_prev_nodes.at(0)->shape(-1);
            }
            this->addWeightTensor({_output_units, _input_units});
            this->addWeightTensor({_output_units});
        }

        _weights = &this->_all_weights.at(0);
        _bias    = &this->_all_weights.at(1);
    }

    void dimChecks()
    {
        for (auto& prev : this->_prev_nodes) {
            if (_input_units != prev->shape(-1)) {
                throw std::invalid_argument(
                    "Output dimsion of previous node (" +
                    std::to_string(prev->shape(-1)) +
                    ") != " + std::to_string(_input_units));
            }
        }
    }

    void callHandler()
    {
        dimChecks();

        for (size_t node_ind = 0; node_ind < this->_prev_nodes.size();
             node_ind++) {
            auto& output = this->_next_nodes[node_ind]->values();
            auto& input  = this->_prev_nodes[node_ind]->values();

            TIndex out_shape = input.shape();
            out_shape[-1]    = _output_units;
            output.setDims(out_shape);

            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = (*_bias)(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(higherDim, i) +=
                            (*_weights)(i, j) * input(higherDim, j);
                    }
                }
            }
        }
    }

    void backCallHandler()
    {
        // TODO: //Calculate dfdw and dfdz here
    }

    void connectNextNodeHandler(TNodeBaseImpl<TElem>&)
    {
        /*
        if (next->shape(-1) != _output_units) {
            throw std::invalid_argument(
                "Output dimsion of next node (" +
                std::to_string(next->shape(-1)) +
                ") != " + std::to_string(_output_units));
        }
        */
    }

    void connectPrevNodeHandler(TNodeBaseImpl<TElem>&)
    {
        /*
        if (prev->shape(-1) != _input_units) {
            throw std::invalid_argument("Output dimsion of previous node (" +
                                        std::to_string(prev->shape(-1)) +
                                        ") != " + std::to_string(_input_units));
        }
        */
    }

public:
    TDenseLayerImpl(size_t output_dim) : _output_units(output_dim) {}

    TDenseLayerImpl(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {
    }

    virtual ~TDenseLayerImpl()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

template <class TElem>
class TLayer {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TLayerBaseImpl<TElem>;

    std::shared_ptr<TLayerBaseImpl<TElem>> _impl;

    TLayer(std::shared_ptr<TLayerBaseImpl<TElem>> impl) : _impl(impl) {}

public:
    void connectNextLayer(const TNode<TElem>& next)
    {
        _impl->connectNextNode(next._impl);
        next._impl->connectPrevLayer(_impl);
    }

    void connectPrevLayer(TNode<TElem>& prev)
    {
        _impl->connectPrevNode(prev._impl);
        prev._impl->connectNextLayer(_impl);
    }

    template <typename... TNodes>
    TNode<TElem> operator()(TNodes&... prev)
    {
        std::vector<TNode<TElem>> nodes{prev...};

        for (auto& node : nodes) {
            connectPrevLayer(node);
        }

        TNode<TElem> out(_impl->createOutputNode(nodes));
        connectNextLayer(out);

        _impl->build();

        return out;
    }

    static TLayer<TElem> TDenseLayer(size_t input_dim, size_t output_dim)
    {
        return TLayer<TElem>(
            std::make_shared<TDenseLayerImpl<TElem>>(input_dim, output_dim));
    }

    TLayerBaseImpl<TElem>* get() { return _impl.get(); }

    TLayerBaseImpl<TElem>* operator->() { return _impl.get(); }
};

} // namespace snnl