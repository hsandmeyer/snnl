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
    std::vector<TNodeBaseImpl<TElem> *>                _prev_nodes;
    std::vector<TTensor<TElem> *>                      _all_weights;
    size_t                                             _prev_ready = 0;
    bool                                               _was_build  = false;

    virtual void connectNextNodeHandler(TNodeBaseImpl<TElem> &next_node) = 0;

    virtual void connectPrevNodeHandler(TNodeBaseImpl<TElem> &prev_node) = 0;

public:
    virtual TNode<TElem> createOutputNode(std::vector<TNode<TElem>> &) = 0;

    virtual void callHandler() = 0;

    virtual void buildHandler(std::vector<size_t> &input_shape) = 0;

    virtual void updateWeights() = 0;

    virtual ~TLayerBaseImpl() {}

    auto &prevNodes() { return _prev_nodes; }

    auto &nextNodes() { return _next_nodes; }

    template <typename... TWeights>
    void registerWeights(TTensor<TElem> &weights, TWeights &... other_weights)
    {
        _all_weights.push_back(&weights);
        registerWeights(other_weights...);
    }

    template <typename... TWeights>
    void registerWeights(TTensor<TElem> &weights)
    {
        _all_weights.push_back(&weights);
    }

    void call()
    {
        _prev_ready++;
        if (_prev_ready == _prev_nodes.size()) {

            callHandler();

            for (auto &node : _next_nodes) {
                node->call();
            }
            _prev_ready = 0;
        }
    }

    void build(std::vector<size_t> &input_shape)
    {
        if (!_was_build) {
            buildHandler(input_shape);
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

    TNodeBaseImpl<TElem> &nextNode(const size_t i)
    {
        return *_next_nodes.at(i);
    }

    TNodeBaseImpl<TElem> &prevNode(const size_t i)
    {
        return *_prev_nodes.at(i);
    }

    std::vector<TTensor<TElem> *> weights() { return _all_weights; }

    TTensor<TElem> &weights(const int i)
    {
        if (_all_weights.size() == 0) {
            throw std::out_of_range(
                "No weights registered yet. Call Layer first");
        }
        return *_all_weights.at(i);
    }
};

template <class TElem>
class TDenseLayerImpl : public TLayerBaseImpl<TElem> {

    TTensor<TElem> _weights;
    TTensor<TElem> _bias;
    size_t         _output_units;
    size_t         _input_units;

    TNode<TElem> createOutputNode(std::vector<TNode<TElem>> &nodes)
    {
        if (nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Dense layer");
        }

        auto shape              = nodes[0]->values().subShape(1, -1);
        shape[shape.size() - 1] = _bias.shape(-1);
        return TNode<TElem>::Default(shape);
    }

    virtual void buildHandler(std::vector<size_t> &input_shape)
    {
        _input_units = input_shape[input_shape.size() - 1];
        _weights.setDims({_output_units, _input_units});
        _bias.setDims({_output_units});
        this->registerWeights(_weights, _bias);
    }

    void updateWeights() {}

    void callHandler()
    {
        for (size_t node_ind = 0; node_ind < this->_prev_nodes.size();
             node_ind++) {
            auto &output = this->_next_nodes[node_ind]->values();
            auto &input  = this->_prev_nodes[node_ind]->values();

            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = _bias(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(higherDim, i) +=
                            _weights(i, j) * input(higherDim, j);
                    }
                }
            }
        }
    }

    void connectNextNodeHandler(TNodeBaseImpl<TElem> &next)
    {
        if (next->shape(-1) != _output_units) {
            throw std::invalid_argument(
                "Output dimsion of next node (" +
                std::to_string(next->shape(-1)) +
                ") != " + std::to_string(_output_units));
        }
    }

    void connectPrevNodeHandler(TNodeBaseImpl<TElem> &prev)
    {
        if (prev->shape(-1) != _input_units) {
            throw std::invalid_argument("Output dimsion of previous node (" +
                                        std::to_string(prev->shape(-1)) +
                                        ") != " + std::to_string(_input_units));
        }
    }

public:
    TDenseLayerImpl(size_t output_dim) : _output_units(output_dim) {}

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
    void connectNextLayer(const TNode<TElem> &next)
    {
        _impl->connectNextNode(next._impl);
        next._impl->connectPrevLayer(_impl);
    }

    void connectPrevLayer(TNode<TElem> &prev)
    {
        _impl->connectPrevNode(prev._impl);
        prev._impl->connectNextLayer(_impl);
    }

    template <typename... TNodes>
    TNode<TElem> operator()(TNodes &... prev)
    {
        std::vector<TNode<TElem>> nodes{prev...};
        _impl->build(nodes[0]->shape());

        for (auto &node : nodes) {
            connectPrevLayer(node);
        }

        TNode<TElem> out(_impl->createOutputNode(nodes));
        connectNextLayer(out);

        return out;
    }

    static TLayer<TElem> TDenseLayer(size_t output_dim)
    {
        return TLayer<TElem>(
            std::make_shared<TDenseLayerImpl<TElem>>(output_dim));
    }

    TLayerBaseImpl<TElem> *get() { return _impl.get(); }

    TLayerBaseImpl<TElem> *operator->() { return _impl.get(); }
};

} // namespace snnl