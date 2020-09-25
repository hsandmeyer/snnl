#include "tensor.h"
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>
#pragma once

namespace snnl {

template <class TElem>
class TLayerBaseImpl;

template <class TElem>
class TLayer;

template <class TElem>
class TNode;

template <class TElem>
class TNodeBaseImpl {

    friend class TNode<TElem>;
    friend class TLayer<TElem>;
    friend class TLayerBaseImpl<TElem>;

    size_t _batch_size = 1;

    TTensor<TElem> _values;

    std::vector<std::shared_ptr<TLayerBaseImpl<TElem>>> _next_layers = {};
    TLayerBaseImpl<TElem> *                             _prev_layer  = nullptr;

protected:
    virtual void
    connectNextLayerHandler(std::shared_ptr<TLayerBaseImpl<TElem>> &) = 0;
    virtual void
                 connectPrevLayerHandler(std::shared_ptr<TLayerBaseImpl<TElem>> &) = 0;
    virtual void callHandler() = 0;

public:
    template <typename TArray>
    TNodeBaseImpl(TArray shape)
    {
        std::vector<size_t> full_dims{_batch_size};
        full_dims.insert(full_dims.end(), shape.begin(), shape.end());
        _values.setDims(full_dims);
    }

    TNodeBaseImpl(const std::initializer_list<size_t> shape)
        : TNodeBaseImpl(std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    template <typename... TArgs>
    TNodeBaseImpl(TArgs... args)
        : TNodeBaseImpl(std::vector<size_t>{static_cast<size_t>(args)...})
    {
    }

    template <typename... Args>
    TElem &operator()(const Args... args)
    {
        return _values(args...);
    }

    template <typename... Args>
    const TElem &operator()(const Args... args) const
    {
        return _values(args...);
    }

    int NDims() const { return _values.NDims() - 1; }

    size_t shape(const int i) const { return _values.shape(i); }

    std::vector<size_t> shape() const { return _values.subShape(1, NDims()); }

    virtual ~TNodeBaseImpl()
    {

        std::cout << "Destroying TNodeBase" << std::endl;
    }

    auto &prevLayer() { return _prev_layer; }

    auto &nextLayers() { return _next_layers; }

    TTensor<TElem> &getTensor() { return _values; }

    void call()
    {
        for (auto &layer : _next_layers) {
            layer->call();
        }
    }

    void connectNextLayer(std::shared_ptr<TLayerBaseImpl<TElem>> &next)
    {
        _next_layers.push_back(next);
        connectNextLayerHandler(next);
    }

    void connectPrevLayer(std::shared_ptr<TLayerBaseImpl<TElem>> &prev)
    {
        if (_prev_layer) {
            throw std::invalid_argument(
                "Node already connected to a previous layer");
        }
        _prev_layer = prev.get();
        connectPrevLayerHandler(prev);
    }
}; // namespace snnl

template <class TElem>
class TDefaultNodeImpl : public TNodeBaseImpl<TElem> {

protected:
    virtual void
    connectNextLayerHandler(std::shared_ptr<TLayerBaseImpl<TElem>> &)
    {
    }
    virtual void
    connectPrevLayerHandler(std::shared_ptr<TLayerBaseImpl<TElem>> &)
    {
    }
    virtual void callHandler() {}

public:
    template <typename TArray>
    TDefaultNodeImpl(TArray shape) : TNodeBaseImpl<TElem>(shape)
    {
    }

    template <typename TArray>
    TDefaultNodeImpl(const std::initializer_list<size_t> shape)
        : TNodeBaseImpl<TElem>(shape)
    {
    }

    template <typename... TArray>
    TDefaultNodeImpl(TArray... args) : TNodeBaseImpl<TElem>(args...)
    {
    }
};

template <class TElem>
class TNode {

    friend class TNodeBaseImpl<TElem>;
    friend class TLayer<TElem>;
    friend class TLayerBaseImpl<TElem>;

    std::shared_ptr<TNodeBaseImpl<TElem>> _impl;

public:
    TNode(const std::shared_ptr<TNodeBaseImpl<TElem>> &impl) : _impl(impl) {}

    template <typename... Args>
    TElem &operator()(const Args... args)
    {
        return _impl(args...);
    }

    template <typename... Args>
    const TElem &operator()(const Args... args) const
    {
        return _impl(args...);
    }

    TNodeBaseImpl<TElem> *get() { return _impl.get(); }

    void connectNextLayer(TLayerBaseImpl<TElem> &next)
    {
        _impl->connectNextLayer(next);
        next._impl->connectPrevNode(_impl);
    }

    void connectPrevLayer(TLayer<TElem> &prev)
    {
        _impl->connectPrevLayer(prev._impl);
        prev._impl->connectNextNode(_impl);
    }

    std::vector<size_t> shape() const { return _impl->shape(); }

    size_t shape(const int i) const { return _impl->shape(i); }

    TTensor<TElem> &getTensor() { return _impl->getTensor(); }

    void call() { _impl->call(); }

    TNode Default() {}

    template <typename... TArgs>
    static TNode Default(TArgs... args)
    {
        return TNode(std::make_shared<TDefaultNodeImpl<TElem>>(args...));
    }

    static TNode Default(const std::initializer_list<size_t> list)
    {
        return TNode(std::make_shared<TDefaultNodeImpl<TElem>>(list));
    }
};

template <class TElem>
class TLayer;

template <class TElem>
class TLayerBaseImpl {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TLayer<TElem>;

protected:
    std::vector<std::shared_ptr<TNodeBaseImpl<TElem>>> _next_nodes;
    std::vector<TNodeBaseImpl<TElem> *>                _prev_nodes;
    size_t                                             _prev_ready = 0;

    virtual void
    connectNextNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>> next_node) = 0;

    virtual void
    connectPrevNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node) = 0;

public:
    virtual TNode<TElem> createOutputNode() = 0;

    virtual void callHandler() = 0;

    virtual void updateWeights() = 0;

    virtual ~TLayerBaseImpl() {}

    auto &prevNodes() { return _prev_nodes; }

    auto &nextNodes() { return _next_nodes; }

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

    void connectNextNode(std::shared_ptr<TNodeBaseImpl<TElem>> next_node)
    {
        _next_nodes.push_back(next_node);
        connectNextNodeHandler(next_node);
    }

    void connectPrevNode(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node)
    {
        _prev_nodes.push_back(prev_node.get());
        connectPrevNodeHandler(prev_node);
    }
};

template <class TElem>
class TDenseLayerImpl : public TLayerBaseImpl<TElem> {

    TTensor<TElem> _weights;
    TTensor<TElem> _bias;

    TNode<TElem> createOutputNode()
    {
        auto shape              = this->_prev_nodes.at(0)->shape();
        shape[shape.size() - 1] = _bias.shape(-1);
        return TNode<TElem>::Default(shape);
    }

    void updateWeights() {}

    void callHandler()
    {
        for (size_t node_ind = 0; node_ind < this->_prev_nodes.size();
             node_ind++) {
            auto &output = this->_next_nodes[node_ind]->getTensor();
            auto &input  = this->_prev_nodes[node_ind]->getTensor();

            for (size_t i = 0; i < output.shape(0); i++) {
                output(i) = _bias(i);
                for (size_t j = 0; j < input.shapeFlattened(1); j++) {
                    output(i) += _weights(i, j) * input(j);
                }
            }
        }
    }

    void connectNextNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>>)
    {
        // TODO dimension check with existing nodes
    }

    void connectPrevNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>>)
    {
        // TODO dimension check with existing nodes
    }

public:
    TDenseLayerImpl(size_t output_dim, size_t input_dim)
        : _weights({output_dim, input_dim}), _bias({output_dim})
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

    TNode<TElem> operator()(TNode<TElem> &prev)
    {
        connectPrevLayer(prev);
        TNode<TElem> out(_impl->createOutputNode());
        connectNextLayer(out);

        return out;
    }

    static TLayer<TElem> TDenseLayer(size_t output_dim, size_t input_dim)
    {
        return TLayer<TElem>(
            std::make_shared<TDenseLayerImpl<TElem>>(output_dim, input_dim));
    }

    static TLayer<TElem> TDenseLayer(size_t output_dim, TNode<TElem> &input)
    {
        return TLayer<TElem>(std::make_shared<TDenseLayerImpl<TElem>>(
            output_dim, input.shape(-1)));
    }

    TLayerBaseImpl<TElem> *get() { return _impl.get(); }
};

} // namespace snnl