#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <memory>

namespace snnl {

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
    virtual void connectNextLayerHandler(TLayerBaseImpl<TElem> &) = 0;

    virtual void connectPrevLayerHandler(TLayerBaseImpl<TElem> &) = 0;

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

    virtual ~TNodeBaseImpl()
    {

        std::cout << "Destroying TNodeBase" << std::endl;
    }

    TLayerBaseImpl<TElem> &prevLayer() { return _prev_layer; }

    TLayerBaseImpl<TElem> &nextLayers() { return _next_layers; }

    TLayerBaseImpl<TElem> &nextLayer(const size_t i)
    {
        return _next_layers.at(i);
    }

    void call()
    {
        for (auto &layer : _next_layers) {
            layer->call();
        }
    }

    void connectNextLayer(std::shared_ptr<TLayerBaseImpl<TElem>> &next)
    {
        _next_layers.push_back(next);
        connectNextLayerHandler(*next);
    }

    void connectPrevLayer(std::shared_ptr<TLayerBaseImpl<TElem>> &prev)
    {
        if (_prev_layer) {
            throw std::invalid_argument(
                "Node already connected to a previous layer");
        }
        _prev_layer = prev.get();
        connectPrevLayerHandler(*prev);
    }

    TTensor<TElem> *operator->() { return &_values; }

    const TTensor<TElem> *operator->() const { return &_values; }

    TTensor<TElem> &values() { return _values; }

    size_t shape(int i) const { return _values.shape(i); }

    std::vector<size_t> &shape() { return _values.shape(); }

    const std::vector<size_t> &shape() const { return _values.shape(); }
};

template <class TElem>
class TDefaultNodeImpl : public TNodeBaseImpl<TElem> {

protected:
    virtual void connectNextLayerHandler(TLayerBaseImpl<TElem> &) {}

    virtual void connectPrevLayerHandler(TLayerBaseImpl<TElem> &) {}

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
        return (*_impl)(args...);
    }

    template <typename... Args>
    const TElem &operator()(const Args &&... args) const
    {
        return (*_impl)(std::forward(args...));
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

    template <typename... TArgs>
    static TNode Default(TArgs... args)
    {
        return TNode(std::make_shared<TDefaultNodeImpl<TElem>>(args...));
    }

    static TNode Default(const std::initializer_list<size_t> list)
    {
        return TNode(std::make_shared<TDefaultNodeImpl<TElem>>(list));
    }

    TNodeBaseImpl<TElem> *operator->() { return _impl.get(); }

    const TNodeBaseImpl<TElem> *operator->() const { return _impl.get(); }
};
} // namespace snnl