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

    TTensor<TElem> _values;
    TTensor<TElem> _gradients;

    std::vector<std::shared_ptr<TLayerBaseImpl<TElem>>> _next_layers = {};
    TLayerBaseImpl<TElem>*                              _prev_layer  = nullptr;

protected:
    virtual void connectNextLayerHandler(TLayerBaseImpl<TElem>&) = 0;

    virtual void connectPrevLayerHandler(TLayerBaseImpl<TElem>&) = 0;

    virtual void callHandler() = 0;

public:
    TNodeBaseImpl() = default;

    template <typename TArray>
    TNodeBaseImpl(TArray shape)
    {
        setDims(shape);
    }

    TNodeBaseImpl(const std::initializer_list<size_t> shape)
        : TNodeBaseImpl(std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    template <typename... Args>
    TElem& value(const Args... args)
    {
        return _values(args...);
    }

    template <typename... Args>
    const TElem& value(const Args... args) const
    {
        return _values(args...);
    }

    template <typename... Args>
    const TElem& delta(const Args... args) const
    {
        return _deltas(args...);
    }

    template <typename... Args>
    TElem& delta(const Args... args)
    {
        return _deltas(args...);
    }

    virtual ~TNodeBaseImpl()
    {

        std::cout << "Destroying TNodeBase" << std::endl;
    }

    TLayerBaseImpl<TElem>& prevLayer() { return _prev_layer; }

    auto& nextLayers() { return _next_layers; }

    TLayerBaseImpl<TElem>& nextLayer(const size_t i)
    {
        return _next_layers.at(i);
    }

    void call()
    {
        for (auto& layer : _next_layers) {
            layer->call();
        }
    }

    void backCall()
    {
        if (_prev_layer) {
            // TODO: calculate deltas here

            _prev_layer->backCall();
        }
    }

    void connectNextLayer(std::shared_ptr<TLayerBaseImpl<TElem>>& next)
    {
        _next_layers.push_back(next);
        connectNextLayerHandler(*next);
    }

    void connectPrevLayer(std::shared_ptr<TLayerBaseImpl<TElem>>& prev)
    {
        if (_prev_layer) {
            throw std::invalid_argument(
                "Node already connected to a previous layer");
        }
        _prev_layer = prev.get();
        connectPrevLayerHandler(*prev);
    }

    // TTensor<TElem>* operator->() { return &_values; }

    // const TTensor<TElem>* operator->() const { return &_values; }

    TTensor<TElem>& values() { return _values; }

    TTensor<TElem>& gradients() { return _gradients; }

    size_t shape(int i) const { return _values.shape(i); }

    TIndex shape() { return _values.shape(); }

    const TIndex shape() const { return _values.shape(); }

    template <typename TArray>
    void setDims(const TArray& arr)
    {
        _values.setDims(arr);
        _gradients.setDims(arr);
    }

    void setDims(const std::initializer_list<size_t> shape)
    {
        _values.setDims(shape);
        _gradients.setDims(shape);
    }
};

template <class TElem>
class TDefaultNodeImpl : public TNodeBaseImpl<TElem> {

protected:
    virtual void connectNextLayerHandler(TLayerBaseImpl<TElem>&) {}

    virtual void connectPrevLayerHandler(TLayerBaseImpl<TElem>&) {}

    virtual void callHandler() {}

public:
    TDefaultNodeImpl() = default;

    template <typename TArray>
    TDefaultNodeImpl(TArray& shape) : TNodeBaseImpl<TElem>(shape)
    {
    }

    template <typename TArray>
    TDefaultNodeImpl(const std::initializer_list<size_t>& shape)
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
    TNode(const std::shared_ptr<TNodeBaseImpl<TElem>>& impl) : _impl(impl) {}

    template <typename... Args>
    TElem& value(const Args... args)
    {
        return _impl->value(args...);
    }

    template <typename... Args>
    const TElem& value(const Args&&... args) const
    {
        return _impl->value(std::forward(args...));
    }

    TTensor<TElem>& values() { return _impl->_values; }

    const TTensor<TElem>& values() const { return _impl->_values; }

    template <typename... Args>
    TElem& grad(const Args... args)
    {
        return _impl->grad(args...);
    }

    template <typename... Args>
    const TElem& grad(const Args&&... args) const
    {
        return _impl->grad(std::forward(args...));
    }

    TNodeBaseImpl<TElem>* get() { return _impl.get(); }

    void connectNextLayer(TLayerBaseImpl<TElem>& next)
    {
        _impl->connectNextLayer(next);
        next._impl->connectPrevNode(_impl);
    }

    void connectPrevLayer(TLayer<TElem>& prev)
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

    TNodeBaseImpl<TElem>* operator->() { return _impl.get(); }

    const TNodeBaseImpl<TElem>* operator->() const { return _impl.get(); }

    template <typename TArray>
    void setDims(const TArray& arr)
    {
        _impl->setDims(arr);
    }

    void setDims(const std::initializer_list<size_t> shape)
    {
        _impl->setDims(shape);
    }
};
} // namespace snnl