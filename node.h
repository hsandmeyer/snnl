#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <memory>

namespace snnl {

template <class TElem>
class TNodeBaseImpl {

    friend class TNode<TElem>;
    friend class TConnector<TElem>;
    friend class TConnectorBaseImpl<TElem>;

    TTensor<TElem> _values;
    TTensor<TElem> _gradient;

    std::vector<TConnectorBaseImpl<TElem>*>    _next_connectors = {};
    std::shared_ptr<TConnectorBaseImpl<TElem>> _prev_connector  = nullptr;

protected:
    virtual void connectNextConnectorHandler(TConnectorBaseImpl<TElem>&){};

    virtual void connectPrevConnectorHandler(TConnectorBaseImpl<TElem>&){};

    virtual void forwardHandler(){};

    virtual void backwardHandler(){};

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
    const TElem& grad(const Args... args) const
    {
        return _gradient(args...);
    }

    template <typename... Args>
    TElem& grad(const Args... args)
    {
        return _gradient(args...);
    }

    void setAllValues(const TElem& elem) { _values.setAllValues(elem); }

    void setAllGrad(const TElem& grad) { _gradient.setAllValues(grad); }

    virtual ~TNodeBaseImpl()
    {

        std::cout << "Destroying TNodeBase" << std::endl;
    }

    TConnectorBaseImpl<TElem>& prevConnector() { return _prev_connector; }

    auto& nextConnectors() { return _next_connectors; }

    TConnectorBaseImpl<TElem>& nextConnector(const size_t i)
    {
        return _next_connectors.at(i);
    }

    void forward()
    {
        for (auto& connector : _next_connectors) {
            connector->forward();
        }
    }

    void backward()
    {
        if (_prev_connector) {
            // TODO: calculate deltas here

            _prev_connector->backward();
        }
    }

    void connectNextConnector(std::shared_ptr<TConnectorBaseImpl<TElem>>& next)
    {
        _next_connectors.push_back(next.get());
        connectNextConnectorHandler(*next);
    }

    void connectPrevConnector(std::shared_ptr<TConnectorBaseImpl<TElem>>& prev)
    {
        if (_prev_connector) {
            throw std::invalid_argument(
                "Node already connected to a previous connector");
        }
        _prev_connector = prev;
        connectPrevConnectorHandler(*prev);
    }

    TTensor<TElem>& values() { return _values; }

    TTensor<TElem>& gradient() { return _gradient; }

    size_t shape(int i) const { return _values.shape(i); }

    TIndex shape() { return _values.shape(); }

    const TIndex shape() const { return _values.shape(); }

    template <typename TArray>
    void setDims(const TArray& arr)
    {
        _values.setDims(arr);
        _gradient.setDims(arr);
    }

    void setDims(const std::initializer_list<size_t> shape)
    {
        _values.setDims(shape);
        _gradient.setDims(shape);
    }
};

template <class TElem>
class TNode {

    friend class TNodeBaseImpl<TElem>;
    friend class TConnector<TElem>;
    friend class TConnectorBaseImpl<TElem>;

    std::shared_ptr<TNodeBaseImpl<TElem>> _impl;

public:
    TNode(const std::shared_ptr<TNodeBaseImpl<TElem>>& impl) : _impl(impl) {}

    template <typename... TArgs>
    TNode(TArgs... args)
        : TNode(std::make_shared<TNodeBaseImpl<TElem>>(args...))
    {
    }

    TNode(const std::initializer_list<size_t> list)
        : TNode(std::make_shared<TNodeBaseImpl<TElem>>(list))
    {
    }

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

    void connectNextConnector(TConnectorBaseImpl<TElem>& next)
    {
        _impl->connectNextConnector(next);
        next._impl->connectPrevNode(_impl);
    }

    void connectPrevConnector(TConnector<TElem>& prev)
    {
        _impl->connectPrevConnector(prev._impl);
        prev._impl->connectNextNode(_impl);
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