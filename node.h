#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <memory>

namespace snnl {

template <class TElem>
class TNode : public std::enable_shared_from_this<TNode<TElem>> {

    friend class TConnector<TElem>;
    friend class TConnector<TElem>;

    TTensor<TElem> _values;
    TTensor<TElem> _gradient;

    size_t _backward_calls;

    bool _is_const  = false;
    bool _is_weight = false;

    std::vector<TConnector<TElem>*> _next_connectors = {};
    TConnectorPtr<TElem>            _prev_connector  = nullptr;

    TNode() = default;

public:
    template <typename... TArgs>
    static ::std::shared_ptr<TNode> create(TArgs&&... args)
    {
        return ::std::shared_ptr<TNode>(
            new TNode(::std::forward<TArgs>(args)...));
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

    virtual ~TNode() { std::cout << "Destroying TNode" << std::endl; }

    TConnector<TElem>& prevConnector() { return _prev_connector; }

    void forward()
    {
        for (auto& connector : _next_connectors) {
            connector->forward(this);
        }
    }

    void backward()
    {
        _backward_calls++;
        if (_prev_connector && _backward_calls == _next_connectors.size()) {
            _prev_connector->backward(this);
        }
        _backward_calls = 0;
    }

    void connectNextConnector(TConnectorPtr<TElem> next)
    {
        for (auto& conn : _next_connectors) {
            if (conn == next.get()) {
                // allready connected
                return;
            }
        }
        _next_connectors.push_back(next.get());
    }

    void connectPrevConnector(TConnectorPtr<TElem> prev)
    {
        if (prev.get() == _prev_connector.get()) {
            return;
        }
        if (_prev_connector) {
            throw std::invalid_argument(
                "Node already connected to a previous connector");
        }
        _prev_connector = prev;
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

    bool isWeight() { return _is_weight; }

    bool isConstant() { return _is_weight; }

    bool isLeave() { return _prev_connector == nullptr; }

    void setWeight(bool val) { _is_weight = val; }

    void setConstent(bool val) { _is_const = val; }

protected:
    TNode(const TNode&) = delete;

    const TNode& operator=(const TNode&) = delete;

    template <typename TArray>
    TNode(const TArray& shape, bool is_weight = false, bool is_const = false)
        : _is_const(is_const), _is_weight(is_weight)
    {
        setDims(shape);
    }

    TNode(const std::initializer_list<size_t> shape)
        : TNode(std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    TNodePtr<TElem> getPtr() { return this->shared_from_this(); }
};

} // namespace snnl