#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <initializer_list>
#include <memory>
#include <set>
#include <stdexcept>

namespace snnl {

template <class TElem>
class TNode : public std::enable_shared_from_this<TNode<TElem>> {

    friend class TConnector<TElem>;
    friend class TConnector<TElem>;

    TTensor<TElem> _values;
    TTensor<TElem> _gradient;

    size_t _backward_calls = 0;

    bool _is_const  = false;
    bool _is_weight = false;

    std::vector<TConnector<TElem>*> _next_connectors = {};
    TConnector<TElem>*              _prev_connector  = nullptr;

    TConnectorShPtr<TElem> _owned_connector;

    TNode() = default;

    void collectNodesInternal(std::set<TNodeShPtr<TElem>>& nodes)
    {
        nodes.emplace(getPtr());
        if (_prev_connector) {
            _prev_connector->collectNodesInternal(this, nodes);
        }
    }

    void collectWeightsInternal(std::set<TNodeShPtr<TElem>>& weights)
    {
        if (_prev_connector) {
            _prev_connector->collectWeightsInternal(this, weights);
        }
    }

    void collectConnectorsInternal(std::set<TConnectorShPtr<TElem>>& connetors)
    {
        if (_prev_connector) {
            _prev_connector->collectNodesInternal(this, connetors);
        }
    }

public:
    template <typename... TArgs>
    static ::std::shared_ptr<TNode> create(TArgs&&... args)
    {
        return ::std::shared_ptr<TNode>(
            new TNode(::std::forward<TArgs>(args)...));
    }
    static ::std::shared_ptr<TNode> create(std::initializer_list<TElem> shape)
    {
        return ::std::shared_ptr<TNode>(new TNode(shape));
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

    size_t shapeFlattened(int i) const { return _values.shapeFlattened(i); }

    void setAllValues(const TElem& elem) { _values.setAllValues(elem); }

    void setAllGrad(const TElem& grad) { _gradient.setAllValues(grad); }

    virtual ~TNode() { std::cout << "Destroying TNode" << std::endl; }

    TConnector<TElem>& prevConnector() { return _prev_connector; }

    void forward()
    {
        // std::cout << "FORWARD on node" << std::endl;
        for (auto& connector : _next_connectors) {
            connector->forward(this);
        }
    }

    void computeGrad()
    {
        if (!_next_connectors.empty()) {
            throw std::invalid_argument(
                "Calling computeGrad on non-leave node");
        }
        _gradient.setAllValues(1);
        if (_prev_connector) {
            _prev_connector->backward(this);
        }
    }

    void backward()
    {
        // std::cout << "BACKWARD on node " << std::endl;
        _backward_calls++;
        if (_prev_connector && _backward_calls == _next_connectors.size()) {
            _prev_connector->backward(this);
            _backward_calls = 0;
        }
    }

    void zeroGrad()
    {
        iterateNodes(
            [](TNode<TElem>& node) { node.gradient().setAllValues(0); });
        iterateWeights(
            [](TNode<TElem>& weight) { weight.gradient().setAllValues(0); });
    }

    void iterateConnectors(std::function<void(TConnector<TElem>&)> func)
    {
        auto all_connectors = collectConnectors();
        for (auto& conn : all_connectors) {
            func(*conn);
        }
    }

    void iterateNodes(std::function<void(TNode<TElem>&)> func)
    {
        auto all_nodes = collectNodes();
        for (auto& node : all_nodes) {
            func(*node);
        }
    }

    void iterateWeights(std::function<void(TNode<TElem>&)> func)
    {
        auto all_weights = collectWeights();
        for (auto& weight : all_weights) {
            func(*weight);
        }
    }

    std::set<TNodeShPtr<TElem>> collectNodes()
    {
        std::set<TNodeShPtr<TElem>> out;
        collectNodesInternal(out);
        return out;
    }

    std::set<TNodeShPtr<TElem>> collectWeights()
    {
        std::set<TNodeShPtr<TElem>> out;
        collectWeightsInternal(out);
        return out;
    }

    std::set<TConnectorShPtr<TElem>> collectConnectors()
    {
        std::set<TConnectorShPtr<TElem>> out;
        collectConnectorsInternal(out);
        return out;
    }

    TTensor<TElem>& values() { return _values; }

    const TTensor<TElem>& values() const { return _values; }

    TTensor<TElem>& gradient() { return _gradient; }

    const TTensor<TElem>& gradient() const { return _gradient; }

    size_t shape(int i) const { return _values.shape(i); }

    TIndex& shape() { return _values.shape(); }

    const TIndex& shape() const { return _values.shape(); }

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

    bool isWeight() const { return _is_weight; }

    bool isConstant() const { return _is_weight; }

    bool isLeave() const { return _prev_connector == nullptr; }

    void setWeight(bool val) { _is_weight = val; }

    void setConstent(bool val) { _is_const = val; }

    size_t NDims() { return _values.NDims(); }

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

    TNodeShPtr<TElem> getPtr() { return this->shared_from_this(); }

    void connectNextConnector(TConnectorShPtr<TElem> next)
    {
        for (auto& conn : _next_connectors) {
            if (conn == next.get()) {
                // allready connected
                return;
            }
        }
        _next_connectors.push_back(next.get());

        auto RemoveCircularOwnership = [&next](TNode<TElem>& node) {
            if (node._owned_connector.get() == next.get()) {
                node._owned_connector = nullptr;
            }
        };

        iterateNodesBackwards(RemoveCircularOwnership);
    }

    void iterateNodesBackwards(std::function<void(TNode<TElem>&)> func)
    {
        _backward_calls++;
        if (_prev_connector && (_backward_calls == _next_connectors.size() ||
                                _next_connectors.empty())) {
            func(*this);
            _backward_calls = 0;
        }
        if (_prev_connector) {
            _prev_connector->iterateNodesBackwards(this, func);
        }
    }

    void connectPrevConnector(TConnectorShPtr<TElem> prev)
    {
        if (prev.get() == _prev_connector) {
            return;
        }
        if (_prev_connector) {
            throw std::invalid_argument(
                "Node already connected to a previous connector");
        }
        _prev_connector  = prev.get();
        _owned_connector = prev;
    }
};

} // namespace snnl