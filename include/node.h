#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace snnl
{

template<class TElem>
class Node : public std::enable_shared_from_this<Node<TElem>>
{

    friend class Connector<TElem>;
    friend class Connector<TElem>;

    Tensor<TElem> _values;
    Tensor<TElem> _gradient;

    bool _is_weight = false;

    ConnectorShPtr<TElem> _prev_connector = nullptr;

    // Extra variables for tracking the backward call.
    // Connected nodes from the last forward call
    std::unordered_set<Node<TElem>*> _connected_nodes = {};
    bool                             _needs_grad      = false;
    size_t                           _backward_calls  = 0;

    Node() = default;

    void collectNodesInternal(std::unordered_set<NodeShPtr<TElem>>& nodes)
    {
        nodes.emplace(getPtr());
        if(_prev_connector) {
            _prev_connector->collectNodesInternal(this, nodes);
        }
    }

    void collectWeightsInternal(std::unordered_set<NodeShPtr<TElem>>& weights)
    {
        if(_prev_connector) {
            _prev_connector->collectWeightsInternal(this, weights);
        }
    }

    void collectConnectorsInternal(std::unordered_set<ConnectorShPtr<TElem>>& connetors)
    {
        if(_prev_connector) {
            _prev_connector->collectNodesInternal(this, connetors);
        }
    }

public:
    template<typename... TArgs>
    static ::std::shared_ptr<Node> create(TArgs&&... args)
    {
        return ::std::shared_ptr<Node>(new Node(std::forward<TArgs>(args)...));
    }

    static ::std::shared_ptr<Node> create(const std::initializer_list<size_t> shape)
    {
        return ::std::shared_ptr<Node>(new Node(shape));
    }

    template<typename... Args>
    TElem& value(const Args... args)
    {
        return _values(args...);
    }

    template<typename... Args>
    const TElem& value(const Args... args) const
    {
        return _values(args...);
    }

    template<typename... Args>
    const TElem& grad(const Args... args) const
    {
        return _gradient(args...);
    }

    template<typename... Args>
    TElem& grad(const Args... args)
    {
        return _gradient(args...);
    }

    size_t shapeFlattened(int i) const { return _values.shapeFlattened(i); }

    void setAllValues(const TElem& elem) { _values.setAllValues(elem); }

    void setAllGrad(const TElem& grad) { _gradient.setAllValues(grad); }

    Connector<TElem>* prevConnector() { return _prev_connector; }

    virtual ~Node() { disconnect(); }

    void computeGrad()
    {
        needsGradAbove();

        _gradient.setAllValues(1);

        if(_prev_connector) {
            _prev_connector->backward(this);
        }
    }

    void zeroGrad()
    {
        iterateNodes([](Node<TElem>& node) {
            node.gradient().setAllValues(0);
        });
        iterateWeights([](Node<TElem>& weight) {
            weight.gradient().setAllValues(0);
        });
    }

    void iterateConnectors(std::function<void(Connector<TElem>&)> func)
    {
        auto all_connectors = collectConnectors();
        for(auto& conn : all_connectors) {
            func(*conn);
        }
    }

    void iterateNodes(std::function<void(Node<TElem>&)> func)
    {
        auto all_nodes = collectNodes();
        for(auto& node : all_nodes) {
            func(*node);
        }
    }

    void iterateWeights(std::function<void(Node<TElem>&)> func)
    {
        auto all_weights = collectWeights();
        for(auto& weight : all_weights) {
            func(*weight);
        }
    }

    std::unordered_set<NodeShPtr<TElem>> collectNodes()
    {
        std::unordered_set<NodeShPtr<TElem>> out;
        collectNodesInternal(out);
        return out;
    }

    std::unordered_set<NodeShPtr<TElem>> collectWeights()
    {
        std::unordered_set<NodeShPtr<TElem>> out;
        collectWeightsInternal(out);
        return out;
    }

    std::unordered_set<ConnectorShPtr<TElem>> collectConnectors()
    {
        std::unordered_set<ConnectorShPtr<TElem>> out;
        collectConnectorsInternal(out);
        return out;
    }

    Tensor<TElem>& values() { return _values; }

    const Tensor<TElem>& values() const { return _values; }

    Tensor<TElem>& gradient() { return _gradient; }

    const Tensor<TElem>& gradient() const { return _gradient; }

    size_t shape(int i) const { return _values.shape(i); }

    Index& shape() { return _values.shape(); }

    bool isScalar() const { return _values.isScalar(); }

    const Index& shape() const { return _values.shape(); }

    template<typename TArray>
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

    bool isLeave() const { return _prev_connector == nullptr; }

    void setWeight(bool val) { _is_weight = val; }

    size_t NDims() { return _values.NDims(); }

    void disconnect()
    {
        if(_prev_connector) {
            _prev_connector->disconnect(this);
            _prev_connector = nullptr;
        }
    }

    NodeShPtr<TElem> getPtr() { return this->shared_from_this(); }

protected:
    Node(const Node&) = delete;

    const Node& operator=(const Node&) = delete;

    template<typename TArray>
    Node(const TArray& shape, bool is_weight = false)
        : _is_weight(is_weight)
    {
        setDims(shape);
    }

    Node(const std::initializer_list<size_t> shape)
        : Node(std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    void backward()
    {
        _backward_calls++;
        if(_prev_connector) {
            if(_backward_calls == _connected_nodes.size()) {
                _prev_connector->backward(this);
                _backward_calls = 0;
                _connected_nodes.clear();
            }
        }
        else {
            _backward_calls = 0;
            _connected_nodes.clear();
        }
    }

    bool needsGradAbove(Node<TElem>* next_node = nullptr)
    {
        if(next_node) {
            if(_connected_nodes.find(next_node) != _connected_nodes.end()) {
                // Already came along this edge. Stop here
                return _needs_grad;
            }
            if(_connected_nodes.empty()) {
                _gradient.setAllValues(static_cast<TElem>(0));
            }
            _connected_nodes.emplace(next_node);
        }

        bool needs_grad = _is_weight;

        if(_prev_connector) {
            needs_grad |= _prev_connector->needsGradAbove(this);
        }

        _needs_grad = needs_grad;
        return needs_grad;
    }

    void connectPrevConnector(ConnectorShPtr<TElem>& prev)
    {
        if(prev.get() == _prev_connector.get()) {
            return;
        }
        if(_prev_connector) {
            throw std::invalid_argument("Node already connected to a previous connector");
        }
        _prev_connector = prev;
    }
};

} // namespace snnl