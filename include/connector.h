#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cwchar>
#include <exception>
#include <gtest/internal/gtest-port.h>
#include <initializer_list>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace snnl
{

template<class TElem>
class Connector : public std::enable_shared_from_this<Connector<TElem>>
{

    friend class Node<TElem>;

protected:
    struct SNodeConnection
    {
        std::vector<NodeShPtr<TElem>> input_nodes;
        Node<TElem>*                  output_node = nullptr;
    };

    std::map<Node<TElem>*, SNodeConnection> _node_connections;

    Connector() = default;

    virtual void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                                Node<TElem>*                         output_node) = 0;

    virtual void backwardHandler(const Node<TElem>*             output_node,
                                 std::vector<NodeShPtr<TElem>>& input_nodes) = 0;

    virtual Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const = 0;

public:
    virtual ~Connector() {}

    Connector(const Connector&) = delete;

    template<template<class> class ChildConnector, typename... TArgs>
    static NodeShPtr<TElem> apply(TArgs&&... args)
    {
        std::shared_ptr<ChildConnector<TElem>> conn = Connector<TElem>::create<ChildConnector>();
        return conn->call(args...);
    }

    template<template<class> class ChildConnector, typename... TArgs>
    static ::std::shared_ptr<ChildConnector<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<ChildConnector<TElem>>(
            new ChildConnector<TElem>(std::forward<TArgs>(args)...));
    }

    ConnectorShPtr<TElem> getPtr() { return this->shared_from_this(); }

    template<typename... NodeShPtrs>
    NodeShPtr<TElem> call(const NodeShPtrs&... prev_nodes)
    {
        static_assert(sizeof...(NodeShPtrs) > 0, "No input nodes provided");

        SNodeConnection                                     nconn;
        std::array<NodeShPtr<TElem>, sizeof...(NodeShPtrs)> prev_nodes_arr{prev_nodes...};

        nconn.input_nodes =
            std::vector<NodeShPtr<TElem>>(prev_nodes_arr.begin(), prev_nodes_arr.end());

        Index            shape  = outputDims(nconn.input_nodes);
        NodeShPtr<TElem> output = Node<TElem>::create(shape);

        auto thisPtr      = getPtr();
        nconn.output_node = output.get();
        output->connectPrevConnector(thisPtr);

        _node_connections[nconn.output_node] = nconn;

        forwardHandler(nconn.input_nodes, nconn.output_node);

        return output;
    }

protected:
    void backward(Node<TElem>* calling_node)
    {
        SNodeConnection& nconn = _node_connections[calling_node];

        bool need_gradient_above = false;
        for(auto& node : nconn.input_nodes) {
            need_gradient_above |= node->_needs_grad;
        }

        if(need_gradient_above) {
            // If there is no weight above, we do not neeed to compute gradients
            backwardHandler(nconn.output_node, nconn.input_nodes);
        }

        for(auto& node : nconn.input_nodes) {
            // Allways call backward, even if no weights are above.
            // This is to reset _backward_calls and _connected_nodes on each
            // node
            node->backward();
        }
    }

    void disconnect(Node<TElem>* next_node)
    {
        SNodeConnection& nconn = _node_connections[next_node];
        for(auto node_ptr : nconn.input_nodes) {
            node_ptr->disconnect();
        }
        _node_connections.erase(next_node);
    }

    void collectWeightsInternal(Node<TElem>*                          calling_node,
                                std::unordered_set<NodeShPtr<TElem>>& weights)
    {
        auto& nconn = _node_connections[calling_node];
        for(auto& prev : nconn.input_nodes) {
            if(prev->isWeight()) {
                weights.emplace(prev);
            }
            prev->collectWeightsInternal(weights);
        }
    }

    void collectNodesInternal(Node<TElem>*                          calling_node,
                              std::unordered_set<NodeShPtr<TElem>>& nodes)
    {
        auto& nconn = _node_connections[calling_node];
        for(auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(nodes);
        }
    }

    void collectConnectorsInternal(Node<TElem>*                               calling_node,
                                   std::unordered_set<ConnectorShPtr<TElem>>& connectors)
    {
        connectors.emplace(getPtr());
        auto& nconn = _node_connections[calling_node];
        for(auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(connectors);
        }
    }

    bool needsGradAbove(Node<TElem>* calling_node)
    {
        auto& nconn           = _node_connections.at(calling_node);
        bool  need_grad_above = false;

        for(auto& prev : nconn.input_nodes) {
            need_grad_above |= prev->needsGradAbove(calling_node);
        }
        return need_grad_above;
    }
};

} // namespace snnl