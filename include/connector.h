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

namespace snnl {

template <class TElem>
class TConnector : public std::enable_shared_from_this<TConnector<TElem>> {

    friend class TNode<TElem>;
    friend class TNode<TElem>;

protected:
    struct SNodeConnection {
        std::vector<TNodeShPtr<TElem>> input_nodes;
        TNode<TElem>*                  output_node = nullptr;
    };

    std::map<TNode<TElem>*, SNodeConnection> _node_connections;

    TConnector() = default;

    virtual void
    forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                   TNode<TElem>*                         output_node) = 0;

    virtual void
    backwardHandler(const TNode<TElem>*             output_node,
                    std::vector<TNodeShPtr<TElem>>& input_nodes) = 0;

    virtual TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const = 0;

public:
    virtual ~TConnector() {}

    TConnector(const TConnector&) = delete;

    template <template <class> class TChildConnector, typename... TArgs>
    static TNodeShPtr<TElem> apply(TArgs&&... args)
    {
        static std::shared_ptr<TChildConnector<TElem>> conn =
            TConnector<TElem>::create<TChildConnector>();
        return conn->call(args...);
    }

    template <template <class> class TChildConnector, typename... TArgs>
    static ::std::shared_ptr<TChildConnector<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<TChildConnector<TElem>>(
            new TChildConnector<TElem>(std::forward<TArgs>(args)...));
    }

    TConnectorShPtr<TElem> getPtr() { return this->shared_from_this(); }

    template <typename... TNodeShPtrs>
    TNodeShPtr<TElem> call(const TNodeShPtrs&... prev_nodes)
    {
        static_assert(sizeof...(TNodeShPtrs) > 0, "No input nodes provided");

        SNodeConnection                                       nconn;
        std::array<TNodeShPtr<TElem>, sizeof...(TNodeShPtrs)> prev_nodes_arr{
            prev_nodes...};

        nconn.input_nodes = std::vector<TNodeShPtr<TElem>>(
            prev_nodes_arr.begin(), prev_nodes_arr.end());

        TIndex            shape  = outputDims(nconn.input_nodes);
        TNodeShPtr<TElem> output = TNode<TElem>::create(shape);

        auto thisPtr      = getPtr();
        nconn.output_node = output.get();
        output->connectPrevConnector(thisPtr);

        _node_connections[nconn.output_node] = nconn;

        forwardHandler(nconn.input_nodes, nconn.output_node);

        return output;
    }

protected:
    void backward(TNode<TElem>* calling_node)
    {
        SNodeConnection& nconn = _node_connections[calling_node];

        bool need_gradient_above = false;
        for (auto& node : nconn.input_nodes) {
            need_gradient_above |= node->_needs_grad;
        }

        if (need_gradient_above) {
            // If there is no weight above, we do not neeed to compute gradients
            backwardHandler(nconn.output_node, nconn.input_nodes);
        }

        for (auto& node : nconn.input_nodes) {
            // Allways call backward, even if no weights are above.
            // This is to reset _backward_calls and _connected_nodes on each
            // node
            node->backward();
        }
    }

    void disconnect(TNode<TElem>* next_node)
    {
        SNodeConnection& nconn = _node_connections[next_node];
        for (auto node_ptr : nconn.input_nodes) {
            node_ptr->disconnect();
        }
        _node_connections.erase(next_node);
    }

    void collectWeightsInternal(TNode<TElem>* calling_node,
                                std::unordered_set<TNodeShPtr<TElem>>& weights)
    {
        auto& nconn = _node_connections[calling_node];
        for (auto& prev : nconn.input_nodes) {
            if (prev->isWeight()) {
                weights.emplace(prev);
            }
            prev->collectWeightsInternal(weights);
        }
    }

    void collectNodesInternal(TNode<TElem>* calling_node,
                              std::unordered_set<TNodeShPtr<TElem>>& nodes)
    {
        auto& nconn = _node_connections[calling_node];
        for (auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(nodes);
        }
    }

    void collectConnectorsInternal(
        TNode<TElem>*                               calling_node,
        std::unordered_set<TConnectorShPtr<TElem>>& connectors)
    {
        connectors.emplace(getPtr());
        auto& nconn = _node_connections[calling_node];
        for (auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(connectors);
        }
    }

    bool countConnectedNodesBackwards(TNode<TElem>* calling_node)
    {
        auto& nconn           = _node_connections.at(calling_node);
        bool  need_grad_above = false;

        for (auto& prev : nconn.input_nodes) {
            need_grad_above |= prev->countConnectedNodesBackwards(calling_node);
        }
        return need_grad_above;
    }
};

} // namespace snnl