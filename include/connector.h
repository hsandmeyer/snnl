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

    std::vector<TNodeShPtr<TElem>> _weights;

    TConnector() = default;

    virtual void
    forwardHandler(const std::vector<TNodeShPtr<TElem>>& input_nodes,
                   const std::vector<TNodeShPtr<TElem>>& weights,
                   TNode<TElem>*                         output_node) = 0;

    virtual void
    backwardHandler(const TNode<TElem>*             output_node,
                    std::vector<TNodeShPtr<TElem>>& weights,
                    std::vector<TNodeShPtr<TElem>>& input_nodes) = 0;

    virtual TIndex
    outputDims(const std::vector<TNodeShPtr<TElem>>& input_nodes) const = 0;

public:
    TConnector(const TConnector&) = delete;

    template <template <class> class TChildConnector, typename... TArgs>
    static ::std::shared_ptr<TChildConnector<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<TChildConnector<TElem>>(
            new TChildConnector<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual ~TConnector() {}

    TConnectorShPtr<TElem> getPtr() { return this->shared_from_this(); }

    TNodeShPtr<TElem> weight(size_t i) const { return _weights.at(i); }

    std::vector<TNode<TElem>*> weights() const { return _weights; }

    template <typename... TNodeShPtrs>
    TNodeShPtr<TElem> call(const TNodeShPtrs&... prev_nodes)
    {
        SNodeConnection nconn;
        static_assert(sizeof...(TNodeShPtrs) > 0, "No input nodes provided");

        for (auto node_ptr :
             std::array<TNodeShPtr<TElem>, sizeof...(TNodeShPtrs)>{
                 prev_nodes...}) {

            nconn.input_nodes.push_back(node_ptr);
        }

        TIndex shape = outputDims(nconn.input_nodes);

        TNodeShPtr<TElem> output = TNode<TElem>::create(shape);
        nconn.output_node        = output.get();

        auto thisPtr = getPtr();
        output->connectPrevConnector(thisPtr);

        _node_connections[nconn.output_node] = nconn;

        forwardHandler(nconn.input_nodes, _weights, nconn.output_node);

        return output;
    }

protected:
    void backward(TNode<TElem>* calling_node)
    {
        SNodeConnection& nconn = _node_connections[calling_node];

        backwardHandler(nconn.output_node, _weights, nconn.input_nodes);

        for (auto& node : nconn.input_nodes) {
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
        for (auto weight : _weights) {
            weights.emplace(weight->getPtr());
        }
        for (auto& prev : nconn.input_nodes) {
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

    void countConnectedNodesBackwards(TNode<TElem>* calling_node)
    {
        auto& nconn = _node_connections.at(calling_node);

        for (auto& prev : nconn.input_nodes) {
            prev->countConnectedNodesBackwards(calling_node);
        }
    }

    TNodeShPtr<TElem>
    addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        return addWeightTensor(TIndex{shape});
    }

    template <typename TArray>
    TNodeShPtr<TElem> addWeightTensor(const TArray& shape)
    {
        TNodeShPtr<TElem> weight = TNode<TElem>::create(shape, true);

        _weights.push_back(weight);

        return weight;
    }
};

} // namespace snnl