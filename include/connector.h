#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cwchar>
#include <exception>
#include <gtest/internal/gtest-port.h>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace snnl {

template <class TElem>
class TConnector : public std::enable_shared_from_this<TConnector<TElem>> {

    friend class TNode<TElem>;
    friend class TNode<TElem>;

protected:
    struct SNodeConnection {
        std::vector<TNode<TElem>*> input_nodes;
        TNode<TElem>*              output_node   = nullptr;
        size_t                     forward_calls = 0;
    };

    std::vector<SNodeConnection> _node_connections;

    std::vector<TNode<TElem>*> _weights;

    std::vector<TNodeShPtr<TElem>> _owned_nodes;

    size_t _num_backward_calls = 0;

    bool _was_build = false;

    TConnector() = default;

    virtual void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                                const std::vector<TNode<TElem>*>& weights,
                                TNode<TElem>* output_node) = 0;

    virtual void backwardHandler(const TNode<TElem>*         output_node,
                                 std::vector<TNode<TElem>*>& weights,
                                 std::vector<TNode<TElem>*>& input_nodes) = 0;

    virtual TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const = 0;

    virtual void buildHandler(bool, const std::vector<TNode<TElem>*>&){};

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

    TNode<TElem>* weight(size_t i) const { return _weights.at(i); }

    std::vector<TNode<TElem>*> weights() const { return _weights; }

    template <typename... TNodeShPtrs>
    TNodeShPtr<TElem> connect(const TNodeShPtrs&... prev_nodes)
    {
        SNodeConnection nconn;
        static_assert(sizeof...(TNodeShPtrs) > 0, "No input nodes provided");

        for (auto node_ptr :
             std::array<TNodeShPtr<TElem>, sizeof...(TNodeShPtrs)>{
                 prev_nodes...}) {

            if (std::find_if(_owned_nodes.begin(), _owned_nodes.end(),
                             [&node_ptr](auto& elem) {
                                 return node_ptr.get() == elem.get();
                             }) == _owned_nodes.end()) {
                _owned_nodes.push_back(node_ptr);
            }

            nconn.input_nodes.push_back(node_ptr.get());
            node_ptr->connectNextConnector(getPtr());
        }

        build(nconn);

        TIndex shape = outputDims(nconn.input_nodes);

        TNodeShPtr<TElem> output = TNode<TElem>::create(shape);
        nconn.output_node        = output.get();
        output->connectPrevConnector(getPtr());
        _node_connections.push_back(nconn);

        return output;
    }

protected:
    void forward(const TNode<TElem>* calling_node)
    {

        auto connections = getForwardNodeConnections(calling_node);

        for (SNodeConnection* nconn : connections) {
#ifdef DEBUG
            if (!nconn->output_node) {
                throw std::runtime_error("Output node is invalid");
            }
#endif

            nconn->forward_calls++;

            if (nconn->forward_calls == numCallingPrevNodes(*nconn)) {

                nconn->output_node->setDims(outputDims(nconn->input_nodes));

                forwardHandler(nconn->input_nodes, _weights,
                               nconn->output_node);
                nconn->output_node->forward();
                nconn->forward_calls = 0;
            }
        }
    }

    void backward(const TNode<TElem>* calling_node)
    {
        SNodeConnection& nconn = getBackwardNodeConnection(calling_node);

        backwardHandler(nconn.output_node, _weights, nconn.input_nodes);

        for (auto& node : nconn.input_nodes) {
            node->backward();
        }
    }

    void build(const SNodeConnection& nconn)
    {
        if (!_was_build) {
            buildHandler(_was_build, nconn.input_nodes);
        }
        _was_build = true;
    }

    void collectWeightsInternal(TNode<TElem>*                calling_node,
                                std::set<TNodeShPtr<TElem>>& weights)
    {
        auto nconn = getBackwardNodeConnection(calling_node);
        for (auto weight : _weights) {
            weights.emplace(weight->getPtr());
        }
        for (auto& prev : nconn.input_nodes) {
            prev->collectWeightsInternal(weights);
        }
    }

    void collectNodesInternal(TNode<TElem>*                calling_node,
                              std::set<TNodeShPtr<TElem>>& nodes)
    {
        auto nconn = getBackwardNodeConnection(calling_node);
        for (auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(nodes);
        }
    }

    void collectConnectorsInternal(TNode<TElem>* calling_node,
                                   std::set<TConnectorShPtr<TElem>>& connectors)
    {
        connectors.emplace(getPtr());
        auto nconn = getBackwardNodeConnection(calling_node);
        for (auto& prev : nconn.input_nodes) {
            prev->collectNodesInternal(connectors);
        }
    }

    void iterateNodesBackwards(TNode<TElem>*                      calling_node,
                               std::function<void(TNode<TElem>&)> func)
    {
        auto nconn = getBackwardNodeConnection(calling_node);
        for (auto& prev : nconn.input_nodes) {
            prev->iterateNodesBackwards(func);
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

        _owned_nodes.push_back(weight);
        _weights.push_back(weight.get());

        weight->connectNextConnector(getPtr());

        return weight;
    }

    size_t numCallingPrevNodes(const SNodeConnection& nodeconnection) const
    {
        size_t ret = 0;
        for (auto& node_ptr : nodeconnection.input_nodes) {
            if (node_ptr->isWeight() || node_ptr->isConstant()) {
                continue;
            }
            ret++;
        }
        return ret;
    }

    SNodeConnection& getBackwardNodeConnection(const TNode<TElem>* calling_node)
    {
        for (auto& node_connection : _node_connections) {
            if (node_connection.output_node == calling_node) {
                return node_connection;
            }
        }
        throw std::out_of_range(
            "Did not find backward calling node in any node connection");
    }

    std::vector<SNodeConnection*>
    getForwardNodeConnections(const TNode<TElem>* calling_node)
    {
        std::vector<SNodeConnection*> out;
        for (auto& node_connection : _node_connections) {
            for (auto node_ptr : node_connection.input_nodes) {
                if (node_ptr == calling_node) {
                    out.push_back(&node_connection);
                }
            }
        }
        if (out.empty()) {
            throw std::invalid_argument(
                "Did not find forward calling node in any node connection");
        }
        return out;
    }
};

} // namespace snnl