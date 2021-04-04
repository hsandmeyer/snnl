#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <algorithm>
#include <cwchar>
#include <exception>
#include <initializer_list>
#include <memory>
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
        TNode<TElem>*              output_node    = nullptr;
        size_t                     forward_calls  = 0;
        size_t                     backward_calls = 0;
    };

    std::vector<SNodeConnection> _node_connections;

    std::vector<TNode<TElem>*> _weights;

    std::vector<TNodePtr<TElem>> _owned_nodes;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    TConnector() = default;

public:
    TConnector(const TConnector&) = delete;

    template <template <class> class TChildConnector, typename... TArgs>
    static ::std::shared_ptr<TConnector> create(TArgs&&... args)
    {
        return ::std::shared_ptr<TChildConnector<TElem>>(
            new TChildConnector<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual void forwardHandler(std::vector<TNode<TElem>*>& input_nodes,
                                TNode<TElem>*               output_node,
                                std::vector<TNode<TElem>*>& weights) = 0;

    virtual void backwardHandler(TNode<TElem>*               output_node,
                                 std::vector<TNode<TElem>*>& input_nodes,
                                 std::vector<TNode<TElem>*>& weights) = 0;

    virtual TIndex outputDims(std::vector<TNode<TElem>*>& input_nodes) = 0;

    virtual void buildHandler(bool was_build_before,
                              std::vector<TNode<TElem>*>&) = 0;

    virtual ~TConnector() {}

    TNodePtr<TElem> addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        return addWeightTensor(TIndex{shape});
    }

    TConnectorPtr<TElem> getPtr() { return this->shared_from_this(); }

    template <typename TArray>
    TNodePtr<TElem> addWeightTensor(const TArray& shape)
    {
        TNodePtr<TElem> weight = TNode<TElem>::create(shape, true);

        _owned_nodes.push_back(weight);
        _weights.push_back(weight.get());

        weight->connectNextConnector(getPtr());

        return weight;
    }

    size_t numCallingPrevNodes(SNodeConnection& nodeconnection)
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

    SNodeConnection& getBackwardNodeConnection(TNode<TElem>* calling_node)
    {
        for (auto& node_connection : _node_connections) {
            if (node_connection.output_node == calling_node) {
                return node_connection;
            }
        }
        throw std::out_of_range(
            "Did not find backward calling node in any node connection");
    }

    SNodeConnection& getForwardNodeConnection(TNode<TElem>* calling_node)
    {
        for (auto& node_connection : _node_connections) {
            for (auto node_ptr : node_connection.input_nodes) {
                if (node_ptr == calling_node) {
                    return node_connection;
                }
            }
        }
        throw std::out_of_range(
            "Did not find forward calling node in any node connection");
    }

    void forward(TNode<TElem>* calling_node)
    {

        SNodeConnection& nconn = getForwardNodeConnection(calling_node);
#ifdef DEBUG
        if (!nconn.output_node) {
            throw std::runtime_error("Output node is invalid");
        }
#endif

        nconn.forward_calls++;

        if (nconn.forward_calls == numCallingPrevNodes(nconn)) {

            nconn.output_node->setDims(outputDims(nconn.input_nodes));

            forwardHandler(nconn.input_nodes, nconn.output_node, _weights);
            nconn.output_node->forward();
            nconn.forward_calls = 0;
        }
    }

    void backward(TNode<TElem>* calling_node)
    {
        SNodeConnection& nconn = getBackwardNodeConnection(calling_node);
        nconn.backward_calls++;

        backwardHandler(nconn.output_node, nconn.input_nodes, _weights);

        for (auto& node : nconn.input_nodes) {
            node->backward();
        }
    }

    void build(SNodeConnection& nconn)
    {
        if (!_was_build) {
            buildHandler(_was_build, nconn.input_nodes);
        }
        _was_build = true;
    }

    TNode<TElem>* weight(size_t i) { return _weights.at(i); }

    template <typename... TNodePtrs>
    TNodePtr<TElem> connect(const TNodePtrs&... prev_nodes)
    {
        SNodeConnection nconn;
        static_assert(sizeof...(TNodePtrs) > 0, "No input nodes provided");

        for (auto node_ptr :
             std::array<TNodePtr<TElem>, sizeof...(TNodePtrs)>{prev_nodes...}) {

            _owned_nodes.push_back(node_ptr);

            nconn.input_nodes.push_back(node_ptr.get());

            node_ptr->connectNextConnector(getPtr());
        }

        build(nconn);

        TIndex shape = outputDims(nconn.input_nodes);

        TNodePtr<TElem> output = TNode<TElem>::create(shape);
        nconn.output_node      = output.get();
        output->connectPrevConnector(getPtr());
        _node_connections.push_back(nconn);

        return output;
    }
};

template <class TElem>
class TDenseConnector : public TConnector<TElem> {

    friend class TConnector<TElem>;

    TNode<TElem>* _W;
    TNode<TElem>* _B;

    std::vector<TNode<TElem>*> _inputs;
    std::vector<TNode<TElem>*> _outputs;

    size_t _input_units = -1;
    size_t _output_units;

    virtual void buildHandler(bool                        was_build_before,
                              std::vector<TNode<TElem>*>& input_nodes) override
    {

        if (!was_build_before) {
            if (_input_units == -1ul) {
                _input_units = input_nodes.front()->shape(-1);
            }
            this->addWeightTensor({_output_units, _input_units});
            this->addWeightTensor({_output_units});

            _W = this->weight(0);
            _B = this->weight(1);
        }
        else {
            dimChecks(input_nodes);
        }
    }

    void dimChecks(std::vector<TNode<TElem>*>& input_nodes)
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one input node per call for dense layer");
        }
        if (input_nodes.empty()) {
            throw std::invalid_argument("No input nodes provided");
        }
        if (_input_units != input_nodes.front()->shape(-1)) {
            throw std::invalid_argument(
                "Output dimsion of previous node (" +
                std::to_string(input_nodes.front()->shape(-1)) +
                ") != " + std::to_string(_input_units));
        }
    }

    TIndex outputDims(std::vector<TNode<TElem>*>& input_nodes) override
    {
        dimChecks(input_nodes);

        TIndex out_shape = input_nodes.front()->shape();
        out_shape[-1]    = _output_units;
        return out_shape;
    }

    void forwardHandler(std::vector<TNode<TElem>*>& input_nodes,
                        TNode<TElem>*               output_node,
                        std::vector<TNode<TElem>*>& weights) override
    {

        dimChecks(input_nodes);

        auto& input  = input_nodes.front()->values();
        auto& output = output_node->values();

        TNode<TElem>& W = *weights.at(0);
        TNode<TElem>& B = *weights.at(1);

        if (input.NDims() > 1) {
            // TODO Generalize this using a better tensor loop with some
            // kind of ellipsis object
            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = B.value(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(higherDim, i) +=
                            W.value(i, j) * input(higherDim, j);
                    }
                }
            }
            /*
            output.forEach([&](const TIndex& ind_in) {
                int i          = ind_in[-1];
                output(ind_in) = _B->value(i);
                auto ind_out   = ind_in;
                for (size_t j = 0; j < input.shape(-1); j++) {
                    ind_out[-1] = j;
                    output(ind_in) += _W->value(i, j) * input(ind_out);
                }
            });
            */
        }
        else {
            for (size_t i = 0; i < output.shape(-1); i++) {
                output(i) = B.value(i);
                for (size_t j = 0; j < input.shape(-1); j++) {
                    output(i) += W.value(i, j) * input(j);
                }
            }
        }
    }

    void backwardHandler(TNode<TElem>*, std::vector<TNode<TElem>*>&,
                         std::vector<TNode<TElem>*>&) override
    {
        // TODO: //Calculate dfdw and dfdz here
    }

    TDenseConnector(size_t output_dim) : _output_units(output_dim) {}

    TDenseConnector(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {
    }

public:
    virtual ~TDenseConnector()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

} // namespace snnl