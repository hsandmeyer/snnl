#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <cmath>
#include <cwchar>
#include <exception>
#include <initializer_list>
#include <memory>
#include <numeric>
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

    std::vector<TNodeShPtr<TElem>> _owned_nodes;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    TConnector() = default;

public:
    TConnector(const TConnector&) = delete;

    template <template <class> class TChildConnector, typename... TArgs>
    static ::std::shared_ptr<TChildConnector<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<TChildConnector<TElem>>(
            new TChildConnector<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                                const std::vector<TNode<TElem>*>& weights,
                                TNode<TElem>* output_node) = 0;

    virtual void backwardHandler(const TNode<TElem>*         output_node,
                                 std::vector<TNode<TElem>*>& weights,
                                 std::vector<TNode<TElem>*>& input_nodes) = 0;

    virtual TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const = 0;

    virtual void buildHandler(bool, const std::vector<TNode<TElem>*>&){};

    virtual ~TConnector() {}

    TNodeShPtr<TElem>
    addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        return addWeightTensor(TIndex{shape});
    }

    TConnectorShPtr<TElem> getPtr() { return this->shared_from_this(); }

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
        nconn.backward_calls++;

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

    TNode<TElem>* weight(size_t i) const { return _weights.at(i); }

    template <typename... TNodeShPtrs>
    TNodeShPtr<TElem> connect(const TNodeShPtrs&... prev_nodes)
    {
        SNodeConnection nconn;
        static_assert(sizeof...(TNodeShPtrs) > 0, "No input nodes provided");

        for (auto node_ptr :
             std::array<TNodeShPtr<TElem>, sizeof...(TNodeShPtrs)>{
                 prev_nodes...}) {

            _owned_nodes.push_back(node_ptr);

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

    virtual void
    buildHandler(bool                              was_build_before,
                 const std::vector<TNode<TElem>*>& input_nodes) override
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

    void dimChecks(const std::vector<TNode<TElem>*>& input_nodes) const
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

    TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const override
    {
        dimChecks(input_nodes);

        TIndex out_shape = input_nodes.front()->shape();
        out_shape[-1]    = _output_units;
        return out_shape;
    }

    void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                        const std::vector<TNode<TElem>*>& weights,
                        TNode<TElem>*                     output_node) override
    {

        std::cout << "FORWARD on dense layer" << std::endl;
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

    void backwardHandler(const TNode<TElem>*, std::vector<TNode<TElem>*>&,
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
    TTensor<TElem>& W()
    {
        if (!_W) {
            throw std::runtime_error(
                "Weights for dense layer not initialized. Connect layer first");
        }
        return _W->values();
    }

    TTensor<TElem>& B()
    {
        if (!_B) {
            throw std::runtime_error(
                "Weights for dense layer not initialized. Connect layer first");
        }
        return _B->values();
    }
    virtual ~TDenseConnector()
    {
        std::cout << "Destroying Dense Connector" << std::endl;
    }
};

template <class TElem>
class TAddConnector : public TConnector<TElem> {
public:
    virtual ~TAddConnector()
    {
        std::cout << "Destroying Add connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const override
    {
        TIndex out_shape = input_nodes.front()->shape();

        for (auto node_ptr : input_nodes) {
            if (node_ptr->shape() != out_shape) {
                throw("Add connector can only work on tensors of exakt same "
                      "shape");
            }
        }
        return out_shape;
    }

    void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                        const std::vector<TNode<TElem>*>&,
                        TNode<TElem>* output_node) override
    {
        std::cout << "FORWARD on Add layer" << std::endl;
        output_node->setAllValues(0);
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                output_node->value(ind) += input_node_ptr->value(ind);
            }
        }
    }

    void backwardHandler(const TNode<TElem>* output_node,
                         std::vector<TNode<TElem>*>&,
                         std::vector<TNode<TElem>*>& input_nodes) override
    {
        for (auto& input_node_ptr : input_nodes) {
            for (size_t ind = 0; ind < output_node->values().shapeFlattened(-1);
                 ind++) {
                input_node_ptr->grad(ind) += output_node->grad(ind);
            }
        }
    }
};

template <class TElem>
class TSumConnector : public TConnector<TElem> {
public:
    virtual ~TSumConnector()
    {
        std::cout << "Destroying Sum connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for sum connector");
        }
        return TIndex{1};
    }

    void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                        const std::vector<TNode<TElem>*>&,
                        TNode<TElem>* output_node) override
    {
        std::cout << "FORWARD on Sum layer" << std::endl;

        output_node->value(0) = 0;
        output_node->value(0) += std::accumulate(
            input_nodes.front()->values().begin(),
            input_nodes.front()->values().end(), static_cast<TElem>(0));
    }

    void backwardHandler(const TNode<TElem>* output_node,
                         std::vector<TNode<TElem>*>&,
                         std::vector<TNode<TElem>*>& input_nodes) override
    {
        TElem output_grad = output_node->gradient()(0);
        for (auto& val : input_nodes.front()->gradient()) {
            val += output_grad;
        }
    }
};

template <class TElem>
class TSigmoidConnector : public TConnector<TElem> {
public:
    virtual ~TSigmoidConnector()
    {
        std::cout << "Destroying Sigmoid connector" << std::endl;
    }

    TIndex
    outputDims(const std::vector<TNode<TElem>*>& input_nodes) const override
    {
        if (input_nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Sigmoid connector");
        }
        return input_nodes.front()->shape();
    }

    void forwardHandler(const std::vector<TNode<TElem>*>& input_nodes,
                        const std::vector<TNode<TElem>*>&,
                        TNode<TElem>* output_node) override
    {
        std::cout << "FORWARD on Sigmoid layer" << std::endl;
        TNode<TElem>* input_node = input_nodes.front();
        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            output_node->value(ind) =
                static_cast<TElem>(1) /
                (static_cast<TElem>(1) + std::exp(-input_node->value(ind)));
        }
    }

    void backwardHandler(const TNode<TElem>* output_node,
                         std::vector<TNode<TElem>*>&,
                         std::vector<TNode<TElem>*>& input_nodes) override
    {
        TNode<TElem>* input_node = input_nodes.front();

        for (size_t ind = 0; ind < output_node->shapeFlattened(-1); ind++) {
            TElem output_value = output_node->value(ind);
            TElem output_grad  = output_node->grad(ind);
            TElem tmp          = std::exp(-output_value) + 1;
            input_node->grad(ind) +=
                std::exp(-output_value) / (tmp * tmp) * output_grad;
        }
    }
};

} // namespace snnl