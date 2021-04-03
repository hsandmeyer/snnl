#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <algorithm>
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
    std::vector<TNode<TElem>*>   _next_nodes;
    std::vector<TNodePtr<TElem>> _prev_nodes;
    std::vector<TNode<TElem>*>   _weights;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    virtual void connectNextNodeHandler(TNode<TElem>&){};

    virtual void connectPrevNodeHandler(TNode<TElem>&){};

    virtual void connectWeightNodeHandler(TNode<TElem>&){};

    TConnector() = default;

public:
    TConnector(const TConnector&) = delete;

    template <template <class> class TChildConnector, typename... TArgs>
    static ::std::shared_ptr<TConnector> create(TArgs&&... args)
    {
        return ::std::shared_ptr<TChildConnector<TElem>>(
            new TChildConnector<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual TNodePtr<TElem> createOutputNode(std::vector<TNodePtr<TElem>>&)
    {
        return TNode<TElem>::create();
    };

    virtual void forwardHandler() = 0;

    virtual void buildHandler(bool was_build_before) = 0;

    virtual void backwardHandler() = 0;

    virtual ~TConnector() {}

    auto& prevNodes() { return _prev_nodes; }

    auto& nextNodes() { return _next_nodes; }

    TNodePtr<TElem> addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        return addWeightTensor(TIndex{shape});
    }

    TConnectorPtr<TElem> getPtr() { return this->shared_from_this(); }

    template <typename TArray>
    TNodePtr<TElem> addWeightTensor(const TArray& shape)
    {
        TNodePtr<TElem> weight = TNode<TElem>::create(shape, true);

        _prev_nodes.push_back(weight);
        connectWeightNodeHandler(*weight);

        _weights.push_back(weight.get());
        return weight;
    }

    size_t numCallingPrevNodes()
    {
        size_t ret = 0;
        for (auto& node_ptr : _prev_nodes) {
            if (node_ptr->isWeight() || node_ptr->isConstant()) {
                continue;
            }
            ret++;
        }
        return ret;
    }

    void forward()
    {
        _prev_ready++;
        if (_prev_ready == numCallingPrevNodes()) {

            forwardHandler();

            for (auto& node : _next_nodes) {
                node->forward();
            }
            _prev_ready = 0;
        }
    }

    void backward()
    {
        _next_ready++;
        if (_next_ready == _next_nodes.size()) {

            backwardHandler();

            for (auto& node : _prev_nodes) {
                node->backward();
            }

            _next_ready = 0;
        }
    }

    void build()
    {
        if (!_was_build) {
            buildHandler(_was_build);
        }
        _was_build = true;
    }

    void connectNextNode(const TNodePtr<TElem>& next_node)
    {
        if (!_next_nodes.empty() && _next_nodes.back() == next_node.get()) {
            return;
        }
        _next_nodes.push_back(next_node.get());
        connectNextNodeHandler(*next_node);
        next_node->connectPrevConnector(getPtr());
    }

    void connectPrevNode(const TNodePtr<TElem>& prev_node)
    {
        if (!_prev_nodes.empty() &&
            _prev_nodes.back().get() == prev_node.get()) {
            return;
        }
        _prev_nodes.push_back(prev_node);
        connectPrevNodeHandler(*prev_node);
        prev_node->connectNextConnector(getPtr());
    }

    TNode<TElem>& nextNode(const size_t i) { return *_next_nodes.at(i); }

    TNode<TElem>& prevNode(const size_t i) { return *_prev_nodes.at(i); }

    TNode<TElem>* weight(size_t i) { return _weights.at(i); }

    template <typename... TNodePtrs>
    TNodePtr<TElem> connect(const TNodePtrs&... prev)
    {
        std::vector<TNodePtr<TElem>> nodes{prev...};

        for (auto& node : nodes) {
            connectPrevNode(node);
        }

        TNodePtr<TElem> out = createOutputNode(nodes);

        connectNextNode(out);
        build();

        return out;
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

    virtual void connectPrevNodeHandler(TNode<TElem>& prev) override
    {
        _inputs.push_back(&prev);
    }

    virtual void connectNextNodeHandler(TNode<TElem>& prev) override
    {
        _outputs.push_back(&prev);
    }

    virtual TNodePtr<TElem>
    createOutputNode(std::vector<TNodePtr<TElem>>& nodes) override
    {
        if (nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Dense Connector");
        }

        return TNode<TElem>::create();
    }

    virtual void buildHandler(bool was_build_before) override
    {

        if (!was_build_before) {
            if (_input_units == -1ul) {
                _input_units = this->_prev_nodes.at(0)->shape(-1);
            }
            this->addWeightTensor({_output_units, _input_units});
            this->addWeightTensor({_output_units});
        }

        _W = this->weight(0);
        _B = this->weight(1);
    }

    void dimChecks(TTensor<TElem>& input)
    {
        if (_input_units != input.shape(-1)) {
            throw std::invalid_argument("Output dimsion of previous node (" +
                                        std::to_string(input.shape(-1)) +
                                        ") != " + std::to_string(_input_units));
        }
    }

    void forwardHandler() override
    {

        for (size_t node_ind = 0; node_ind < this->_inputs.size(); node_ind++) {

            auto& output = _outputs[node_ind]->values();
            auto& input  = _inputs[node_ind]->values();

            dimChecks(input);

            TIndex out_shape = input.shape();
            out_shape[-1]    = _output_units;
            output.setDims(out_shape);

            if (out_shape.NDims() > 1) {
                // TODO Generalize this using a better tensor loop with some
                // kind of ellipsis object
                for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                     higherDim++) {
                    for (size_t i = 0; i < output.shape(-1); i++) {
                        output(higherDim, i) = _B->value(i);
                        for (size_t j = 0; j < input.shape(-1); j++) {
                            output(higherDim, i) +=
                                _W->value(i, j) * input(higherDim, j);
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
                    output(i) = _B->value(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(i) += _W->value(i, j) * input(j);
                    }
                }
            }
        }
    }

    void backwardHandler() override
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