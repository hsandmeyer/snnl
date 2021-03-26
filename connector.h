#pragma once
#include "forward_declare.h"
#include "tensor.h"
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace snnl {

template <class TElem>
class TConnectorBaseImpl {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TConnector<TElem>;

protected:
    std::vector<std::shared_ptr<TNodeBaseImpl<TElem>>> _next_nodes;
    std::vector<TNodeBaseImpl<TElem>*>                 _prev_nodes;

    std::vector<TTensor<TElem>> _all_weights;

    size_t _prev_ready = 0;
    size_t _next_ready = 0;
    bool   _was_build  = false;

    virtual void connectNextNodeHandler(TNodeBaseImpl<TElem>&){};

    virtual void connectPrevNodeHandler(TNodeBaseImpl<TElem>&){};

public:
    virtual TNode<TElem> createOutputNode(std::vector<TNode<TElem>>&)
    {
        return TNode<TElem>();
    };

    virtual void forwardHandler() = 0;

    virtual void buildHandler(bool was_build_before) = 0;

    virtual void backwardHandler() = 0;

    virtual ~TConnectorBaseImpl() {}

    auto& prevNodes() { return _prev_nodes; }

    auto& nextNodes() { return _next_nodes; }

    void addWeightTensor(const std::initializer_list<size_t>& shape)
    {
        _all_weights.push_back(TTensor<TElem>(shape));
    }

    template <typename TArray>
    void addWeightTensor(const TArray& shape)
    {
        _all_weights.push_back(TTensor<TElem>(shape));
    }

    void forward()
    {
        _prev_ready++;
        if (_prev_ready == _prev_nodes.size()) {

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

    void connectNextNode(std::shared_ptr<TNodeBaseImpl<TElem>> next_node)
    {
        _next_nodes.push_back(next_node);
        connectNextNodeHandler(*next_node);
    }

    void connectPrevNode(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node)
    {
        _prev_nodes.push_back(prev_node.get());
        connectPrevNodeHandler(*prev_node);
    }

    TNodeBaseImpl<TElem>& nextNode(const size_t i)
    {
        return *_next_nodes.at(i);
    }

    TNodeBaseImpl<TElem>& prevNode(const size_t i)
    {
        return *_prev_nodes.at(i);
    }

    std::vector<TTensor<TElem>*> weights() { return _all_weights; }

    TTensor<TElem>& weights(const int i)
    {
        if (_all_weights.size() == 0) {
            throw std::out_of_range(
                "No weights registered yet. Call Connector first");
        }
        return _all_weights.at(i);
    }
};

template <class TElem>
class TDenseConnectorImpl : public TConnectorBaseImpl<TElem> {

    TTensor<TElem>* _weights;
    TTensor<TElem>* _bias;
    size_t          _input_units = -1;
    size_t          _output_units;

    TNode<TElem> createOutputNode(std::vector<TNode<TElem>>& nodes) const
    {
        if (nodes.size() > 1) {
            throw std::invalid_argument(
                "Maximal one node per call for Dense Connector");
        }

        return TNode<TElem>::Default();
    }

    virtual void buildHandler(bool was_build_before)
    {

        if (!was_build_before) {
            if (_input_units == -1ul) {
                _input_units = this->_prev_nodes.at(0)->shape(-1);
            }
            this->addWeightTensor({_output_units, _input_units});
            this->addWeightTensor({_output_units});
        }

        _weights = &this->_all_weights.at(0);
        _bias    = &this->_all_weights.at(1);
    }

    void dimChecks()
    {
        for (auto& prev : this->_prev_nodes) {
            if (_input_units != prev->shape(-1)) {
                throw std::invalid_argument(
                    "Output dimsion of previous node (" +
                    std::to_string(prev->shape(-1)) +
                    ") != " + std::to_string(_input_units));
            }
        }
    }

    void forwardHandler()
    {
        dimChecks();

        for (size_t node_ind = 0; node_ind < this->_prev_nodes.size();
             node_ind++) {
            auto& output = this->_next_nodes[node_ind]->values();
            auto& input  = this->_prev_nodes[node_ind]->values();

            TIndex out_shape = input.shape();
            out_shape[-1]    = _output_units;
            output.setDims(out_shape);

            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = (*_bias)(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        output(higherDim, i) +=
                            (*_weights)(i, j) * input(higherDim, j);
                    }
                }
            }
        }
    }

    void backwardHandler()
    {
        // TODO: //Calculate dfdw and dfdz here
    }

public:
    TDenseConnectorImpl(size_t output_dim) : _output_units(output_dim) {}

    TDenseConnectorImpl(size_t input_dim, size_t output_dim)
        : _input_units(input_dim), _output_units(output_dim)
    {
    }

    virtual ~TDenseConnectorImpl()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

template <class TElem>
class TConnector {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TConnectorBaseImpl<TElem>;

    std::shared_ptr<TConnectorBaseImpl<TElem>> _impl;

    TConnector(std::shared_ptr<TConnectorBaseImpl<TElem>> impl) : _impl(impl) {}

public:
    void connectNextNode(const TNode<TElem>& next)
    {
        _impl->connectNextNode(next._impl);
        next._impl->connectPrevConnector(_impl);
    }

    void connectPrevNode(TNode<TElem>& prev)
    {
        _impl->connectPrevNode(prev._impl);
        prev._impl->connectNextConnector(_impl);
    }

    template <typename... TNodes>
    TNode<TElem> operator()(TNodes&... prev)
    {
        std::vector<TNode<TElem>> nodes{prev...};

        for (auto& node : nodes) {
            connectPrevNode(node);
        }

        TNode<TElem> out(_impl->createOutputNode(nodes));
        connectNextNode(out);

        _impl->build();

        return out;
    }

    static TConnector<TElem> TDenseConnector(size_t input_dim,
                                             size_t output_dim)
    {
        return TConnector<TElem>(std::make_shared<TDenseConnectorImpl<TElem>>(
            input_dim, output_dim));
    }

    TConnectorBaseImpl<TElem>* get() { return _impl.get(); }

    TConnectorBaseImpl<TElem>* operator->() { return _impl.get(); }
};

} // namespace snnl