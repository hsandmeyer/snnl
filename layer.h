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
class TLayerBaseImpl {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TLayer<TElem>;

protected:
    std::vector<std::shared_ptr<TNodeBaseImpl<TElem>>> _next_nodes;
    std::vector<TNodeBaseImpl<TElem> *>                _prev_nodes;
    std::vector<TTensor<TElem> *>                      _all_weights;
    size_t                                             _prev_ready = 0;

    virtual void
    connectNextNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>> next_node) = 0;

    virtual void
    connectPrevNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node) = 0;

public:
    virtual TNode<TElem> createOutputNode() = 0;

    virtual void callHandler() = 0;

    virtual void updateWeights() = 0;

    virtual ~TLayerBaseImpl() {}

    auto &prevNodes() { return _prev_nodes; }

    auto &nextNodes() { return _next_nodes; }

    template <typename... TWeights>
    void registerWeights(TTensor<TElem> &weights, TWeights &... other_weights)
    {
        _all_weights.push_back(&weights);
        registerWeights(other_weights...);
    }

    template <typename... TWeights>
    void registerWeights(TTensor<TElem> &weights)
    {
        _all_weights.push_back(&weights);
    }

    void call()
    {
        _prev_ready++;
        if (_prev_ready == _prev_nodes.size()) {

            callHandler();

            for (auto &node : _next_nodes) {
                node->call();
            }
            _prev_ready = 0;
        }
    }

    void connectNextNode(std::shared_ptr<TNodeBaseImpl<TElem>> next_node)
    {
        _next_nodes.push_back(next_node);
        connectNextNodeHandler(next_node);
    }

    void connectPrevNode(std::shared_ptr<TNodeBaseImpl<TElem>> prev_node)
    {
        _prev_nodes.push_back(prev_node.get());
        connectPrevNodeHandler(prev_node);
    }
};

template <class TElem>
class TDenseLayerImpl : public TLayerBaseImpl<TElem> {

    TTensor<TElem> _weights;
    TTensor<TElem> _bias;

    TNode<TElem> createOutputNode()
    {
        auto shape              = this->_prev_nodes.at(0)->subShape(1, -1);
        shape[shape.size() - 1] = _bias.shape(-1);
        return TNode<TElem>::Default(shape);
    }

    void updateWeights() {}

    void callHandler()
    {
        for (size_t node_ind = 0; node_ind < this->_prev_nodes.size();
             node_ind++) {
            auto &output = this->_next_nodes[node_ind]->getTensor();
            auto &input  = this->_prev_nodes[node_ind]->getTensor();

            for (size_t higherDim = 0; higherDim < input.shapeFlattened(-2);
                 higherDim++) {
                for (size_t i = 0; i < output.shape(-1); i++) {
                    output(higherDim, i) = _bias(i);
                    for (size_t j = 0; j < input.shape(-1); j++) {
                        // std::cout << "TEST " << i << " " << j << " "
                        //          << _weights(i, j) << " "
                        //          << input(higherDim, j) << std::endl;
                        output(higherDim, i) +=
                            _weights(i, j) * input(higherDim, j);
                    }
                }
            }
        }
    }

    void connectNextNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>>)
    {
        // TODO dimension check with existing nodes
    }

    void connectPrevNodeHandler(std::shared_ptr<TNodeBaseImpl<TElem>>)
    {
        // TODO dimension check with existing nodes
    }

public:
    TDenseLayerImpl(size_t output_dim, size_t input_dim)
        : _weights({output_dim, input_dim}), _bias({output_dim})
    {
        this->registerWeights(_weights, _bias);
    }

    virtual ~TDenseLayerImpl()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

template <class TElem>
class TLayer {

    friend class TNodeBaseImpl<TElem>;
    friend class TNode<TElem>;
    friend class TLayerBaseImpl<TElem>;

    std::shared_ptr<TLayerBaseImpl<TElem>> _impl;

    TLayer(std::shared_ptr<TLayerBaseImpl<TElem>> impl) : _impl(impl) {}

public:
    void connectNextLayer(const TNode<TElem> &next)
    {
        _impl->connectNextNode(next._impl);
        next._impl->connectPrevLayer(_impl);
    }

    void connectPrevLayer(TNode<TElem> &prev)
    {
        _impl->connectPrevNode(prev._impl);
        prev._impl->connectNextLayer(_impl);
    }

    TNode<TElem> operator()(TNode<TElem> &prev)
    {
        connectPrevLayer(prev);
        TNode<TElem> out(_impl->createOutputNode());
        connectNextLayer(out);

        return out;
    }

    static TLayer<TElem> TDenseLayer(size_t output_dim, size_t input_dim)
    {
        return TLayer<TElem>(
            std::make_shared<TDenseLayerImpl<TElem>>(output_dim, input_dim));
    }

    static TLayer<TElem> TDenseLayer(size_t output_dim, TNode<TElem> &input)
    {
        return TLayer<TElem>(std::make_shared<TDenseLayerImpl<TElem>>(
            output_dim, input.shape(-1)));
    }

    TLayerBaseImpl<TElem> *get() { return _impl.get(); }

    std::vector<TTensor<TElem> *> getWeights() { return _impl->_all_weights; }

    TTensor<TElem> &getWeights(const int i)
    {
        return *_impl->_all_weights.at(i);
    }
};

} // namespace snnl