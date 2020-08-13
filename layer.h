#include "tensor.h"
#include <memory>
#include <vector>
#pragma once

namespace snnl {

template <class TElem>
class TLayerConnector;

template <class TElem>
class TLayer {

    std::shared_ptr<TTensor<TElem>>     _values;
    std::shared_ptr<TTensor<TElem>>     _activations;
    std::vector<TLayerConnector<TElem>> _prev_conn = {};
    std::vector<TLayerConnector<TElem>> _next_conn = {};

public:
    template <typename... TArgs>
    TLayer(TArgs... args)
        : _values(std::make_shared<TTensor<TElem>>(args...)),
          _activations(std::make_shared<TTensor<TElem>>(args...))
    {
    }

    template <typename... Args>
    TElem &operator()(const Args... args)
    {
        return _values(args...);
    }

    template <typename... Args>
    const TElem &operator()(const Args... args) const
    {
        return _values(args...);
    }

    void connectNextLayer(TLayerConnector<TElem> &layer)
    {
        _next_conn.push_back(layer);
    }

    void connectPrevLayer(TLayerConnector<TElem> &layer)
    {
        _prev_conn.push_back(layer);
    }
};

template <class TElem>
class TLayerConnector {
    std::vector<TLayer<TElem>> _next_layers;
    std::vector<TLayer<TElem>> _prev_layers;

public:
    TLayer<TElem> operator()(const TLayer<TElem> prev)
    {
        _prev_layers.push_back(prev);
        prev.connectNextLayer(*this);

        TLayer<TElem> out;
        out.connectPrevLayer(*this);
        _next_layers.push_back(out);

        return out;
    }

    virtual void call() = 0;

    virtual void updateWeights() = 0;
};

} // namespace snnl