#include "tensor.h"
#include <initializer_list>
#include <memory>
#include <vector>
#pragma once

namespace snnl {

template <class TElem>
class TConnectorBase;

template <class TElem>
class TConnector;

template <class TElem>
class TLayer;

template <class TElem>
class TLayerBase {

    friend class TLayer<TElem>;
    friend class TConnector<TElem>;
    friend class TConnectorBase<TElem>;

    TTensor<TElem> _values;
    TTensor<TElem> _activations;

    std::vector<std::shared_ptr<TConnectorBase<TElem>>> _next_conn = {};
    std::vector<TConnectorBase<TElem> *>                _prev_conn = {};

public:
    template <typename... TArgs>
    TLayerBase(TArgs... args) : _values(args...), _activations(args...)
    {
    }

    TLayerBase(const std::initializer_list<size_t> list)
        : _values(list), _activations(list)
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

    virtual ~TLayerBase() { std::cout << "Destroying TLayerBase" << std::endl; }
};

template <class TElem>
class TLayer {

    friend class TLayerBase<TElem>;
    friend class TConnector<TElem>;
    friend class TConnectorBase<TElem>;

    std::shared_ptr<TLayerBase<TElem>> _impl;

public:
    template <typename... TArgs>
    TLayer(TArgs... args) : _impl(std::make_shared<TLayerBase<TElem>>(args...))
    {
    }

    TLayer(const std::initializer_list<size_t> list)
        : _impl(std::make_shared<TLayerBase<TElem>>(list))
    {
    }

    TLayer(const std::shared_ptr<TLayerBase<TElem>> &impl) : _impl(impl) {}

    template <typename... Args>
    TElem &operator()(const Args... args)
    {
        return _impl(args...);
    }

    template <typename... Args>
    const TElem &operator()(const Args... args) const
    {
        return _impl(args...);
    }

    TLayerBase<TElem> *get() { return _impl.get(); }

    void nextectNextConnector(TConnector<TElem> &next)
    {
        _impl->_next_conn.push_back(next._impl);
        next._impl->prev_conn.push_back(_impl.get());
    }

    void connectPrevConnector(TConnector<TElem> &prev)
    {
        _impl->_prev_conn.push_back(prev._impl);
        prev._impl->_next_conn.push_back(_impl);
    }
};

template <class TElem>
class TConnector;

template <class TElem>
class TConnectorBase {

    friend class TLayerBase<TElem>;
    friend class TLayer<TElem>;
    friend class TConnector<TElem>;

    friend TConnector<TElem>;

    std::vector<std::shared_ptr<TLayerBase<TElem>>> _next_layers;
    std::vector<TLayerBase<TElem> *>                _prev_layers;

public:
    virtual TLayer<TElem> createOutputLayer() = 0;

    virtual void call() = 0;

    virtual void updateWeights() = 0;

    virtual ~TConnectorBase() {}
};

template <class TElem>
class TDenseConnectorImpl : public TConnectorBase<TElem> {

    std::initializer_list<size_t> _dims;

public:
    TDenseConnectorImpl(std::initializer_list<size_t> dims) : _dims(dims) {}

    TLayer<TElem> createOutputLayer() { return TLayer<TElem>(_dims); }

    void updateWeights() {}

    void call() {}

    virtual ~TDenseConnectorImpl()
    {
        std::cout << "Destroying Dense Impl" << std::endl;
    }
};

template <class TElem>
class TConnector {

    friend class TLayerBase<TElem>;
    friend class TLayer<TElem>;
    friend class TConnectorBase<TElem>;

    std::shared_ptr<TConnectorBase<TElem>> _impl;

    TConnector(std::shared_ptr<TConnectorBase<TElem>> impl) : _impl(impl) {}

public:
    void connectNextLayer(const TLayer<TElem> &next)
    {
        _impl->_next_layers.push_back(next._impl);
        next._impl->_prev_conn.push_back(_impl.get());
    }

    void connectPrevLayer(TLayer<TElem> &prev)
    {
        _impl->_prev_layers.push_back(prev.get());
        prev._impl->_next_conn.push_back(_impl);
    }

    TLayer<TElem> operator()(TLayer<TElem> &prev)
    {
        connectPrevLayer(prev);
        TLayer<TElem> out(_impl->createOutputLayer());
        connectNextLayer(out);

        return out;
    }

    static TConnector<TElem> TDenseConnector(std::initializer_list<size_t> dims)
    {
        return TConnector<TElem>(
            std::make_shared<TDenseConnectorImpl<TElem>>(dims));
    }
};

} // namespace snnl