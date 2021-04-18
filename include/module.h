#pragma once
#include "connector.h"
#include "forward_declare.h"
#include <set>
#include <vector>

namespace snnl {

template <class TElem>
class TModule {
protected:
    std::set<TNodeShPtr<TElem>> _weights;

    std::set<TModuleShPtr<TElem>> _modules;

    TNodeShPtr<TElem> addWeight(const std::initializer_list<size_t>& shape)
    {
        return addWeight(TIndex{shape});
    }

    template <typename TArray>
    TNodeShPtr<TElem> addWeight(const TArray& shape)
    {
        TNodeShPtr<TElem> weight = TNode<TElem>::create(shape, true);
        weight->setWeight(true);

        _weights.emplace(weight);

        return weight;
    }

    template <template <class> class TChildModule, typename... TArgs>
    std::shared_ptr<TChildModule<TElem>> addModule(TArgs&&... args)
    {
        std::shared_ptr<TChildModule<TElem>> module =
            TModule<TElem>::create<TChildModule>(std::forward<TArgs>(args)...);
        // Capture the weights of a child module to save them as well
        _weights.insert(module->weights().begin(), module->weights().end());
        return module;
    }

    TModule<TElem>() = default;

    virtual TNodeShPtr<TElem>
    callHandler(std::vector<TNodeShPtr<TElem>> inputs) = 0;

public:
    const std::set<TNodeShPtr<TElem>>& weights() { return _weights; }

    template <typename... TNodeShPtrs>
    TNodeShPtr<TElem> call(const TNodeShPtrs&... prev_nodes)
    {
        return callHandler(std::vector<TNodeShPtr<TElem>>{prev_nodes...});
    }

    template <template <class> class TChildModule, typename... TArgs>
    static ::std::shared_ptr<TChildModule<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<TChildModule<TElem>>(
            new TChildModule<TElem>(::std::forward<TArgs>(args)...));
    }
};

} // namespace snnl