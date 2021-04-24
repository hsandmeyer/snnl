#pragma once
#include "connector.h"
#include "forward_declare.h"
#include <set>
#include <vector>

namespace snnl {

template <class TElem>
class Module {
protected:
    std::set<NodeShPtr<TElem>> _weights;

    std::set<ModuleShPtr<TElem>> _modules;

    NodeShPtr<TElem> addWeight(const std::initializer_list<size_t>& shape)
    {
        return addWeight(Index{shape});
    }

    template <typename TArray>
    NodeShPtr<TElem> addWeight(const TArray& shape)
    {
        NodeShPtr<TElem> weight = Node<TElem>::create(shape, true);
        weight->setWeight(true);

        _weights.emplace(weight);

        return weight;
    }

    template <template <class> class ChildModule, typename... TArgs>
    std::shared_ptr<ChildModule<TElem>> addModule(TArgs&&... args)
    {
        std::shared_ptr<ChildModule<TElem>> module =
            Module<TElem>::create<ChildModule>(std::forward<TArgs>(args)...);
        // Capture the weights of a child module to save them as well
        _weights.insert(module->weights().begin(), module->weights().end());
        return module;
    }

    Module<TElem>() = default;

    virtual NodeShPtr<TElem>
    callHandler(std::vector<NodeShPtr<TElem>> inputs) = 0;

public:
    const std::set<NodeShPtr<TElem>>& weights() { return _weights; }

    template <typename... NodeShPtrs>
    NodeShPtr<TElem> call(const NodeShPtrs&... prev_nodes)
    {
        return callHandler(std::vector<NodeShPtr<TElem>>{prev_nodes...});
    }

    template <template <class> class ChildModule, typename... TArgs>
    static ::std::shared_ptr<ChildModule<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<ChildModule<TElem>>(
            new ChildModule<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual ~Module() {}
};

} // namespace snnl