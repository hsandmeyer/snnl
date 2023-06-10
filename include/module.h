#pragma once
#include "connector.h"
#include "forward_declare.h"
#include "tools.h"
#include <fstream>
#include <pthread.h>
#include <set>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

namespace snnl
{

template<class TElem>
class Module
{

protected:
    std::set<NodeShPtr<TElem>> _weights;

    // Vector of above. We need to keep the insertion order for io operations to
    // work
    std::vector<NodeShPtr<TElem>> _weightsSortedByInsertion;

    std::set<ModuleShPtr<TElem>> _modules;

    NodeShPtr<TElem> addWeight(const std::initializer_list<size_t>& shape)
    {
        return addWeight(Index{shape});
    }

    template<typename TArray>
    NodeShPtr<TElem> addWeight(const TArray& shape)
    {
        NodeShPtr<TElem> weight = Node<TElem>::create(shape, true);
        weight->setWeight(true);
        insertWeight(weight);

        return weight;
    }

    void insertWeight(NodeShPtr<TElem>& weight)
    {
        if(not _weights.contains(weight)) {
            _weightsSortedByInsertion.push_back(weight);
        }
        _weights.emplace(weight);
    }

    template<template<class> class ChildModule, typename... TArgs>
    std::shared_ptr<ChildModule<TElem>> addModule(TArgs&&... args)
    {
        std::shared_ptr<ChildModule<TElem>> module =
            Module<TElem>::create<ChildModule>(std::forward<TArgs>(args)...);
        // Capture the weights of a child module to save them as well
        for(auto weight : module->_weightsSortedByInsertion) {
            insertWeight(weight);
        }
        return module;
    }

    virtual NodeShPtr<TElem> callHandler(std::vector<NodeShPtr<TElem>> inputs) = 0;

    Module() = default;

public:
    std::vector<uint8_t> toByteArray()
    {
        std::vector<uint8_t> out;
        for(auto weight : _weightsSortedByInsertion) {
            auto values = weight->values().toByteArray();
            out.insert(out.end(), values.begin(), values.end());
        }
        return out;
    }

    void fromByteArray(const std::vector<uint8_t>& array)
    {
        auto currentVec = array;
        auto begin      = array.begin();
        auto end        = array.end();
        for(auto& weight : _weightsSortedByInsertion) {

            Tensor<TElem> weight_read;

            size_t bytes_read = weight_read.fromByteArray(begin, end);

            if(weight->values().shape() != weight_read.shape()) {

                std::cout << "Tensor incompatible with target: " << weight_read.shape()
                          << std::string(" vs. ") << weight->values().shape();
                throw std::domain_error("Tensor incompatible with target");
            }
            weight->values() = weight_read;
            begin += bytes_read;
        }
    }

    void saveToFile(std::string file_name)
    {

        file_name = append_if_not_endswith(file_name, ".snnl");

        std::ofstream out(file_name);

        if(not out.is_open()) {
            throw std::invalid_argument("Could not open " + file_name);
        }

        auto bytes = toByteArray();
        for(auto val : bytes) {
            out << val;
        }

        out.close();
    }

    void loadFromFile(std::string file_name)
    {
        file_name = append_if_not_endswith(file_name, ".snnl");

        std::ifstream ifs(file_name);

        if(not ifs.is_open()) {
            throw std::invalid_argument("Could not open " + file_name);
        }

        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)),
                                   std::istreambuf_iterator<char>());

        fromByteArray(bytes);
    }

    const std::set<NodeShPtr<TElem>>& weights() { return _weights; }

    template<typename... NodeShPtrs>
    NodeShPtr<TElem> call(const NodeShPtrs&... prev_nodes)
    {
        return callHandler(std::vector<NodeShPtr<TElem>>{prev_nodes...});
    }

    template<template<class> class ChildModule, typename... TArgs>
    static ::std::shared_ptr<ChildModule<TElem>> create(TArgs&&... args)
    {
        return std::shared_ptr<ChildModule<TElem>>(
            new ChildModule<TElem>(::std::forward<TArgs>(args)...));
    }

    virtual ~Module() {}
};

} // namespace snnl