#pragma once
#include "forward_declare.h"
#include "node.h"
#include "tensor.h"
#include <stdexcept>

namespace snnl
{
template<typename TElem, size_t NumTensors>
class BatchGenerator
{
    std::array<Tensor<TElem>, NumTensors> _data;

    std::vector<size_t> _shuffled_indices;
    size_t              _epoch_size;
    size_t              _current_index = 0;
    size_t              _epoch_counter = 0;
    size_t              _epoch         = 0;
    bool                _mute          = false;

    std::function<void(size_t)> _epoch_callback = [](size_t epoch) {
        std::cout << "Reaching epoch " + std::to_string(epoch) << std::endl;
    };

    std::mt19937_64 _rng;

    void reshuffle()
    {
        std::shuffle(_shuffled_indices.begin(), _shuffled_indices.end(), _rng);
        _current_index = 0;
    }

public:
    template<typename... TArgs>
    BatchGenerator(TArgs... args)
        : _data{args...}
        , _rng(time(NULL))
    {
        for(size_t i = 0; i < _data.size(); i++) {
            if(_data[0].shape(0) != _data[i].shape(0)) {
                throw std::domain_error("Dimensions of tensors mismatch");
            }
        }
        _shuffled_indices.resize(_data[0].shape(0));
        for(size_t i = 0; i < _shuffled_indices.size(); i++) {
            _shuffled_indices[i] = i;
        }
        _epoch_size = _shuffled_indices.size();

        reshuffle();
    }

    std::array<NodeShPtr<TElem>, NumTensors> generateBatch(size_t batch_size)
    {
        std::array<NodeShPtr<TElem>, NumTensors> out;

        for(size_t i = 0; i < NumTensors; i++) {
            auto shape_out = _data[i].shape();
            shape_out[0]   = batch_size;

            auto batch = Node<TElem>::create(shape_out);
            out[i]     = batch;
        }

        for(size_t batch_index = 0; batch_index < batch_size; batch_index++) {

            for(size_t i = 0; i < NumTensors; i++) {
                out[i]->values().viewAs(batch_index, ellipsis()) =
                    _data[i].viewAs(_shuffled_indices[_current_index], ellipsis());
            }

            _current_index++;
            _epoch_counter++;

            if(_current_index >= _shuffled_indices.size()) {
                reshuffle();
            }
            if(_epoch_counter >= _epoch_size) {
                _epoch++;
                if(not _mute) {
                    _epoch_callback(_epoch);
                }
                _epoch_counter = 0;
                reshuffle();
            }
        }
        return out;
    }

    void setEpochSize(size_t epoch_size)
    {
        if(epoch_size == 0) {
            throw std::domain_error("Invalid epoch size 0");
        }
        else {
            _epoch_size = epoch_size;
        }
    }
    void setEpochCallBack(std::function<void(size_t)> epoch_callback)
    {
        _epoch_callback = epoch_callback;
    }

    void reset()
    {
        reshuffle();
        _epoch         = 0;
        _epoch_counter = 0;
    }

    void mute() { _mute = true; }
};

template<typename... TArgs>
BatchGenerator(TArgs...)
    -> BatchGenerator<typename std::tuple_element_t<0, std::tuple<TArgs...>>::type,
                      sizeof...(TArgs)>;

} // namespace snnl