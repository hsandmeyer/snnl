#pragma once
#include "index.h"
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace snnl {
template <class TElem>
class TTensor {

    int                _NDims;
    TIndex             _shape;
    TIndex             _strides;
    std::vector<TElem> _data = {};

    template <typename TArray>
    void fillDims(const TArray& shape)
    {
        std::cout << "Constructing tensor with ";
        _shape.setNDims(_NDims);

        int i = 0;
        for (auto dim_len : shape) {
            std::cout << dim_len << " ";
            _shape[i] = dim_len;
            i++;
        }
        std::cout << std::endl;

        fillStrides();
    }

    void fillStrides()
    {
        _strides.setNDims(_NDims);
        _strides[_NDims - 1] = 1;

        for (int i = _NDims - 2; i >= 0; --i) {
            _strides[i] = _shape[i + 1] * _strides[i + 1];
        }

        _data.resize(NElems());
    }

    template <size_t N, typename... Ts>
    size_t dataOffset(size_t i, Ts... im)
    {
        return i * _strides[_strides.NDims() - sizeof...(Ts) - 1] +
               dataOffset<N + 1>(im...);
    }

    template <size_t N>
    size_t dataOffset(size_t j)
    {
        return j;
    }

    void forEach(TIndex& index, int dim,
                 std::function<TElem(const TIndex&)> caller)
    {

        if (dim + 1 < NDims()) {
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            // Last axis
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim]     = i;
                (*this)(index) = caller(index);
            }
        }
    }

    void forEach(TIndex& index, int dim,
                 std::function<void(const TIndex&)> caller)
    {

        if (dim + 1 < NDims()) {
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            // Last axis
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                caller(index);
            }
        }
    }

public:
    typedef typename std::vector<TElem>::iterator       iterator;
    typedef typename std::vector<TElem>::const_iterator const_iterator;

    TTensor() : _NDims(0) {}

    TTensor(const std::initializer_list<size_t> shape) : _NDims(shape.size())
    {
        fillDims(shape);
    }

    template <typename TArray>
    TTensor(const TArray& shape) : _NDims(shape.size())
    {
        fillDims(shape);
    }

    void forEach(std::function<void(const TIndex&)> func)
    {
        TIndex index(NDims());
        forEach(index, 0, func);
    }

    void modifyForEach(std::function<TElem(const TIndex&)> func)
    {
        TIndex index(NDims());
        forEach(index, 0, func);
    }

    void rangeAllDims(int axis, TElem start, TElem step)
    {
        if (axis < 0) {
            axis += NDims();
        }
        modifyForEach([&](const TIndex& index) -> TElem {
            return start + index[axis] * step;
        });
    }

    template <typename TArray>
    void setDims(const TArray& arr)
    {
        _NDims = arr.size();
        fillDims(arr);
    }
    void setDims(const std::initializer_list<size_t> shape)
    {
        _NDims = shape.size();
        fillDims(shape);
    }

    void addDim(const size_t dim_len)
    {
        _NDims++;
        _shape.addDim(dim_len);
        _strides.addDim(0);

        fillStrides();
    }

    int NDims() const { return _NDims; }

    size_t NElems() { return _strides[0] * _shape[0]; }

    // Length of axis i, if all following axis ( > i) are flattened
    // into i
    size_t shapeFlattened(int i)
    {

        if (i < 0) {
            i += NDims();
        }

        if (i >= NDims() || i < 0) {
            return 1;
        }

        return NElems() / stride(i);
    }

    template <typename... T>
    TElem& operator()(T... indices)
    {
#ifdef DEBUG
        return _data.at(dataOffset<0>(indices...));
#else
        return _data[dataOffset<0>(indices...)];
#endif
    }

    template <typename... T>
    const TElem& operator()(T... indices) const
    {
#ifdef DEBUG
        return _data.at(dataOffset<0>(indices...));
#else
        return _data[dataOffset<0>(indices...)];
#endif
    }

    const TElem& operator()(const TIndex& index_vec) const
    {
        return const_cast<TElem&>(
            (*const_cast<TTensor<TElem>*>(this))(index_vec));
    }

    TElem& operator()(const TIndex& index_vec)
    {
        size_t index = 0;
        for (size_t i = _strides.NDims(); i-- > 0;) {
            index += index_vec[i] * _strides[i];
        }
        return _data[index];
    }

    size_t shape(int i) const
    {
        if (i < 0) {
            i += NDims();
        }

#ifdef DEBUG

        if (i < 0 || i >= static_cast<int>(_shape.size())) {
            throw std::out_of_range(std::string(
                "Shape: Index " + std::to_string(i) + " is out of range"));
        }
#endif

        return _shape[i];
    }

    size_t stride(int i) const
    {
        if (i < 0) {
            i += NDims();
        }
#ifdef DEBUG

        if (i < 0 || i >= static_cast<int>(_strides.size())) {
            throw std::out_of_range(std::string(
                "stride: Index " + std::to_string(i) + " is out of range"));
        }
#endif
        return _strides[i];
    }

    TIndex& shape() { return _shape; }

    const TIndex& shape() const { return _shape; }

    TIndex subShape(int first, int last) const
    {
        if (first < 0) {
            first += NDims();
        }
        if (last < 0) {
            last += NDims();
        }
        return TIndex(_shape.cbegin() + first, _shape.cbegin() + first + last);
    }

    void setAllValues(TElem value) { std::fill(begin(), end(), value); }

    auto begin() { return _data.begin(); }

    auto end() { return _data.end(); }
};
} // namespace snnl