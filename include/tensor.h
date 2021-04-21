#pragma once
#include "index.h"
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace snnl {
template <class TElem>
class TTensor {

    int    _NDims;
    TIndex _shape;
    TIndex _strides;

    std::mt19937_64    _rng;
    std::vector<TElem> _data = {};

    template <typename TArray>
    void fillDims(const TArray& shape)
    {
        // std::cout << "Setting tensor dimension: (";
        _shape.setNDims(_NDims);

        int i = 0;
        for (auto dim_len : shape) {
            //   std::cout << dim_len << " ";
            _shape[i] = dim_len;
            i++;
        }
        // std::cout << ")" << std::endl;

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

    template <typename... Ts>
    size_t dataOffset(size_t i, Ts... im) const
    {
        return i * _strides[_strides.NDims() - sizeof...(Ts) - 1] +
               dataOffset(im...);
    }

    size_t dataOffset(size_t j) const { return j; }

    void stream(TIndex& ind, int dim, std::ostream& o) const
    {

        if (dim + 1 < NDims()) {
            o << "{";
            for (size_t i = 0; i < _shape[dim]; ++i) {
                ind[dim] = i;
                stream(ind, dim + 1, o);
                if (i < _shape[dim] - 1) {
                    o << ",\n";
                    o << std::string(dim + 1, ' ');
                }
            }
            o << "}";
        }
        else {
            o << "{";
            for (size_t i = 0; i < _shape[dim]; ++i) {
                ind[dim] = i;
                o << std::scientific << (*this)(ind);
                if (i < _shape[dim] - 1) {
                    o << ",";
                }
            }
            o << "}";
        }
    }

    friend std::ostream& operator<<(std::ostream& o, const TTensor& t)
    {

        TIndex index(t.NDims());
        auto   p = o.precision();
        o.precision(6);
        t.stream(index, 0, o);
        o.precision(p);
        return o;
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

    // TODO: global seed
    TTensor() : _NDims(0), _rng(time(NULL)) {}

    TTensor(const std::initializer_list<size_t> shape)
        : _NDims(shape.size()), _rng(time(NULL))
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

    void arangeAlongAxis(int axis, TElem start, TElem stop)
    {
        if (axis < 0) {
            axis += NDims();
        }

        TElem step = (stop - start) / static_cast<TElem>(shape(axis));

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

    size_t NElems() const { return _strides[0] * _shape[0]; }

    // Length of axis i, if all following axis ( > i) are flattened
    // into i
    size_t shapeFlattened(int i) const
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
        return _data.at(dataOffset(indices...));
#else
        return _data[dataOffset(indices...)];
#endif
    }

    template <typename... T>
    const TElem& operator()(T... indices) const
    {
#ifdef DEBUG
        return _data.at(dataOffset(indices...));
#else
        return _data[dataOffset(indices...)];
#endif
    }

    const TElem& operator()(const TIndex& index_vec) const
    {
        return const_cast<TElem&>(
            (*const_cast<TTensor<TElem>*>(this))(index_vec));
    }

    size_t index(const TIndex& index_vec) const
    {
        size_t index = 0;
        for (size_t i = 0; i < static_cast<size_t>(NDims()); ++i) {
            index += index_vec[i] * _strides[i];
        }
        return index;
    }

    TElem& operator()(const TIndex& index_vec)
    {
        return _data[index(index_vec)];
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

    void setFlattenedValues(std::initializer_list<TElem> flattened_values)
    {
        if (flattened_values.size() != _data.size()) {
            throw std::invalid_argument(
                "Flattened array does not match data size");
        }
        std::copy(flattened_values.begin(), flattened_values.end(),
                  _data.begin());
    }

    template <typename TArray>
    void setFlattenedValues(const TArray& flattened_values)
    {
        if (flattened_values.size() != _data.size()) {
            throw std::invalid_argument(
                "Flattened array does not match data size");
        }
        std::copy(flattened_values.begin(), flattened_values.end(),
                  _data.begin());
    }

    void setAllValues(TElem value) { std::fill(begin(), end(), value); }

    void normal(TElem mean = 0, TElem stddev = 1)
    {
        std::normal_distribution<double> dist(mean, stddev);
        for (auto& val : *this) {
            val = static_cast<TElem>(dist(_rng));
        }
    }

    void uniform(TElem min = -1, TElem max = 1)
    {
        std::uniform_real_distribution<double> dist(min, max);
        for (auto& val : *this) {
            val = static_cast<TElem>(dist(_rng));
        }
    }

    void xavier(size_t input_units, size_t output_units)
    {
        TElem xav_max =
            std::sqrt(6.f / static_cast<TElem>(input_units + output_units));
        TElem xav_min = -xav_max;
        uniform(xav_min, xav_max);
    }

    auto begin() { return _data.begin(); }

    auto end() { return _data.end(); }

    template <typename TElemOther>
    TTensor& operator*=(const TTensor<TElemOther>& other)
    {
        if (other._shape != _shape) {
            throw std::invalid_argument("Operatr *=: Size missmatch");
        }
        for (size_t ind = 0; ind < shapeFlattened(-1); ind++) {
            (*this)(ind) *= other(ind);
        }
        return *this;
    }

    template <typename TElemOther>
    TTensor& operator-=(const TTensor<TElemOther>& other)
    {
        if (other._shape != _shape) {
            throw std::invalid_argument("Operatr *=: Size missmatch");
        }
        for (size_t ind = 0; ind < shapeFlattened(-1); ind++) {
            (*this)(ind) -= other(ind);
        }
        return *this;
    }

    template <typename TElemOther>
    TTensor& operator/=(const TTensor<TElemOther>& other)
    {
        if (other._shape != _shape) {
            throw std::invalid_argument("Operatr *=: Size missmatch");
        }
        for (size_t ind = 0; ind < shapeFlattened(-1); ind++) {
            (*this)(ind) /= other(ind);
        }
        return *this;
    }

    template <typename TElemOther>
    TTensor& operator+=(const TTensor<TElemOther>& other)
    {
        if (other._shape != _shape) {
            throw std::invalid_argument("Operatr *=: Size missmatch");
        }
        for (size_t ind = 0; ind < shapeFlattened(-1); ind++) {
            (*this)(ind) += other(ind);
        }
        return *this;
    }

    template <typename TElemA, typename TElemB>
    friend TTensor<TElemA> operator+(const TTensor<TElemA>& a,
                                     const TTensor<TElemB>& b)
    {
        TTensor<TElemA> out(a);
        out += b;
        return out;
    }

    template <typename TElemA, typename TElemB>
    friend TTensor<TElemA> operator-(const TTensor<TElemA>& a,
                                     const TTensor<TElemB>& b)
    {
        TTensor<TElemA> out(a);
        out -= b;
        return out;
    }

    template <typename TElemA, typename TElemB>
    friend TTensor<TElemA> operator*(const TTensor<TElemA>& a,
                                     const TTensor<TElemB>& b)
    {
        TTensor<TElemA> out(a);
        out *= b;
        return out;
    }

    template <typename TElemA, typename TElemB>
    friend TTensor<TElemA> operator/(const TTensor<TElemA>& a,
                                     const TTensor<TElemB>& b)
    {
        TTensor<TElemA> out(a);
        out /= b;
        return out;
    }
};
} // namespace snnl