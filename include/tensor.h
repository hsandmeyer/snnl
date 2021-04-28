#pragma once
#include "index.h"
#include <array>
#include <cstddef>
#include <forward_declare.h>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace snnl {

template <typename TElemA, typename TElemB, typename Op>
Tensor<TElemA> elementWiseCombination(const Tensor<TElemA>& a,
                                      const Tensor<TElemB>& b, Op op);

template <class TElem>
class Tensor {

    size_t _NDims;
    Index  _shape;
    Index  _strides;

    std::mt19937_64                     _rng;
    std::shared_ptr<std::vector<TElem>> _data = {};

    template <typename TArray>
    void fillDims(const TArray& shape)
    {
        checkResizeAllowed(shape);

        _NDims = shape.size();
        _shape.setNDims(shape.size());

        int i = 0;
        for (auto dim_len : shape) {
            _shape[i] = dim_len;
            i++;
        }

        fillStrides();
    }

    template <typename TArray>
    size_t NElemsFromShape(TArray& shape)
    {
        if (shape.size() == 0) {
            // Scalar
            return 1;
        }
        return std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<size_t>());
    }

    template <typename TArray>
    void checkResizeAllowed(TArray& shape)
    {
        if (_data->size() != NElemsFromShape(shape) && _data.use_count() > 1) {
            throw std::domain_error("Trying to resize a tensor which "
                                    "is used somewhere else");
        }
    }

    void fillStrides(bool realloc = true)
    {
        _strides.setNDims(_NDims);
        if (_NDims != 0) {
            _strides[_NDims - 1] = 1;

            for (int i = _NDims - 2; i >= 0; --i) {
                _strides[i] = _shape[i + 1] * _strides[i + 1];
            }
        }

        if (realloc) {
            if (_data->size() != NElems()) {
                _data->resize(NElems());
            }
        }
    }

    template <typename... Ts>
    size_t dataOffset(size_t i, Ts... im) const
    {
        return i * _strides[_strides.NDims() - sizeof...(Ts) - 1] +
               dataOffset(im...);
    }

    size_t dataOffset(size_t j) const { return j; }

    // For scalar
    size_t dataOffset() const { return 0; }

    void stream(Index& ind, int dim, std::ostream& o) const
    {
        if (_NDims == 0) {
            o << (*_data)[0];
            return;
        }

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

    friend std::ostream& operator<<(std::ostream& o, const Tensor& t)
    {

        Index index(t.NDims());
        auto  p = o.precision();
        o.precision(6);
        t.stream(index, 0, o);
        o.precision(p);
        return o;
    }

    void forEach(Index& index, int dim,
                 std::function<TElem(const Index&)> caller)
    {

        if (dim + 1 < NDims()) {
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            if (_NDims == 0) {
                (*this)(index) = caller(index);
                return;
            }
            // Last axis
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim]     = i;
                (*this)(index) = caller(index);
            }
        }
    }

    void forEach(Index& index, int dim,
                 std::function<void(const Index&)> caller)
    {

        if (dim + 1 < NDims()) {
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            if (_NDims == 0) {
                caller(index);
                return;
            }
            // Last axis
            for (size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                caller(index);
            }
        }
    }

    size_t index(const Index& index_vec) const
    {
#ifdef DEBUG
        if (index_vec.size() > _NDims) {
            throw std::out_of_range(
                "Number of inidizes exceeds number of dimension");
        }
#endif
        size_t index = 0;
        // Backwards, because we allow less indexes than NDims.
        // We start from the right
        for (size_t i = _strides.NDims(); i-- > 0;) {
            index += index_vec[i] * _strides[i];
        }
        return index;
    }

    template <typename... T>
    void numIndicesCheck(T...) const
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
        // For empty parameter stacks, this is always false (as intended). We
        // have to disable the warning here
        if (sizeof...(T) > _NDims) {
            throw std::out_of_range(
                "Number of inidizes exceeds number of dimension");
        }
#pragma GCC diagnostic pop
    }

public:
    typedef typename std::vector<TElem>::iterator       iterator;
    typedef typename std::vector<TElem>::const_iterator const_iterator;

    // TODO: global seed
    Tensor()
        : _NDims(0), _rng(time(NULL)),
          _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(std::array<size_t, 0>{});
    }

    Tensor(const std::initializer_list<size_t> shape)
        : _NDims(shape.size()), _rng(time(NULL)),
          _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    template <typename TArray>
    Tensor(const TArray& shape)
        : _NDims(shape.size()), _rng(time(NULL)),
          _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    Tensor(const Tensor& other)
        : _NDims(other._NDims), _shape(other._shape), _strides(other._strides),
          _rng(other._rng),
          _data(std::make_shared<std::vector<TElem>>(*other._data))
    {
    }

    Tensor(Tensor&& shape) = default;

    Tensor& operator=(Tensor&& shape) = default;

    Tensor& operator=(const Tensor& other)
    {
        _NDims   = other._NDims;
        _shape   = other._shape;
        _strides = other._strides;
        _rng     = other._rng;
        _data    = std::make_shared<std::vector<TElem>>(*other._data);
        return *this;
    }

    auto begin() { return _data->begin(); }

    auto end() { return _data->end(); }

    auto begin() const { return _data->begin(); }

    auto end() const { return _data->end(); }

    int NDims() const { return _NDims; }

    Index& shape() { return _shape; }

    const Index& shape() const { return _shape; }

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

    size_t shapeFlattened(int i) const
    {
        if (i < 0) {
            i += NDims();
        }

        if (i >= NDims() || i < 0) {
            throw std::out_of_range("shapeFlattened Axis " + std::to_string(i) +
                                    " is out of range");
        }

        return NElems() / stride(i);
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

    bool isScalar() const { return _NDims == 0; }

    size_t NElems() const
    {
        if (_NDims == 0) {
            return 1;
        }
        return _strides[0] * _shape[0];
    }

    template <typename... T>
    TElem& operator()(T... indexes)
    {
#ifdef DEBUG
        numIndicesCheck(indexes...);
        return _data->at(dataOffset(indexes...));
#else
        return (*_data)[dataOffset(indexes...)];
#endif
    }

    template <typename... T>
    const TElem& operator()(T... indexes) const
    {
#ifdef DEBUG
        numIndicesCheck(indexes...);
        return _data->at(dataOffset(indexes...));
#else
        return (*_data)[dataOffset(indexes...)];
#endif
    }

    const TElem& operator()(const Index& index_vec) const
    {
        return const_cast<TElem&>(
            (*const_cast<Tensor<TElem>*>(this))(index_vec));
    }

    TElem& operator()(const Index& index_vec)
    {
#ifdef DEBUG
        return (*_data)[index(index_vec)];
#else
        return _data->at(index(index_vec));
#endif
    }

    template <typename TArray>
    void setDims(const TArray& arr)
    {
        fillDims(arr);
    }

    void setDims(const std::initializer_list<size_t> shape) { fillDims(shape); }

    void appendAxis(const size_t dim_len)
    {
        _NDims++;
        _shape.appendAxis(dim_len);
        _strides.appendAxis(0);

        checkResizeAllowed(_shape);
        fillStrides();
    }

    void appendAxis()
    {
        _NDims++;
        _shape.appendAxis(1);
        _strides.appendAxis(0);
        fillStrides(false);
    }

    void prependAxis()
    {
        _NDims++;
        _shape.prependAxis(1);
        _strides.prependAxis(0);
        fillStrides(false);
    }

    Tensor<TElem> view() { return viewAs(_shape); }

    Tensor<TElem> view() const { return viewAs(_shape); }

    Tensor<TElem> viewAs(std::initializer_list<size_t> shape)
    {
        return viewAs(std::vector<size_t>(shape.begin(), shape.end()));
    }

    Tensor<TElem> viewAs(std::initializer_list<size_t> shape) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewAs(shape);
    }

    template <typename TArray>
    Tensor<TElem> viewAs(const TArray& arr)
    {
        Tensor out;
        out._data = _data;

        out.setDims(arr);

        auto itbegin = _strides.begin();

        for (auto stride : out._strides) {
            if (stride == 1 || stride == out.NElems()) {
                continue;
            }

            itbegin = std::find(itbegin, _strides.end(), stride);
            if (itbegin == _strides.end()) {
                throw std::domain_error(
                    "View of tensor does not evenly fit into source tensor");
            }
        }
        return out;
    }

    template <typename TArray>
    Tensor<TElem> viewAs(const TArray& arr) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewAs(arr);
    }

    Tensor<TElem> reshapeFromIndices(std::initializer_list<long> list)
    {
        return reshapeFromIndices(std::vector<long>(list.begin(), list.end()));
    }

    /*
    Flatten some of the dimensions of the tensor: Pass an array of axis
    indexes. These axes as well as the first are preserved. The remaining axes
    are flattened into the next valid axis to the left. If axes inidices are
    repeated, a new axis with length one is inserted to the left. If an axis
    index is 0, a new axis of length 1 is inserted at the front. If an axis
    index is larger or equal NDims, a new axes of length 1 is inserted at the
    end. e.g.

    Tensor<int> t({2, 2, 2});
    Tensor<int> t_view = t.flattenAtAxisPositions({1});
    t_view.shape(); //{2, 4}

    t_view = t.flattenAtAxisPositions({2});
    t_view.shape(); //{4, 2}

    t_view = t.flattenAtAxisPositions({1,2});
    t_view.shape(); //{2, 2, 2}

    t_view = t.flattenAtAxisPositions({0,1});
    t_view.shape(); //{1, 2, 4}

    t_view = t.flattenAtAxisPositions({2,3});
    t_view.shape(); //{4, 2, 1}

    t_view = t.flattenAtAxisPositions({1, 2, 2});
    t_view.shape(); //{2, 2, 1, 2}
    */
    template <typename TArray>
    Tensor<TElem> reshapeFromIndices(TArray axes)
    {
        for (auto& axis : axes) {
            if (axis < 0) {
                axis += NDims();
            }
        }
        std::sort(axes.begin(), axes.end());

        Index sizeView(axes.size() + 1);

        long currentAxis = 0;
        for (size_t i = 0; i < axes.size(); i++) {
            sizeView[i] = 1;

            long end = 0;
            if (axes[i] >= 0 && axes[i] < NDims()) {
                end = axes[i];
            }
            else if (axes[i] >= NDims()) {
                end = NDims();
            }
            for (; currentAxis < end; currentAxis++) {
                sizeView[i] *= shape(currentAxis);
            }
        }
        sizeView[-1] = 1;
        for (; currentAxis < NDims(); currentAxis++) {
            sizeView[-1] *= shape(currentAxis);
        }

        return viewAs(sizeView);
    }

    template <typename TArray>
    Tensor<TElem> reshapeFromIndices(TArray arr) const
    {
        return const_cast<Tensor<TElem>*>(this)->reshapeFromIndices(arr);
    }

    Tensor<TElem> reshapeFromIndices(std::initializer_list<long> list) const
    {
        return const_cast<Tensor<TElem>*>(this)->reshapeFromIndices(list);
    }

    Tensor<TElem> shrinkToNDimsFromRight(const size_t NDims)
    {
        if (NDims == 0) {
            throw std::invalid_argument("Shrinking to scalar ist not allowed");
        }
        Index newShape(NDims);
        long  shape_ind = 0;

        for (shape_ind = 1;
             shape_ind <= static_cast<long>(std::min(_NDims, NDims - 1));
             shape_ind++) {
            newShape[-shape_ind] = shape(-shape_ind);
        }
        if (NDims <= _NDims) {
            newShape[0] = shapeFlattened(-NDims);
        }
        else {
            for (; shape_ind <= static_cast<long>(NDims); shape_ind++) {
                newShape[-shape_ind] = 1;
            }
        }
        return viewAs(newShape);
    }

    Tensor<TElem> shrinkToNDimsFromLeft(const size_t NDims)
    {
        if (NDims == 0) {
            throw std::invalid_argument("Shrinking to scalar ist not allowed");
        }
        Index  newShape(NDims);
        size_t shape_ind = 0;

        for (shape_ind = 0; shape_ind < std::min(_NDims, NDims - 1);
             shape_ind++) {
            newShape[shape_ind] = shape(shape_ind);
        }
        if (NDims <= _NDims) {
            if (static_cast<long>(NDims - 2) < 0) {
                newShape[-1] = NElems();
            }
            else {
                newShape[-1] = _strides[NDims - 2];
            }
        }
        else {
            for (; shape_ind < NDims; shape_ind++) {
                newShape[shape_ind] = 1;
            }
        }
        return viewAs(newShape);
    }

    Tensor<TElem> shrinkToNDimsFromLeft(const size_t NDims) const
    {
        return const_cast<Tensor<TElem>*>(this)->shrinkToNDimsFromLeft(NDims);
    }

    Tensor<TElem> shrinkToNDimsFromRight(const size_t NDims) const
    {
        return const_cast<Tensor<TElem>*>(this)->shrinkToNDimsFromRight(NDims);
    }

    void forEach(std::function<void(const Index&)> func)
    {
        Index index(NDims());
        forEach(index, 0, func);
    }

    void modifyForEach(std::function<TElem(const Index&)> func)
    {
        Index index(NDims());
        forEach(index, 0, func);
    }

    void arangeAlongAxis(int axis, TElem start, TElem stop)
    {
        if (axis < 0) {
            axis += NDims();
        }

        TElem step = 0;

        if (_NDims != 0) {
            step = (stop - start) / static_cast<TElem>(shape(axis));
        }
        else {
            (*_data)[0] = start;
            return;
        }

        modifyForEach([&](const Index& index) -> TElem {
            return start + index[axis] * step;
        });
    }

    void setAllValues(TElem value) { std::fill(begin(), end(), value); }

    void setFlattenedValues(std::initializer_list<TElem> flattened_values)
    {
        return setFlattenedValues(std::vector<TElem>(flattened_values.begin(),
                                                     flattened_values.end()));
    }

    template <typename TArray>
    void setFlattenedValues(const TArray& flattened_values)
    {
        if (flattened_values.size() != _data->size()) {
            throw std::invalid_argument(
                "Flattened array does not match data size");
        }
        std::copy(flattened_values.begin(), flattened_values.end(),
                  _data->begin());
    }

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

    /*Elementwise modification in place using operation defined by op. If
    other.NDims() is smaller, than NDims(), broadcasting
    takes place. Otherwise the dimension has to match exactly*/
    template <typename TElemOther, typename Op>
    Tensor& elementWiseModification(const Tensor<TElemOther>& other, Op op)
    {
        if (other.NDims() > NDims()) {
            throw std::invalid_argument(
                "Operatr *=: Cannot multiply with tensor of higher dimension");
        }
        for (long i = 1; i <= other.NDims(); i++) {
            if (shape(-i) != other.shape(-i)) {
                throw std::invalid_argument(
                    "Operatr *=: Dimension mismatch. Shape at " +
                    std::to_string(-i) +
                    " is unequal: " + std::to_string(shape(-i)) + " vs " +
                    std::to_string(other.shape(-i)));
            }
        }

        Tensor<TElem> this_view  = reshapeFromIndices({-other.NDims()});
        Tensor<TElem> other_view = other.shrinkToNDimsFromLeft(1);

        for (size_t i = 0; i < this_view.shape(0); i++) {
            for (size_t j = 0; j < this_view.shape(1); j++) {
                this_view(i, j) = op(this_view(i, j), other_view(j));
            }
        }
        return *this;
    }

    template <typename TElemOther>
    Tensor& operator+=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(
            other, [](TElem a, TElemOther b) { return a + b; });
    }

    template <typename TElemOther>
    Tensor& operator-=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(
            other, [](TElem a, TElemOther b) { return a - b; });
    }

    template <typename TElemOther>
    Tensor& operator*=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(
            other, [](TElem a, TElemOther b) { return a * b; });
    }

    template <typename TElemOther>
    Tensor& operator/=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(
            other, [](TElem a, TElemOther b) { return a / b; });
    }

    template <typename TElemB>
    friend Tensor<TElem> operator+(const Tensor<TElem>&  a,
                                   const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b,
                                      [](TElem a, TElemB b) { return a + b; });
    }

    template <typename TElemB>
    friend Tensor<TElem> operator-(const Tensor<TElem>&  a,
                                   const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b,
                                      [](TElem a, TElemB b) { return a - b; });
    }

    template <typename TElemB>
    friend Tensor<TElem> operator*(const Tensor<TElem>&  a,
                                   const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b,
                                      [](TElem a, TElemB b) { return a * b; });
    }

    template <typename TElemB>
    friend Tensor<TElem> operator/(const Tensor<TElem>&  a,
                                   const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b,
                                      [](TElem a, TElemB b) { return a / b; });
    }
};

template <typename TElemA, typename TElemB, typename Op>
Tensor<TElemA> elementWiseCombination(const Tensor<TElemA>& a,
                                      const Tensor<TElemB>& b, Op op)
{
    long NDims_smaller = std::min(a.NDims(), b.NDims());
    for (long i = 1; i <= NDims_smaller; i++) {
        if (a.shape(-i) != b.shape(-i)) {
            throw std::invalid_argument(
                "Operatr *=: Dimension mismatch. Shape at " +
                std::to_string(-i) +
                " is unequal: " + std::to_string(a.shape(-i)) + " vs " +
                std::to_string(b.shape(-i)));
        }
    }

    if (a.NDims() > b.NDims()) {
        Tensor<TElemA> out(a.shape());
        Tensor<TElemA> out_view = out.reshapeFromIndices({-b.NDims()});
        Tensor<TElemA> a_view   = a.reshapeFromIndices({-b.NDims()});
        Tensor<TElemB> b_view   = b.shrinkToNDimsFromLeft(1);

        for (size_t i = 0; i < a_view.shape(0); i++) {
            for (size_t j = 0; j < a_view.shape(1); j++) {
                out_view(i, j) = op(a_view(i, j), b_view(j));
            }
        }
        return out;
    }
    else {
        Tensor<TElemA> out(b.shape());
        Tensor<TElemA> out_view = out.reshapeFromIndices({-a.NDims()});
        Tensor<TElemB> b_view   = b.reshapeFromIndices({-a.NDims()});
        Tensor<TElemA> a_view   = a.shrinkToNDimsFromLeft(1);

        for (size_t i = 0; i < b_view.shape(0); i++) {
            for (size_t j = 0; j < b_view.shape(1); j++) {
                out_view(i, j) = op(a_view(j), b_view(i, j));
            }
        }
        return out;
    }
}

} // namespace snnl