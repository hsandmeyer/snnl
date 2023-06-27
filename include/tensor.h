#pragma once
#include "index.h"
#include "tools.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

namespace snnl
{

template<typename TElemA, typename TElemB, typename Op>
Tensor<TElemA> elementWiseCombination(const Tensor<TElemA>& a, const Tensor<TElemB>& b, Op op);

struct All
{};

inline All all()
{
    return All();
}

struct Ellipsis
{};

inline Ellipsis ellipsis()
{
    return Ellipsis();
}

struct Full
{};

struct Range
{
    long minIndex = -1;
    long maxIndex = -1;
};

inline Range range(Full, long maxIndex)
{
    return Range{-1, maxIndex};
}

inline Range range(long minIndex, Full)
{
    return Range{minIndex, -1};
}

inline Range range(long minIndex, long maxIndex)
{
    return Range{minIndex, maxIndex};
}

struct NewAxis
{};

inline NewAxis newAxis()
{
    return NewAxis();
}

template<typename TElem>
class Node;

template<class TElem>
class Tensor
{

    size_t _NDims;
    Index  _shape;
    Index  _strides;

    std::mt19937_64 _rng;

    size_t                              _mem_offset      = 0;
    bool                                _is_partial_view = false;
    std::shared_ptr<std::vector<TElem>> _data            = {};

    template<typename TArray>
    void fillDims(const TArray& shape)
    {
        checkResizeAllowed(shape);

        _mem_offset = 0;
        _NDims      = shape.size();
        _shape.setNDims(shape.size());

        int i = 0;
        for(auto dim_len : shape) {
            _shape[i] = dim_len;
            i++;
        }

        fillStrides();
    }

    template<typename TArray>
    size_t NElemsFromShape(TArray& shape)
    {
        if(shape.size() == 0) {
            // Scalar
            return 1;
        }
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }

    template<typename TArray>
    void checkResizeAllowed(TArray& shape)
    {
        if(_data->size() != NElemsFromShape(shape) && _data.use_count() > 1) {
            throw std::domain_error("Trying to resize a tensor which "
                                    "is used somewhere else");
        }
    }

    void fillStrides(bool realloc = true)
    {
        _strides.setNDims(_NDims);
        if(_NDims != 0) {
            _strides[_NDims - 1] = 1;

            for(int i = _NDims - 2; i >= 0; --i) {
                _strides[i] = _shape[i + 1] * _strides[i + 1];
            }
        }

        if(realloc) {
            if(_data->size() != NElems()) {
                _data->resize(NElems());
            }
        }
    }

    template<typename... Ts>
    size_t dataOffset(size_t i, Ts... im) const
    {
        return i * _strides[_strides.NDims() - sizeof...(Ts) - 1] + dataOffset(im...);
    }

    size_t dataOffset(size_t j) const { return j * _strides[NDims() - 1] + _mem_offset; }

    // For scalar
    size_t dataOffset() const { return 0; }

    void stream(Index& ind, long dim, std::ostream& o) const
    {
        if(_NDims == 0) {
            o << (*_data)[0];
            return;
        }

        if(dim + 1 < NDims()) {
            o << "{";
            for(size_t i = 0; i < _shape[dim]; ++i) {
                ind[dim] = i;
                stream(ind, dim + 1, o);
                if(i < _shape[dim] - 1) {
                    o << ",\n";
                    o << std::string(dim + 1, ' ');
                }
            }
            o << "}";
        }
        else {
            o << "{";
            for(size_t i = 0; i < _shape[dim]; ++i) {
                ind[dim] = i;
                o << std::scientific << (*this)(ind);
                if(i < _shape[dim] - 1) {
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

    void forEach(Index& index, int dim, std::function<TElem(const Index&)> caller)
    {

        if(dim + 1 < NDims()) {
            for(size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            if(_NDims == 0) {
                (*this)(index) = caller(index);
                return;
            }
            // Last axis
            for(size_t i = 0; i < _shape[dim]; ++i) {
                index[dim]     = i;
                (*this)(index) = caller(index);
            }
        }
    }

    void forEach(Index& index, int dim, std::function<void(const Index&)> caller)
    {

        if(dim + 1 < NDims()) {
            for(size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                forEach(index, dim + 1, caller);
            }
        }
        else {
            if(_NDims == 0) {
                caller(index);
                return;
            }
            // Last axis
            for(size_t i = 0; i < _shape[dim]; ++i) {
                index[dim] = i;
                caller(index);
            }
        }
    }

    size_t index(const Index& index_vec) const
    {
#ifdef DEBUG
        if(index_vec.size() > _NDims) {
            throw std::out_of_range("Number of inidizes exceeds number of dimension");
        }
#endif
        if(NDims() == 0 && index_vec.size() > 0) {
            // For iterators of scalars, which need to have a size. Otherwise there is no end()
            return index_vec[0];
        }
        size_t index = _mem_offset;
        // Backwards, because we allow less indexes than NDims.
        // We start from the right
        for(size_t i = _strides.NDims(); i-- > 0;) {
            index += index_vec[i] * _strides[i];
        }
        return index;
    }

    template<typename... T>
    void numIndicesCheck(T...) const
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
        // For empty parameter stacks, this is always false (as intended). We
        // have to disable the warning here
        if(sizeof...(T) > _NDims) {
            throw std::out_of_range("Number of inidizes exceeds number of dimension");
        }
#pragma GCC diagnostic pop
    }

    size_t calcNumNewAxis(size_t numNewAxis) { return numNewAxis; }

    template<typename TArg, typename... TArgs>
    size_t calcNumNewAxis(size_t numNewAxis, TArg, TArgs... args)
    {
        if(std::is_same_v<TArg, NewAxis>) {
            numNewAxis++;
        }
        return calcNumNewAxis(numNewAxis, args...);
    }

    template<typename... TArgs>
    std::tuple<size_t, size_t> calcSizeOfEllipsisAndNumNewAxis(TArgs... args)
    {
        size_t numNewAxis   = calcNumNewAxis(0, args...);
        size_t sizeEllipsis = NDims() - sizeof...(args) + numNewAxis + 1;
        return std::tuple(sizeEllipsis, numNewAxis);
    }

    std::tuple<size_t, Index, Index> calcDimsAndMemOffset(size_t, size_t mem_offset, Index newDims,
                                                          Index newStrides, size_t)
    {
        return std::tuple<size_t, Index, Index>(mem_offset, newDims, newStrides);
    }

    template<typename... TArgs>
    std::tuple<size_t, Index, Index>
    calcDimsAndMemOffset(size_t position, size_t mem_offset, Index newDims, Index newStrides,
                         size_t sizeEllipsis, NewAxis, TArgs... args)
    {

        newDims.appendAxis(1);
        if(int(position) > 0) {
            newStrides.appendAxis(_strides[position - 1]);
        }
        else {
            newStrides.appendAxis(_strides[0] * _shape[0]);
        }

        return calcDimsAndMemOffset(position, mem_offset, newDims, newStrides, sizeEllipsis,
                                    args...);
    }

    template<typename... TArgs>
    std::tuple<size_t, Index, Index>
    calcDimsAndMemOffset(size_t position, size_t mem_offset, Index newDims, Index newStrides,
                         size_t sizeEllipsis, size_t index, TArgs... args)
    {

        mem_offset += _strides[position] * index;
        position++;
        return calcDimsAndMemOffset(position, mem_offset, newDims, newStrides, sizeEllipsis,
                                    args...);
    }

    template<typename... TArgs>
    std::tuple<size_t, Index, Index> calcDimsAndMemOffset(size_t position, size_t mem_offset,
                                                          Index newDims, Index newStrides,
                                                          size_t sizeEllipsis, All, TArgs... args)
    {
        newDims.appendAxis(_shape[position]);
        newStrides.appendAxis(_strides[position]);
        position++;
        return calcDimsAndMemOffset(position, mem_offset, newDims, newStrides, sizeEllipsis,
                                    args...);
    }

    template<typename... TArgs>
    std::tuple<size_t, Index, Index>
    calcDimsAndMemOffset(size_t position, size_t mem_offset, Index newDims, Index newStrides,
                         size_t sizeEllipsis, Ellipsis, TArgs... args)
    {
        for(size_t i = 0; i < sizeEllipsis; i++) {
            newDims.appendAxis(_shape[position]);
            newStrides.appendAxis(_strides[position]);
            position++;
        }
        return calcDimsAndMemOffset(position, mem_offset, newDims, newStrides, sizeEllipsis,
                                    args...);
    }

    template<typename... TArgs>
    std::tuple<size_t, Index, Index>
    calcDimsAndMemOffset(size_t position, size_t mem_offset, Index newDims, Index newStrides,
                         size_t sizeEllipsis, Range range, TArgs... args)
    {
        size_t start_index = 0;
        size_t end_index   = _shape[position];
        if(range.minIndex >= 0) {
            start_index = range.minIndex;
        }
        if(range.maxIndex >= 0) {
            end_index = range.maxIndex;
        }
        mem_offset += start_index * _strides[position];
        newStrides.appendAxis(_strides[position]);
        newDims.appendAxis(end_index - start_index);
        position++;
        return calcDimsAndMemOffset(position, mem_offset, newDims, newStrides, sizeEllipsis,
                                    args...);
    }

public:
    typedef typename std::vector<TElem>::iterator       iterator;
    typedef typename std::vector<TElem>::const_iterator const_iterator;

    class Iterator
    {
        Tensor<TElem>* _ptr;
        Index          _position;
        size_t         _index;

    public:
        Iterator(Tensor<TElem>& source, Index position)
            : _ptr(&source)
            , _position(position)
        {
            _index = _ptr->index(_position);
        }

        Iterator operator++()
        {
            for(size_t i = _position.size(); i-- > 0;) {
                _position[i]++;
                // We need to iterate beyond the end
                if(i != 0 && _position[i] >= _ptr->shape(i)) {
                    _position[i] = 0;
                }
                else {
                    break;
                }
            }
            _index = _ptr->index(_position);
            return *this;
        }
        bool operator!=(const Iterator& other) { return _index != other._index; }

        TElem& operator*() { return _ptr->_data->at(_index); }

        const TElem& operator*() const { return _ptr->_data.at(_index); }

        const Index& position() const { return _position; }
    };

    // TODO: global seed
    Tensor()
        : _NDims(0)
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(std::array<size_t, 0>{});
    }

    Tensor(const std::initializer_list<int> shape)
        : _NDims(shape.size())
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    Tensor(const std::initializer_list<size_t> shape)
        : _NDims(shape.size())
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    Tensor(const Index& shape)
        : _NDims(shape.size())
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    Tensor(const std::vector<size_t>& shape)
        : _NDims(shape.size())
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    template<size_t N>
    Tensor(const std::array<size_t, N>& shape)
        : _NDims(shape.size())
        , _rng(time(NULL))
        , _data(std::make_shared<std::vector<TElem>>())
    {
        fillDims(shape);
    }

    Tensor(const Tensor&) = default;

    Tensor(Tensor&&) = default;

    Tensor copy()
    {

        Tensor out(_shape);
        auto   itbegin = begin();
        auto   itend   = end();

        for(auto it = itbegin; it != itend; ++it) {
            out(it.position()) = *it;
        }
        return out;
    }

    Tensor(std::shared_ptr<Node<TElem>> node)
        : Tensor(node->values())
    {
    }

    Tensor& operator=(const Tensor& other)
    {
        if(other.shape() != this->shape()) {
            if(_is_partial_view) {
                throw std::invalid_argument(
                    "Cannot assign to partial view with different shape: This shape = " + shape() +
                    ", other shape = " + other.shape());
            }
            // Dimensions do not match. Create new tensor
            _NDims = other._NDims;
            _shape = other._shape;
            _data  = std::make_shared<std::vector<TElem>>();
            fillStrides();
        }
        auto itbegin = other.begin();
        auto itend   = other.end();

        for(auto it = itbegin; it != itend; ++it) {
            // Copy values by iterator to manage copying from views
            (*this)(it.position()) = *it;
        }
        return *this;
    }

    Iterator begin()
    {
        Index begin(std::max(_shape.NDims(), 1l));
        for(size_t i = 0; i < begin.size(); i++) {
            begin[i] = 0;
        }
        Iterator out(*this, begin);
        return out;
    }

    Iterator end()
    {
        Index end(std::max(_shape.NDims(), 1l));
        for(size_t i = 1; i < end.size(); i++) {
            end[i] = 0;
        }
        if(NDims() > 0) {
            end[0] = _shape[0];
        }
        else {
            end[0] = 1;
        }
        Iterator out(*this, end);
        return out;
    }

    Iterator begin() const { return const_cast<Tensor<TElem>*>(this)->begin(); }

    Iterator end() const { return const_cast<Tensor<TElem>*>(this)->end(); }

    long NDims() const { return _NDims; }

    bool isScalar() const { return _NDims == 0; }

    size_t size() const { return NElems(); }

    Index& shape() { return _shape; }

    const Index& shape() const { return _shape; }

    size_t shape(int i) const
    {
        if(i < 0) {
            i += NDims();
        }

#ifdef DEBUG

        if(i < 0 || i >= static_cast<int>(_shape.size())) {
            throw std::out_of_range(
                std::string("Shape: Index " + std::to_string(i) + " is out of range"));
        }
#endif

        return _shape[i];
    }

    size_t stride(int i) const
    {
        if(i < 0) {
            i += NDims();
        }
#ifdef DEBUG

        if(i < 0 || i >= static_cast<int>(_strides.size())) {
            throw std::out_of_range(
                std::string("stride: Index " + std::to_string(i) + " is out of range"));
        }
#endif
        return _strides[i];
    }

    size_t NElems() const
    {
        if(_NDims == 0) {
            return 1;
        }
        return _strides[0] * _shape[0];
    }

    template<typename... T>
    TElem& operator()(T... indexes)
    {
#ifdef DEBUG
        numIndicesCheck(indexes...);
        return _data->at(dataOffset(indexes...));
#else
        return (*_data)[dataOffset(indexes...)];
#endif
    }

    template<typename... T>
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
        return const_cast<TElem&>((*const_cast<Tensor<TElem>*>(this))(index_vec));
    }

    TElem& operator()(const Index& index_vec)
    {
#ifdef DEBUG
        return (*_data)[index(index_vec)];
#else
        return _data->at(index(index_vec));
#endif
    }

    template<typename TArray>
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

    Tensor<TElem> flatten() { return viewFromIndices({}); }
    Tensor<TElem> flatten() const { return viewFromIndices({}); }

    Tensor<TElem> viewAs(std::initializer_list<size_t> shape)
    {
        return viewAs(std::vector<size_t>(shape.begin(), shape.end()));
    }

    Tensor<TElem> viewAs(std::initializer_list<size_t> shape) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewAs(shape);
    }

    template<typename... TArgs>
    Tensor<TElem> viewAs(TArgs... args)
    {
        auto [sizeEllipsis, numNewAxis] = calcSizeOfEllipsisAndNumNewAxis(args...);
        auto [mem_offset, newShape, newStrides] =
            calcDimsAndMemOffset(0, 0, {}, {}, sizeEllipsis, args...);

        Tensor<TElem> out;

        out._NDims           = newShape.size();
        out._data            = _data;
        out._mem_offset      = mem_offset;
        out._shape           = newShape;
        out._is_partial_view = NElemsFromShape(newShape) != NElems();
        out._strides         = newStrides;
        return out;
    }

    template<typename TArray>
    Tensor<TElem> viewAs(const TArray& arr)
    {
        Tensor out;
        out._data = _data;

        out.setDims(arr);

        auto itbegin = _strides.begin();

        for(auto stride : out._strides) {
            if(stride == 1 || stride == out.NElems()) {
                continue;
            }

            itbegin = std::find(itbegin, _strides.end(), stride);
            if(itbegin == _strides.end()) {
                throw std::domain_error("View of tensor does not evenly fit into source tensor");
            }
        }
        return out;
    }

    template<typename TArray>
    Tensor<TElem> viewAs(const TArray& arr) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewAs(arr);
    }

    Tensor<TElem> viewFromIndices(std::initializer_list<long> list)
    {
        return viewFromIndices(std::vector<long>(list.begin(), list.end()));
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
    Tensor<int> t_view = t.viewFromIndices({1});
    t_view.shape(); //{2, 4}

    t_view = t.viewFromIndices({2});
    t_view.shape(); //{4, 2}

    t_view = t.viewFromIndices({1,2});
    t_view.shape(); //{2, 2, 2}

    t_view = t.viewFromIndices({0,1});
    t_view.shape(); //{1, 2, 4}

    t_view = t.viewFromIndices({2,3});
    t_view.shape(); //{4, 2, 1}

    t_view = t.viewFromIndices({1, 2, 2});
    t_view.shape(); //{2, 2, 1, 2}
    */
    template<typename TArray>
    Tensor<TElem> viewFromIndices(TArray axes)
    {
        for(auto& axis : axes) {
            if(axis < 0) {
                axis += NDims();
            }
        }
        std::sort(axes.begin(), axes.end());

        Index sizeView(axes.size() + 1);

        long currentAxis = 0;
        for(size_t i = 0; i < axes.size(); i++) {
            sizeView[i] = 1;

            long end = 0;
            if(axes[i] >= 0 && axes[i] < NDims()) {
                end = axes[i];
            }
            else if(axes[i] >= NDims()) {
                end = NDims();
            }
            for(; currentAxis < end; currentAxis++) {
                sizeView[i] *= shape(currentAxis);
            }
        }
        sizeView[-1] = 1;
        for(; currentAxis < NDims(); currentAxis++) {
            sizeView[-1] *= shape(currentAxis);
        }

        return viewAs(sizeView);
    }

    template<typename TArray>
    Tensor<TElem> viewFromIndices(TArray arr) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewFromIndices(arr);
    }

    Tensor<TElem> viewFromIndices(std::initializer_list<long> list) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewFromIndices(list);
    }

    // Keep N dims on the right. Squeeze remaining dimensions on the left into axis 0 or add
    // additional axis:
    // With N = 2:
    // shape = {2, 2, 2} -> {4, 2}
    // shape = {2} -> {1, 2}
    Tensor<TElem> viewWithNDimsOnTheRight(const size_t NDims)
    {
        if(NDims == 0) {
            throw std::invalid_argument("Shrinking to scalar ist not allowed");
        }
        Index newShape(NDims);
        long  shape_ind = 0;

        for(shape_ind = 1; shape_ind <= static_cast<long>(std::min(_NDims, NDims - 1)); shape_ind++)
        {
            newShape[-shape_ind] = shape(-shape_ind);
        }
        if(NDims <= _NDims) {
            newShape[0] = NElems() / _strides[_NDims - NDims];
        }
        else {
            for(; shape_ind <= static_cast<long>(NDims); shape_ind++) {
                newShape[-shape_ind] = 1;
            }
        }
        return viewAs(newShape);
    }

    // Keep N dims on the left. Squeeze remaining dimensions on the  right into axis -1 or add
    // additional axis:
    // With N = 2:
    // shape = {2, 2, 2} -> {2, 4}
    // shape = {2} -> {2, 1}
    Tensor<TElem> viewWithNDimsOnTheLeft(const size_t NDims)
    {
        if(NDims == 0) {
            throw std::invalid_argument("Shrinking to scalar ist not allowed");
        }
        Index  newShape(NDims);
        size_t shape_ind = 0;

        for(shape_ind = 0; shape_ind < std::min(_NDims, NDims - 1); shape_ind++) {
            newShape[shape_ind] = shape(shape_ind);
        }
        if(NDims <= _NDims) {
            if(static_cast<long>(NDims - 2) < 0) {
                newShape[-1] = NElems();
            }
            else {
                newShape[-1] = _strides[NDims - 2];
            }
        }
        else {
            for(; shape_ind < NDims; shape_ind++) {
                newShape[shape_ind] = 1;
            }
        }
        return viewAs(newShape);
    }

    void adjustToNDimsOnTheLeft(const size_t NDims)
    {
        auto view        = viewWithNDimsOnTheLeft(NDims);
        _NDims           = view._NDims;
        _shape           = view._shape;
        _strides         = view._strides;
        _mem_offset      = view._mem_offset;
        _is_partial_view = view._is_partial_view;
    }

    void adjustToNDimsOnTheRight(const size_t NDims)
    {
        auto view        = viewWithNDimsOnTheRight(NDims);
        _NDims           = view._NDims;
        _shape           = view._shape;
        _strides         = view._strides;
        _mem_offset      = view._mem_offset;
        _is_partial_view = view._is_partial_view;
    }

    Tensor<TElem> viewWithNDimsOnTheLeft(const size_t NDims) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewWithNDimsOnTheLeft(NDims);
    }

    Tensor<TElem> viewWithNDimsOnTheRight(const size_t NDims) const
    {
        return const_cast<Tensor<TElem>*>(this)->viewWithNDimsOnTheRight(NDims);
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
        if(axis < 0) {
            axis += NDims();
        }

        TElem step = 0;

        if(_NDims != 0) {
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
        return setFlattenedValues(
            std::vector<TElem>(flattened_values.begin(), flattened_values.end()));
    }

    template<typename TArray>
    void setFlattenedValues(const TArray& flattened_values)
    {
        if(flattened_values.size() != _data->size()) {
            throw std::invalid_argument("Flattened array does not match data size");
        }
        std::copy(flattened_values.begin(), flattened_values.end(), _data->begin());
    }

    void normal(TElem mean = 0, TElem stddev = 1)
    {
        std::normal_distribution<double> dist(mean, stddev);
        for(auto& val : *this) {
            val = static_cast<TElem>(dist(_rng));
        }
    }

    void uniform(TElem min = -1, TElem max = 1)
    {
        std::uniform_real_distribution<double> dist(min, max);
        for(auto& val : *this) {
            val = static_cast<TElem>(dist(_rng));
        }
    }

    void xavier(size_t input_units, size_t output_units)
    {
        TElem xav_max = std::sqrt(6.f / static_cast<TElem>(input_units + output_units));
        TElem xav_min = -xav_max;
        uniform(xav_min, xav_max);
    }

    void he_normal(size_t input_units)
    {
        TElem mean = 0;
        TElem var  = sqrt(2. / input_units);
        normal(mean, var);
    }

    std::vector<TElem>& rawData() { return *_data; }

    /*Elementwise modification in place using operation defined by op. If
    other.NDims() is smaller than NDims(), broadcasting
    takes place. Otherwise the dimension has to match exactly*/
    template<typename TElemOther, typename Op>
    Tensor& elementWiseModification(const Tensor<TElemOther>& other, Op op)
    {
        if(other.NDims() > NDims()) {
            throw std::invalid_argument(
                "Operatr *=: Cannot multiply with tensor of higher dimension");
        }
        for(long i = 1; i <= other.NDims(); i++) {
            if(shape(-i) != other.shape(-i)) {
                throw std::invalid_argument("Operatr *=: Dimension mismatch. Shape at " +
                                            std::to_string(-i) +
                                            " is unequal: " + std::to_string(shape(-i)) + " vs " +
                                            std::to_string(other.shape(-i)));
            }
        }

        long ref_axis = -other.NDims();
        if(other.isScalar()) {
            ref_axis = NDims();
        }
        Tensor<TElem> this_view  = viewFromIndices({ref_axis});
        Tensor<TElem> other_view = other.viewWithNDimsOnTheLeft(1);

        for(size_t i = 0; i < this_view.shape(0); i++) {
            for(size_t j = 0; j < this_view.shape(1); j++) {
                this_view(i, j) = op(this_view(i, j), other_view(j));
            }
        }
        return *this;
    }

    template<typename TElemOther>
    Tensor& operator+=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(other, [](TElem a, TElemOther b) {
            return a + b;
        });
    }

    template<typename TElemOther>
    Tensor& operator-=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(other, [](TElem a, TElemOther b) {
            return a - b;
        });
    }

    template<typename TElemOther>
    Tensor& operator*=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(other, [](TElem a, TElemOther b) {
            return a * b;
        });
    }

    template<typename TElemOther>
    Tensor& operator/=(const Tensor<TElemOther>& other)
    {
        return elementWiseModification(other, [](TElem a, TElemOther b) {
            return a / b;
        });
    }

    Tensor& operator+=(const TElem& a)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        (*this) += tmp;
        return *this;
    }

    Tensor& operator-=(const TElem& a)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        (*this) -= tmp;
        return *this;
    }

    Tensor& operator*=(const TElem& a)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        (*this) *= tmp;
        return *this;
    }

    Tensor& operator/=(const TElem& a)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        (*this) /= tmp;
        return *this;
    }

    template<typename TElemB>
    friend Tensor<TElem> operator+(const Tensor<TElem>& a, const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b, [](TElem a, TElemB b) {
            return a + b;
        });
    }

    template<typename TElemB>
    friend Tensor<TElem> operator-(const Tensor<TElem>& a, const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b, [](TElem a, TElemB b) {
            return a - b;
        });
    }

    template<typename TElemB>
    friend Tensor<TElem> operator*(const Tensor<TElem>& a, const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b, [](TElem a, TElemB b) {
            return a * b;
        });
    }

    template<typename TElemB>
    friend Tensor<TElem> operator/(const Tensor<TElem>& a, const Tensor<TElemB>& b)
    {
        return elementWiseCombination(a, b, [](TElem a, TElemB b) {
            return a / b;
        });
    }

    friend Tensor<TElem> operator+(const Tensor<TElem>& a, const TElem& b)
    {
        Tensor<TElem> tmp;
        tmp() = b;
        return elementWiseCombination(a, tmp, [](TElem a, TElem b) {
            return a + b;
        });
    }

    friend Tensor<TElem> operator+(const TElem& a, const Tensor<TElem>& b)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        return elementWiseCombination(tmp, b, [](TElem a, TElem b) {
            return a + b;
        });
    }

    friend Tensor<TElem> operator-(const Tensor<TElem>& a, const TElem& b)
    {
        Tensor<TElem> tmp;
        tmp() = b;
        return elementWiseCombination(a, tmp, [](TElem a, TElem b) {
            return a - b;
        });
    }

    friend Tensor<TElem> operator-(const TElem& a, const Tensor<TElem>& b)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        return elementWiseCombination(tmp, b, [](TElem a, TElem b) {
            return a - b;
        });
    }

    friend Tensor<TElem> operator*(const Tensor<TElem>& a, const TElem& b)
    {
        Tensor<TElem> tmp;
        tmp() = b;
        return elementWiseCombination(a, tmp, [](TElem a, TElem b) {
            return a * b;
        });
    }

    friend Tensor<TElem> operator*(const TElem& a, const Tensor<TElem>& b)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        return elementWiseCombination(tmp, b, [](TElem a, TElem b) {
            return a * b;
        });
    }

    friend Tensor<TElem> operator/(const Tensor<TElem>& a, const TElem& b)
    {
        Tensor<TElem> tmp;
        tmp() = b;
        return elementWiseCombination(a, tmp, [](TElem a, TElem b) {
            return a / b;
        });
    }

    friend Tensor<TElem> operator/(const TElem& a, const Tensor<TElem>& b)
    {
        Tensor<TElem> tmp;
        tmp() = a;
        return elementWiseCombination(tmp, b, [](TElem a, TElem b) {
            return a / b;
        });
    }

    std::vector<uint8_t> toByteArray() const
    {
        if(_is_partial_view) {
            throw std::invalid_argument("Cannot save views");
        }
        std::vector<uint8_t> out;
        auto                 shape = _shape.toByteArray();

        out.insert(out.end(), shape.begin(), shape.end());

        uint8_t* ptr = reinterpret_cast<uint8_t*>(&(*_data)[0]);

        for(size_t i = 0; i < _data->size() * sizeof(TElem); i++) {
            out.push_back(ptr[i]);
        }

        return out;
    }

    template<typename Iterator>
    size_t fromByteArray(Iterator begin, Iterator end)
    {
        Index  shape;
        size_t bytes_read = shape.fromByteArray(begin, end);

        fillDims(shape);

        if((end - begin - bytes_read) / sizeof(TElem) < size()) {
            throw std::range_error("fromByteArray: Invalid array size");
        }

        begin += bytes_read;
        const TElem* ptr = reinterpret_cast<const TElem*>(&*begin);

        for(size_t i = 0; i < _data->size(); i++) {
            (*_data)[i] = ptr[i];
        }
        return bytes_read + _data->size() * sizeof(TElem);
    }

    size_t fromByteArray(const std::vector<uint8_t>& array)
    {
        return fromByteArray(array.begin(), array.end());
    }

    bool saveToBMP(const std::string& filename, TElem minVal = 0, TElem maxVal = 255)
    {

        // From
        // https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries

        if(NDims() < 3) {
            throw(std::invalid_argument("Require at least 3 dimension to save a bmp"));
        }
        int           height = _shape[-3];
        int           width  = _shape[-2];
        unsigned char image[height][width][BYTES_PER_PIXEL];
        const char*   imageFileName = filename.c_str();

        int i, j;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {

                if(_shape[-1] < 3) {
                    uint8_t val = uint8_t(std::round((double((*this)(i, j, -1)) - minVal) * 255. /
                                                     (maxVal - minVal)));

                    // For now: Use colored images for monochrome data...
                    image[height - i - 1][j][2] = val; /// red
                    image[height - i - 1][j][1] = val; /// green
                    image[height - i - 1][j][0] = val; /// blue
                }
                else {
                    image[height - i - 1][j][2] = uint8_t(std::round(
                        (double((*this)(i, j, -3)) - minVal) * 255. / (maxVal - minVal))); /// red
                    image[height - i - 1][j][1] = uint8_t(std::round(
                        (double((*this)(i, j, -2)) - minVal) * 255. / (maxVal - minVal))); /// green
                    image[height - i - 1][j][0] = uint8_t(std::round(
                        (double((*this)(i, j, -1)) - minVal) * 255. / (maxVal - minVal))); /// blue
                }
            }
        }

        return generateBitmapImage((unsigned char*)image, height, width, imageFileName);
    }
};

template<typename TElemA, typename TElemB, typename Op>
Tensor<TElemA> elementWiseCombination(const Tensor<TElemA>& a, const Tensor<TElemB>& b, Op op)
{
    long NDims_smaller = std::min(a.NDims(), b.NDims());
    for(long i = 1; i <= NDims_smaller; i++) {
        if(a.shape(-i) != b.shape(-i)) {
            throw std::invalid_argument("Operatr *=: Dimension mismatch. Shape at " +
                                        std::to_string(-i) +
                                        " is unequal: " + std::to_string(a.shape(-i)) + " vs " +
                                        std::to_string(b.shape(-i)));
        }
    }

    if(a.NDims() > b.NDims()) {

        long ref_axis = -b.NDims();
        if(b.isScalar()) {
            ref_axis = a.NDims();
        }
        Tensor<TElemA> out(a.shape());
        Tensor<TElemA> out_view = out.viewFromIndices({ref_axis});
        Tensor<TElemA> a_view   = a.viewFromIndices({ref_axis});
        Tensor<TElemB> b_view   = b.viewWithNDimsOnTheLeft(1);

        for(size_t i = 0; i < a_view.shape(0); i++) {
            for(size_t j = 0; j < a_view.shape(1); j++) {
                out_view(i, j) = op(a_view(i, j), b_view(j));
            }
        }
        return out;
    }
    else {
        long ref_axis = -a.NDims();
        if(a.isScalar()) {
            ref_axis = b.NDims();
        }

        Tensor<TElemA> out(b.shape());
        Tensor<TElemA> out_view = out.viewFromIndices({ref_axis});
        Tensor<TElemB> b_view   = b.viewFromIndices({ref_axis});
        Tensor<TElemA> a_view   = a.viewWithNDimsOnTheLeft(1);

        for(size_t i = 0; i < b_view.shape(0); i++) {
            for(size_t j = 0; j < b_view.shape(1); j++) {
                out_view(i, j) = op(a_view(j), b_view(i, j));
            }
        }
        return out;
    }
}

inline void checkNan(std::string identifier, const Tensor<float>& t)
{
    for(float& val : t) {
        if(std::isinf(val) || std::isnan(val)) {
            throw(std::domain_error("Found nan at " + identifier));
        }
    }
}

} // namespace snnl