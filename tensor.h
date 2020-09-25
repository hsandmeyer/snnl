#include <array>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace snnl {
template <class TElem, class TContainer = std::vector<TElem>>
class TTensor {

    int                 _NDims;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    TContainer          _data = {};

    template <typename TArray>
    void fillDims(const TArray &shape)
    {
        std::cout << "Constructing tensor with ";
        _shape.resize(_NDims);

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
        _strides.resize(_NDims);
        _strides[_NDims - 1] = 1;

        for (int i = _NDims - 2; i >= 0; --i) {
            _strides[i] = _shape[i + 1] * _strides[i + 1];
        }

        _data.resize(NElems());
    }

    template <size_t N, typename... Ts>
    size_t dataOffset(size_t i, Ts... im)
    {
        return i * _strides[_strides.size() - sizeof...(Ts) - 1] +
               dataOffset<N + 1>(im...);
    }

    template <size_t N>
    size_t dataOffset(size_t j)
    {
        return j;
    }

public:
    typedef typename TContainer::iterator       iterator;
    typedef typename TContainer::const_iterator const_iterator;

    TTensor() : _NDims(0) {}

    TTensor(const std::initializer_list<size_t> shape)
        : _NDims(shape.size()), _shape(_NDims), _strides(_NDims)
    {
        fillDims(shape);
    }

    template <size_t NDims>
    TTensor(const std::array<size_t, NDims> &shape)
        : _NDims(NDims), _shape(_NDims), _strides(_NDims)
    {
        fillDims(shape);
    }

    template <int NDims>
    TTensor(const std::array<size_t, NDims> &shape)
        : _NDims(NDims), _shape(_NDims), _strides(_NDims)
    {
        fillDims(shape);
    }

    template <typename TArray>
    TTensor(const TArray &shape)
        : _NDims(shape.size()), _shape(_NDims), _strides(_NDims)
    {
        fillDims(shape);
    }

    template <typename TArray>
    void setDims(const TArray &arr)
    {
        _NDims = arr.size();
        fillDims(arr);
    }

    void addDim(const size_t dim_len)
    {
        _NDims++;
        _shape.push_back(dim_len);
        _strides.push_back(0);

        fillStrides();
    }

    int NDims() const { return _NDims; }

    size_t NElems() { return _strides[0] * _shape[0]; }

    // Length of axis i, if all following axis ( > i) are flattened
    // into i
    size_t shapeFlattened(const int i) { return NElems() / stride(i); }

    template <typename... T>
    TElem &operator()(T... indices)
    {
#ifdef DEBUG
        return _data.at(dataOffset<0>(indices...));
#else
        return _data[dataOffset<0>(indices...)];
#endif
    }

    template <typename... T>
    const TElem &operator()(T... indices) const
    {
#ifdef DEBUG
        return _data.at(dataOffset<0>(indices...));
#else
        return _data[dataOffset<0>(indices...)];
#endif
    }

    size_t shape(const int i) const
    {
        if (i >= 0) {
            return _shape[i];
        }
        else {
            return _shape[_NDims + i];
        }
    }

    size_t stride(const int i) const
    {
        if (i >= 0) {
            return _strides[i];
        }
        else {
            return _strides[_NDims + i];
        }
    }

    std::vector<size_t> &shape() { return _shape; }

    const std::vector<size_t> &shape() const { return _shape; }

    std::vector<size_t> subShape(int first, int last) const
    {
        return std::vector<size_t>(_shape.begin() + first,
                                   _shape.begin() + first + last);
    }

    auto begin() { return _data.begin(); }

    auto end() { return _data.end(); }
};
} // namespace snnl