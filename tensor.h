#include <array>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace snnl {
template <class TElem, class TContainer = std::vector<TElem>>
class TTensor {

    int                 _NDims;
    std::vector<size_t> _dims;
    std::vector<size_t> _strides;
    TContainer          _data = {};

    template <typename TArray>
    void fillDims(const TArray &dims)
    {
        if (_NDims < 1) {
            throw std::invalid_argument("At least on dimension needed");
        }

        std::cout << "Constructing tensor with ";
        int i = 0;
        for (auto dim_len : dims) {
            std::cout << dim_len << " ";
            _dims[i] = dim_len;
            i++;
        }
        std::cout << std::endl;

        _strides[_NDims - 1] = 1;

        for (int i = _NDims - 2; i >= 0; --i) {
            _strides[i] = _dims[i + 1] * _strides[i + 1];
        }

        _data.resize(_strides[0] * _dims[0]);
    }

    template <size_t N, typename... Ts>
    size_t dataOffset(size_t i, Ts... im)
    {
        return i * _strides[N] + dataOffset<N + 1>(im...);
    }

    template <size_t N>
    size_t dataOffset(size_t j)
    {
        return j;
    }

public:
    typedef typename TContainer::iterator       iterator;
    typedef typename TContainer::const_iterator const_iterator;

    TTensor(const std::initializer_list<size_t> dims)
        : _NDims(dims.size()), _dims(_NDims), _strides(_NDims)
    {
        fillDims(dims);
    }

    template <size_t NDims>
    TTensor(const std::array<size_t, NDims> &dims)
        : _NDims(NDims), _dims(_NDims), _strides(_NDims)
    {
        fillDims(dims);
    }

    template <int NDims>
    TTensor(const std::array<size_t, NDims> dims)
        : _NDims(NDims), _dims(_NDims), _strides(_NDims)
    {
        fillDims(dims);
    }

    template <typename... T>
    TElem &operator()(T... indices)
    {
        return _data[dataOffset<0>(indices...)];
    }

    template <typename... T>
    const TElem &operator()(T... indices) const
    {
        return _data[dataOffset<0>(indices...)];
    }

    auto begin() { return _data.begin(); }

    auto end() { return _data.end(); }
};
} // namespace snnl