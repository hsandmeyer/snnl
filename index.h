#include <initializer_list>
#include <vector>

class TIndex {
    std::vector<size_t> _shape;

public:
    template <typename Integer,
              std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    size_t &operator[](Integer i)
    {
        return _shape[i];
    }

    template <typename Integer,
              std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    size_t &operator[](Integer i)
    {
        if (i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
    }

    template <typename Integer,
              std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    const size_t &operator[](Integer i) const
    {
        return _shape[i];
    }

    template <typename Integer,
              std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    const size_t &operator[](Integer i) const
    {
        if (i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
    }

    TIndex(std::initializer_list<size_t> list) : _shape(list) {}

    template <typename... TArgs>
    TIndex(TArgs &&... args) : _shape(std::forward<TArgs>(args)...)
    {
    }

    TIndex() = default;

    TIndex(TIndex &) = default;

    TIndex(const TIndex &) = default;

    TIndex(TIndex &&) = default;

    auto begin() { return _shape.begin(); }

    auto end() { return _shape.end(); }

    auto cbegin() const { return _shape.cbegin(); }

    auto cend() const { return _shape.cend(); }

    auto rbegin() const { return _shape.rbegin(); }

    auto rend() const { return _shape.rend(); }

    void addDim(size_t i) { _shape.push_back(i); }

    void removeDim() { _shape.pop_back(); }

    void setNDims(size_t NDims) { _shape.resize(NDims); }

    size_t NDims() { return _shape.size(); }
};