#pragma once
#include <initializer_list>
#include <vector>

class TIndex {
    std::vector<size_t> _shape;

public:
    template <typename Integer,
              std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    size_t& operator[](Integer i)
    {
#ifdef DEBUG
        return _shape.at(i);
#else
        return _shape[i];
#endif
    }

    template <typename Integer,
              std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    size_t& operator[](Integer i)
    {
#ifdef DEBUG
        if (i < 0) {
            return _shape.at(_shape.size() + i);
        }
        return _shape.at(i);
#else
        if (i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
#endif
    }

    template <typename Integer,
              std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    const size_t& operator[](Integer i) const
    {
#ifdef DEBUG
        return _shape.at(i);
#else
        return _shape[i];
#endif
    }

    template <typename Integer,
              std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    const size_t& operator[](Integer i) const
    {
#ifdef DEBUG
        if (i < 0) {
            return _shape.at(_shape.size() + i);
        }
        return _shape.at(i);
#else
        if (i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
#endif
    }

    TIndex(std::initializer_list<size_t> list) : _shape(list) {}

    TIndex(size_t size) : _shape(size) {}

    TIndex() = default;

    TIndex(TIndex&) = default;

    TIndex(const TIndex&) = default;

    TIndex(TIndex&&) = default;

    auto begin() { return _shape.begin(); }

    auto end() { return _shape.end(); }

    auto begin() const { return _shape.begin(); }

    auto end() const { return _shape.end(); }

    auto cbegin() const { return _shape.cbegin(); }

    auto cend() const { return _shape.cend(); }

    auto rbegin() const { return _shape.rbegin(); }

    auto rend() const { return _shape.rend(); }

    auto size() const { return _shape.size(); }

    void addDim(size_t i) { _shape.push_back(i); }

    void removeDim() { _shape.pop_back(); }

    void setNDims(size_t NDims) { _shape.resize(NDims); }

    size_t NDims() const { return _shape.size(); }

    bool operator!=(const TIndex& b) { return _shape != b._shape; }
};