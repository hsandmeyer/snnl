#pragma once
#include <initializer_list>
#include <ostream>
#include <vector>

class Index {
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

    Index(std::initializer_list<size_t> list) : _shape(list) {}

    Index(size_t size) : _shape(size) {}

    Index() = default;

    Index(Index&) = default;

    Index(const Index&) = default;

    Index(Index&&) = default;

    Index& operator=(Index&&) = default;

    Index& operator=(const Index&) = default;

    auto begin() { return _shape.begin(); }

    auto end() { return _shape.end(); }

    auto begin() const { return _shape.begin(); }

    auto end() const { return _shape.end(); }

    auto cbegin() const { return _shape.cbegin(); }

    auto cend() const { return _shape.cend(); }

    auto rbegin() const { return _shape.rbegin(); }

    auto rend() const { return _shape.rend(); }

    auto size() const { return _shape.size(); }

    void appendAxis(size_t i) { _shape.push_back(i); }

    void prependAxis(size_t i) { _shape.insert(_shape.begin(), i); }

    void removeDim() { _shape.pop_back(); }

    void setNDims(size_t NDims) { _shape.resize(NDims); }

    size_t NDims() const { return _shape.size(); }

    bool operator!=(const Index& b) const { return _shape != b._shape; }

    bool operator==(const Index& b) const { return _shape == b._shape; }

    friend std::ostream& operator<<(std::ostream& o, Index ind)
    {
        o << "{";
        for (size_t i = 0; i < ind._shape.size(); i++) {
            o << ind[i];
            if (i < ind.size() - 1) {
                o << ", ";
            }
        }
        o << "}";
        return o;
    }
};