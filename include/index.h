#pragma once
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

class Index
{
    std::vector<size_t> _shape;

public:
    template<typename Integer, std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    size_t& operator[](Integer i)
    {
#ifdef DEBUG
        return _shape.at(i);
#else
        return _shape[i];
#endif
    }

    template<typename Integer, std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    size_t& operator[](Integer i)
    {
#ifdef DEBUG
        if(i < 0) {
            return _shape.at(_shape.size() + i);
        }
        return _shape.at(i);
#else
        if(i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
#endif
    }

    template<typename Integer, std::enable_if_t<std::is_unsigned<Integer>::value, int> = 0>
    const size_t& operator[](Integer i) const
    {
#ifdef DEBUG
        return _shape.at(i);
#else
        return _shape[i];
#endif
    }

    template<typename Integer, std::enable_if_t<std::is_signed<Integer>::value, int> = 0>
    const size_t& operator[](Integer i) const
    {
#ifdef DEBUG
        if(i < 0) {
            return _shape.at(_shape.size() + i);
        }
        return _shape.at(i);
#else
        if(i < 0) {
            return _shape[_shape.size() + i];
        }
        return _shape[i];
#endif
    }

    Index(std::initializer_list<size_t> list)
        : _shape(list)
    {
    }

    Index(size_t size)
        : _shape(size)
    {
    }

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

    long NDims() const { return _shape.size(); }

    Index copyNDims(size_t NDims)
    {
        Index out(NDims);

        for(size_t i = 0; i < out.size(); i++) {
            if(i < size()) {
                out[i] = _shape[i];
            }
            else {
                out[i] = 1;
            }
        }
        return out;
    }

    bool operator!=(const Index& b) const { return _shape != b._shape; }

    bool operator==(const Index& b) const { return _shape == b._shape; }

    friend std::ostream& operator<<(std::ostream& o, Index ind)
    {
        o << "{";
        for(size_t i = 0; i < ind._shape.size(); i++) {
            o << ind[i];
            if(i < ind.size() - 1) {
                o << ", ";
            }
        }
        o << "}";
        return o;
    }

    friend std::string operator+(std::string s, Index ind)
    {
        std::stringstream str;
        str << s;
        str << ind;
        return str.str();
    }

    friend std::string operator+(Index ind, std::string s)
    {
        std::stringstream str;
        str << ind;
        str << s;
        return str.str();
    }

    std::vector<uint8_t> toByteArray() const
    {
        std::vector<uint8_t> out;
        size_t               size = _shape.size();
        out.reserve(_shape.size() * sizeof(size_t) + sizeof(size));

        const uint8_t* ptr = reinterpret_cast<uint8_t*>(&size);

        for(size_t i = 0; i < sizeof(size); i++) {
            out.push_back(ptr[i]);
        }

        if(size > 0) {
            ptr = reinterpret_cast<const uint8_t*>(&_shape[0]);

            for(size_t i = 0; i < sizeof(size_t) * size; i++) {
                out.push_back(ptr[i]);
            }
        }
        return out;
    }

    size_t fromByteArray(const std::vector<uint8_t>& array)
    {
        return fromByteArray(array.begin(), array.end());
    }

    template<typename Iterator>
    size_t fromByteArray(Iterator begin, Iterator end)
    {
        size_t array_size = begin - end;

        if(array_size < sizeof(size_t)) {
            throw std::range_error("Index: fromByteArray - invalid array size)");
        }

        const size_t NElems = *(reinterpret_cast<const size_t*>(&*begin));
        _shape.resize(NElems);
        const size_t* ptr;

        if(NElems > 0) {

            if((array_size - sizeof(size_t)) / sizeof(size_t) < _shape.size()) {
                throw std::range_error("Index: fromByteArray - invalid array size)");
            }

            begin += sizeof(size_t);
            ptr = reinterpret_cast<const size_t*>(&*begin);

            for(size_t i = 0; i < NElems; i++) {
                _shape[i] = ptr[i];
            }
        }
        return sizeof(size_t) + NElems * sizeof(size_t);
    }
};