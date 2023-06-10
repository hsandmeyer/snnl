#pragma once
#include <string>

namespace snnl
{
inline bool ends_with(std::string const& value, std::string const& ending)
{
    if(ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline std::string append_if_not_endswith(const std::string& str, const std::string& end)
{
    if(not ends_with(str, end)) {
        return str + end;
    }
    return str;
}
} // namespace snnl