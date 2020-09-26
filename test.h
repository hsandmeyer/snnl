#pragma once
#include "cpplap.h"
#include <iostream>
#include <string>

namespace cpplap {

template <typename T> bool test_exact(const T &res, const T &ref, const std::string test_name)
{
    if (res == ref) {
        std::cout << test_name << " test passed\n";
        return true;
    }
    else {
        std::cout << test_name << " test Failed\nresult was\n" << res << "\nReference is\n" << ref << std::endl;
        return false;
    }
}

template <typename T> bool test_rel(const T &res, const T &ref, const std::string test_name)
{
    if (cmp_rel(res, ref)) {
        std::cout << test_name << " test passed\n";
        return true;
    }
    else {
        std::cout << test_name << " test Failed\nresult was\n" << res << "\nReference is\n" << ref << std::endl;
        return false;
    }
}
} // namespace cpplap
