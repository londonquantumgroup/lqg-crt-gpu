#pragma once
#include "types.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <string>

using boost::multiprecision::cpp_int;

static inline cpp_int read_big_decimal(const std::string &s) {
    cpp_int N = 0;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') {
            N = N * 10 + (ch - '0');
        }
    }
    return N;
}

static inline size_t bitlen_cppint(const cpp_int &x) {
    cpp_int t = x;
    size_t bits = 0;
    while (t > 0) {
        t >>= 1;
        ++bits;
    }
    return bits ? bits : 1;
}