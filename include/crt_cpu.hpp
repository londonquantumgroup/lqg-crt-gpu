#pragma once
#include <vector>
#include <unordered_set>
#include <boost/multiprecision/cpp_int.hpp>
using boost::multiprecision::cpp_int;
using u32 = uint32_t;

std::vector<cpp_int> generate_divisors(
    int M,
    int divisor_bits,
    const std::unordered_set<u32>& exclude_set);