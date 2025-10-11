#pragma once
#include "types.hpp"
#include "bigint_utils.hpp"
#include <vector>
#include <unordered_set>

std::vector<cpp_int> generate_divisors(int M, int divisor_bits, 
                                       const std::unordered_set<u32>& exclude_set);

// Host helpers to mirror GPU divisor generation
u32 divisor_at_32(u32 base_seed, int idx);
u64 divisor_at_64(u64 base_seed, int idx);