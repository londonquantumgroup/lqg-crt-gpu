#pragma once
#include "types.hpp"
#include <vector>

std::vector<u32> sieve_primes(u32 limit);
extern std::vector<u32> global_primes;