#pragma once
#include "crt_utils.hpp"
#include <vector>

void run_cgbn_benchmark(
    const cpp_int& N,
    const std::vector<cpp_int>& divisors,
    int M,
    double ms_per_cand_cpu);