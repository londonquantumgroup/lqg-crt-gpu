#include "divisors.hpp"
#include "primes.hpp"

u32 divisor_at_32(u32 base_seed, int idx) {
    return (u32)(((uint64_t)idx * 104729u + base_seed) | 1u);
}

u64 divisor_at_64(u64 base_seed, int idx) {
    return (((u64)idx * 104729ull + base_seed) | 1ull);
}

std::vector<cpp_int> generate_divisors(int M, int divisor_bits, 
                                       const std::unordered_set<u32>& exclude_set) {
    std::vector<cpp_int> divisors;
    divisors.reserve(M);
    
    if (divisor_bits == 32) {
        for (size_t i = 0; i < global_primes.size() && (int)divisors.size() < M; ++i) {
            u32 p = global_primes[i];
            if (exclude_set.find(p) == exclude_set.end()) {
                divisors.push_back(cpp_int(p));
            }
        }
    } else if (divisor_bits == 64) {
        for (size_t i = 0; i < global_primes.size() && (int)divisors.size() < M; i += 2) {
            if (i + 1 >= global_primes.size()) break;
            u64 val = ((u64)global_primes[i] << 32) | global_primes[i + 1];
            divisors.push_back(cpp_int(val));
        }
    } else {
        // For 128+ bits: combine multiple primes
        cpp_int base = cpp_int(1) << (divisor_bits - 1);
        for (int i = 0; i < M; ++i) {
            cpp_int offset = cpp_int(i * 104729 + 1234567);
            cpp_int divisor = base + offset;
            if (divisor % 2 == 0) divisor += 1;
            divisors.push_back(divisor);
        }
    }
    return divisors;
}