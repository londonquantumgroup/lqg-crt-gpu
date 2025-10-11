#pragma once
#include "crt_utils.hpp"

// sieve + global primes
inline std::vector<u32> sieve_primes(u32 limit) {
  const u32 SIEVE_MAX = 50000000u;
  if (limit > SIEVE_MAX) limit = SIEVE_MAX;
  std::vector<char> is(limit + 1, true);
  if (limit >= 0) is[0] = false;
  if (limit >= 1) is[1] = false;
  for (u32 p = 2; (uint64_t)p * p <= limit; ++p) if (is[p])
    for (u32 q = p * p; q <= limit; q += p) is[q] = false;
  std::vector<u32> out; out.reserve(limit/10);
  for (u32 i=2;i<=limit;++i) if (is[i]) out.push_back(i);
  return out;
}

inline const std::vector<u32>& global_primes() {
  static std::vector<u32> gp = sieve_primes(50000000u);
  return gp;
}

// host mirror for GPU divisor generation sequences
inline u32 divisor_at_32(u32 base_seed, int idx) {
  return (u32)(( (uint64_t)idx * 104729u + base_seed) | 1u);
}
inline u64 divisor_at_64(u64 base_seed, int idx) {
  return ( (u64)idx * 104729ull + base_seed) | 1ull;
}