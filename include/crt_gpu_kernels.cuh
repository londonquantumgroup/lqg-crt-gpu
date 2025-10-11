#pragma once
#include "crt_utils.hpp"

// device helpers
__device__ __forceinline__ u32 mul_mod_u32(u32 a, u32 b, u32 mod) {
  unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
  return (u32)(prod % mod);
}
__device__ __forceinline__ u64 mul_mod_u64(u64 a, u64 b, u64 mod) {
  unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
  return (u64)(prod % mod);
}

// kernels
__global__ void generate_divisors_kernel_32(u32 base_seed, int M, u32* out);
__global__ void generate_divisors_kernel_64(u64 base_seed, int M, u64* out);

__global__ void remainders_via_crt_32(const u32* __restrict__ P,
                                      u32* __restrict__ out,
                                      int M,
                                      const u32* __restrict__ c,
                                      const u32* __restrict__ m,
                                      int k);

__global__ void remainders_via_crt_64(const u64* __restrict__ P,
                                      u64* __restrict__ out,
                                      int M,
                                      const u32* __restrict__ c,
                                      const u32* __restrict__ m,
                                      int k);