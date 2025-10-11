#pragma once
#include "types.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// Device helper functions
__device__ __forceinline__ u32 mul_mod_u32(u32 a, u32 b, u32 mod);
__device__ __forceinline__ u32 mul_mod_u32_safe(u32 a, u32 b, u32 mod);
__device__ __forceinline__ u64 mul_mod_u64(u64 a, u64 b, u64 mod);

// Divisor generation kernels
__global__ void generate_divisors_kernel_32(u32 base_seed, int M, u32* out);
__global__ void generate_divisors_kernel_64(u64 base_seed, int M, u64* out);

// CRT kernels
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

// Note: cgbn_divrem_kernel is declared and implemented only in main.cu
// to avoid CGBN multiple definition issues