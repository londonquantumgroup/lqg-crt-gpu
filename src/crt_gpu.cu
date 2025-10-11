#include "../include/crt_gpu_kernels.cuh"

__global__ void generate_divisors_kernel_32(u32 base_seed, int M, u32* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  out[idx] = ((uint64_t)idx * 104729u + base_seed) | 1u;
}

__global__ void generate_divisors_kernel_64(u64 base_seed, int M, u64* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  out[idx] = ((uint64_t)idx * 104729ull + base_seed) | 1ull;
}

__global__ void remainders_via_crt_32(const u32* __restrict__ P,
                                      u32* __restrict__ out,
                                      int M,
                                      const u32* __restrict__ c,
                                      const u32* __restrict__ m,
                                      int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  u32 p = P[idx];
  if (p < 2u) { out[idx] = 0u; return; }

  u32 rem = c[0] % p;
  unsigned long long prefix = (unsigned long long)(m[0] % p);
  #pragma unroll 4
  for (int i=1;i<k;++i) {
    if (prefix == 0ull) break;
    u32 ci = c[i] % p;
    unsigned long long term = ( (unsigned long long)ci * prefix ) % p;
    rem = (u32)(((unsigned long long)rem + term) % p);
    if (i+1 < k) {
      unsigned long long mi = m[i] % p;
      prefix = (prefix * mi) % p;
    }
  }
  out[idx] = rem;
}

__global__ void remainders_via_crt_64(const u64* __restrict__ P,
                                      u64* __restrict__ out,
                                      int M,
                                      const u32* __restrict__ c,
                                      const u32* __restrict__ m,
                                      int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  u64 p = P[idx];
  if (p < 2u) { out[idx] = 0ull; return; }

  u64 rem = (u64)c[0] % p;
  u64 prefix = (u64)m[0] % p;
  #pragma unroll 4
  for (int i=1;i<k;++i) {
    if (prefix == 0ull) break;
    u64 ci = (u64)c[i] % p;
    u64 term = mul_mod_u64(ci, prefix, p);
    rem = (rem + term) % p;
    if (i+1 < k) {
      u64 mi = (u64)m[i] % p;
      prefix = (mi == 0ull) ? 0ull : mul_mod_u64(prefix, mi, p);
    }
  }
  out[idx] = rem;
}