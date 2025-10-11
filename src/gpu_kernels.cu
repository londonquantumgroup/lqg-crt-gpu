#include "gpu_kernels.cuh"
#include <stdint.h>

__device__ __forceinline__ u32 mul_mod_u32(u32 a, u32 b, u32 mod) {
    unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
    return (u32)(prod % mod);
}

__device__ __forceinline__ u32 mul_mod_u32_safe(u32 a, u32 b, u32 mod) {
    unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
    return (u32)(prod % mod);
}

__device__ __forceinline__ u64 mul_mod_u64(u64 a, u64 b, u64 mod) {
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    return (u64)(prod % mod);
}

__global__ void generate_divisors_kernel_32(u32 base_seed, int M, u32* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    u32 val = (u32)(((uint64_t)idx * 104729u + base_seed) | 1u);
    out[idx] = val;
}

__global__ void generate_divisors_kernel_64(u64 base_seed, int M, u64* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    u64 val = ((u64)idx * 104729ull + base_seed) | 1ull;
    out[idx] = val;
}

__global__ void remainders_via_crt_32(const u32* __restrict__ P,
                                      u32* __restrict__ out,
                                      int M,
                                      const u32* __restrict__ c,
                                      const u32* __restrict__ m,
                                      int k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M) return;
    u32 p = P[idx];
    if (p < 2u) {
        out[idx] = 0u;
        return;
    }

    u32 rem = c[0] % p;
    u64 prefix = (u64)(m[0] % p);
#pragma unroll 4
    for (int i = 1; i < k; ++i) {
        if (prefix == 0ull) break;
        u32 ci = c[i] % p;
        u64 term = ((u64)ci * prefix) % p;
        rem = (u32)(((u64)rem + term) % p);
        if (i + 1 < k) {
            u64 mi = m[i] % p;
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
                                      int k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M) return;
    u64 p = P[idx];
    if (p < 2u) {
        out[idx] = 0ull;
        return;
    }
    
    u64 rem = (u64)(c[0]) % p;
    u64 prefix = (u64)(m[0]) % p;
#pragma unroll 4
    for (int i = 1; i < k; ++i) {
        if (prefix == 0ull) break;
        u64 ci = (u64)(c[i]) % p;
        u64 term = mul_mod_u64(ci, prefix, p);
        rem = (rem + term) % p;
        if (i + 1 < k) {
            u64 mi = (u64)(m[i]) % p;
            if (mi == 0ull) {
                prefix = 0ull;
            } else {
                prefix = mul_mod_u64(prefix, mi, p);
            }
        }
    }
    out[idx] = rem;
}

#ifndef NO_CGBN
__global__ void cgbn_divrem_kernel(cgbn_error_report_t *report,
                                   const cgbn_bn_mem_t *d_N_single,
                                   cgbn_bn_mem_t *divs,
                                   cgbn_bn_mem_t *qouts,
                                   cgbn_bn_mem_t *routs,
                                   int count)
{
    cgbn_context ctx(cgbn_no_checks, report, 0);
    cgbn_env env(ctx);
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / CGBN_TPI;
    if (instance >= count) return;
    
    cgbn_bn_t n, d, q, r;
    cgbn_load(env, n, (cgbn_bn_mem_t*)d_N_single);
    cgbn_load(env, d, &divs[instance]);
    
    if (cgbn_compare_ui32(env, d, 0) == 0) {
        cgbn_set_ui32(env, q, 0);
        cgbn_set_ui32(env, r, 0);
    } else {
        cgbn_div_rem(env, q, r, n, d);
    }
    
    cgbn_store(env, &qouts[instance], q);
    cgbn_store(env, &routs[instance], r);
}
#endif