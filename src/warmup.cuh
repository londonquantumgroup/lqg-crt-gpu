#ifndef WARMUP_CUH
#define WARMUP_CUH

#include "types.hpp"

// GPU warmup functions
void warmup_gpu_context();
void warmup_crt_kernel_32(const u32* d_c, const u32* d_m, int k_used);
void warmup_crt_kernel_64(const u32* d_c, const u32* d_m, int k_used);

#endif // WARMUP_CUH