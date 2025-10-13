#include "types.hpp"
#include "timing.hpp"
#include "cuda_utils.hpp"
#include "gpu_kernels.cuh"
#include <vector>

void warmup_gpu_context() {
    auto t_warm_start = now_tp();
    cudaFree(0);
    cudaDeviceSynchronize();
    double warmup_ms = ms_since(t_warm_start);
    printf("[warmup] GPU context initialized in %.2f ms\n", warmup_ms);
}

void warmup_crt_kernel_32(const u32* d_c, const u32* d_m, int k_used) {
    auto t_warm_start = now_tp();
    
    const int WARMUP_SIZE = 1000;
    u32 *d_P_warm = nullptr, *d_out_warm = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_P_warm, WARMUP_SIZE * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_out_warm, WARMUP_SIZE * sizeof(u32)));
    
    // Fill with dummy divisors
    std::vector<u32> dummy_divisors(WARMUP_SIZE);
    for (int i = 0; i < WARMUP_SIZE; ++i) {
        dummy_divisors[i] = 1000003 + i;
    }
    CUDA_CHECK(cudaMemcpy(d_P_warm, dummy_divisors.data(), 
                         WARMUP_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (WARMUP_SIZE + threads - 1) / threads;
    
    // Run twice to warm up all caches
    remainders_via_crt_32<<<blocks, threads>>>(
        d_P_warm, d_out_warm, WARMUP_SIZE, d_c, d_m, k_used
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    remainders_via_crt_32<<<blocks, threads>>>(
        d_P_warm, d_out_warm, WARMUP_SIZE, d_c, d_m, k_used
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_P_warm));
    CUDA_CHECK(cudaFree(d_out_warm));
    
    double warmup_ms = ms_since(t_warm_start);
    printf("[warmup] CRT-32 kernel warmed up in %.2f ms\n", warmup_ms);
}

void warmup_crt_kernel_64(const u32* d_c, const u32* d_m, int k_used) {
    auto t_warm_start = now_tp();
    
    const int WARMUP_SIZE = 1000;
    u64 *d_P_warm = nullptr, *d_out_warm = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_P_warm, WARMUP_SIZE * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_out_warm, WARMUP_SIZE * sizeof(u64)));
    
    std::vector<u64> dummy_divisors(WARMUP_SIZE);
    for (int i = 0; i < WARMUP_SIZE; ++i) {
        dummy_divisors[i] = 10000000000003ULL + i;
    }
    CUDA_CHECK(cudaMemcpy(d_P_warm, dummy_divisors.data(), 
                         WARMUP_SIZE * sizeof(u64), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (WARMUP_SIZE + threads - 1) / threads;
    
    remainders_via_crt_64<<<blocks, threads>>>(
        d_P_warm, d_out_warm, WARMUP_SIZE, d_c, d_m, k_used
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    remainders_via_crt_64<<<blocks, threads>>>(
        d_P_warm, d_out_warm, WARMUP_SIZE, d_c, d_m, k_used
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_P_warm));
    CUDA_CHECK(cudaFree(d_out_warm));
    
    double warmup_ms = ms_since(t_warm_start);
    printf("[warmup] CRT-64 kernel warmed up in %.2f ms\n", warmup_ms);
}