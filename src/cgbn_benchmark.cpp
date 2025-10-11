#include "../include/crt_cgbn.hpp"
#include "../include/crt_cgbn_kernels.cuh"
#include "../include/crt_utils.hpp"
#include <vector>

#ifdef ENABLE_CGBN

void run_cgbn_benchmark(
    const cpp_int& N,
    const std::vector<cpp_int>& divisors,
    int M,
    double ms_per_cand_cpu)
{
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_bytes = M * sizeof(cgbn_bn_mem_t) * 3;

    if (required_bytes > free_mem * 0.8) {
        printf("[CGBN div_rem] Skipping: M=%d requires ~%.2f GB, only %.2f GB free.\n",
               M, required_bytes / 1e9, free_mem / 1e9);
        return;
    }

    size_t nbits = bitlen_cppint(N);
    size_t max_div_bits = 0;
    for (int i = 0; i < M; ++i) {
        size_t db = bitlen_cppint(divisors[i]);
        if (db > max_div_bits) max_div_bits = db;
    }

    if(nbits > CGBN_BITS || max_div_bits > CGBN_BITS) {
        size_t needed = std::max(nbits, max_div_bits);
        fprintf(stderr, "[CGBN div_rem] Skipping: Need %zu bits, CGBN_BITS=%d. "
                "Rebuild with -DCGBN_BITS=%zu\n", 
                needed, CGBN_BITS, ((needed+31)/32)*32);
        return;
    }

    printf("[CGBN div_rem] Running comparison benchmark\n");

    // Allocate pinned host memory
    cgbn_bn_mem_t *h_divs = nullptr, *h_q = nullptr, *h_r = nullptr;
    cudaHostAlloc(&h_divs, M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_q,   M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_r,   M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);

    // Convert divisors to CGBN format
    for(int i = 0; i < M; ++i) {
        cppint_to_cgbn_mem(divisors[i], h_divs[i]);
    }

    // Prepare N
    cgbn_bn_mem_t N_mem;
    cppint_to_cgbn_mem(N, N_mem);
    cgbn_bn_mem_t *d_N_single = nullptr;
    CUDA_CHECK(cudaMalloc(&d_N_single, sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N_single, &N_mem, sizeof(cgbn_bn_mem_t), 
                         cudaMemcpyHostToDevice));

    // Allocate device memory
    cgbn_bn_mem_t *d_divs = nullptr, *d_qm = nullptr, *d_rm = nullptr;
    cgbn_error_report_t *report = nullptr;

    // Use our wrapper function instead of ifdefs
    cgbn_error_report_alloc_impl(&report);

    CUDA_CHECK(cudaMalloc(&d_divs, M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_qm,   M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_rm,   M * sizeof(cgbn_bn_mem_t)));

    // H2D timing
    auto t_h2d_full_start = now_tp();
    CUDA_CHECK(cudaMemcpy(d_divs, h_divs, M * sizeof(cgbn_bn_mem_t),
                         cudaMemcpyHostToDevice));
    double cgbn_h2d_ms = ms_since(t_h2d_full_start);

    // Kernel timing
    cudaEvent_t c0, c1; 
    CUDA_CHECK(cudaEventCreate(&c0)); 
    CUDA_CHECK(cudaEventCreate(&c1));

    auto t_cgbn_launch_start = now_tp();
    CUDA_CHECK(cudaEventRecord(c0));

    int threads2 = 128;
    if(threads2 % CGBN_TPI) threads2 += (CGBN_TPI - (threads2 % CGBN_TPI));
    int blocks2 = (M * CGBN_TPI + threads2 - 1) / threads2;

    cgbn_divrem_kernel<<<blocks2, threads2>>>(
        report, d_N_single, d_divs, d_qm, d_rm, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(c1));
    CUDA_CHECK(cudaEventSynchronize(c1));

    double cgbn_launch_ms = ms_since(t_cgbn_launch_start);
    float cgbn_divrem_ms = 0.f; 
    CUDA_CHECK(cudaEventElapsedTime(&cgbn_divrem_ms, c0, c1));

    // D2H timing
    auto t_d2h_full_start = now_tp();
    CUDA_CHECK(cudaMemcpy(h_r, d_rm, M * sizeof(cgbn_bn_mem_t),
                         cudaMemcpyDeviceToHost));
    double cgbn_d2h_ms = ms_since(t_d2h_full_start);

    // Verify results
    int mism_cgbn = 0;
    for(int i = 0; i < M; ++i) {
        cpp_int cgbn_rem = cgbn_mem_to_cppint(h_r[i]);
        cpp_int expect = N % divisors[i];
        if(cgbn_rem != expect) {
            if(mism_cgbn < 5) 
                fprintf(stderr, "[CGBN div_rem] Mismatch at %d: "
                        "cpu=%s cgbn=%s\n", 
                        i, expect.str().c_str(), cgbn_rem.str().c_str());
            ++mism_cgbn;
        }
    }

    // Calculate metrics
    double cgbn_mps = (cgbn_divrem_ms > 0) ? 
                      (M / 1e6) / (cgbn_divrem_ms / 1000.0) : 0.0;
    double cgbn_ms_per_cand = (cgbn_divrem_ms > 0) ? 
                              cgbn_divrem_ms / double(M) : 
                              std::numeric_limits<double>::quiet_NaN();
    double Mcross_cgbn = (cgbn_ms_per_cand > 0) ? 
                        (ms_per_cand_cpu / cgbn_ms_per_cand) : -1.0;

    printf("CGBN div_rem | %d divisions | %.3f ms (%.2f M/s) | mismatches=%d\n", 
           M, cgbn_divrem_ms, cgbn_mps, mism_cgbn);
    printf("CGBN div_rem ms per candidate: %.9f | "
           "CPU/CGBN crossover @ M=%.9f\n", 
           cgbn_ms_per_cand, Mcross_cgbn);

    // Full pipeline timing
    double cgbn_full_ms = cgbn_h2d_ms + cgbn_launch_ms + cgbn_d2h_ms;
    double cgbn_full_mps = (cgbn_full_ms > 0.0) ? 
                          ((M / 1e6) / (cgbn_full_ms / 1000.0)) : 0.0;
    printf("CGBN div_rem full | %d divisions | %.3f ms (%.2f M/s) | "
           "h2d=%.2f ms kernel=%.2f ms d2h=%.2f ms\n",
           M, cgbn_full_ms, cgbn_full_mps, 
           cgbn_h2d_ms, (double)cgbn_divrem_ms, cgbn_d2h_ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_divs));
    CUDA_CHECK(cudaFree(d_qm));   
    CUDA_CHECK(cudaFree(d_rm));
    CUDA_CHECK(cudaFree(d_N_single));

    // Use our wrapper function instead of ifdefs
    cgbn_error_report_free_impl(report);

    CUDA_CHECK(cudaEventDestroy(c0));
    CUDA_CHECK(cudaEventDestroy(c1));

    cudaFreeHost(h_divs);
    cudaFreeHost(h_q);
    cudaFreeHost(h_r);
}

#else

void run_cgbn_benchmark(
    const cpp_int&,
    const std::vector<cpp_int>&,
    int,
    double)
{
    // No-op when CGBN disabled
}

#endif // ENABLE_CGBN