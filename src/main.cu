// --- Main Application Logic ---
#include "types.hpp"
#include "timing.hpp"
#include "bigint_utils.hpp"
#include "crt.hpp"
#include "divisors.hpp"
#include "cuda_utils.hpp"
#include "gpu_kernels.cuh"
#include "warmup.cuh"

#ifndef NO_CGBN
#include "cgbn_utils.hpp"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <limits>

#ifndef NO_CGBN
__global__ void cgbn_divrem_kernel(cgbn_error_report_t *report,
                                   const cgbn_bn_mem_t *d_N_single,
                                   cgbn_bn_mem_t *divs,
                                   cgbn_bn_mem_t *qouts,
                                   cgbn_bn_mem_t *routs,
                                   int count);
#endif

// ... all helper structs (print_usage, SetupData, DeviceCRTData, etc.) identical ...

void run_crt_benchmark_32(int M, const cpp_int& N,
                         const std::vector<cpp_int>& divisors,
                         const DeviceCRTData& crt_data, int k_used,
                         BenchmarkResults& results,
                         const u64 BASE_SEED_64, const u32 BASE_SEED_32) {
    const int CHUNK_SIZE = 10000000;

    u32 *d_P = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, M * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(u32)));

    if (GPU_GENERATE_DIVISORS) {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        generate_divisors_kernel_32<<<blocks, threads>>>(BASE_SEED_32, M, d_P);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::vector<u32> P_cpu(M);
        for (int i = 0; i < M; ++i) P_cpu[i] = (u32)divisors[i];
        CUDA_CHECK(cudaMemcpy(d_P, P_cpu.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
    }

    // --- Dual-stream setup ---
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // Two pinned host buffers for async D2H copies
    const int BUFFER_COUNT = 2;
    u32* host_buffers[BUFFER_COUNT];
    for (int i = 0; i < BUFFER_COUNT; ++i)
        CUDA_CHECK(cudaHostAlloc(&host_buffers[i], CHUNK_SIZE * sizeof(u32), cudaHostAllocDefault));

    // Aggregate timers
    cudaEvent_t global_start, global_end;
    CUDA_CHECK(cudaEventCreate(&global_start));
    CUDA_CHECK(cudaEventCreate(&global_end));
    CUDA_CHECK(cudaEventRecord(global_start));

    // Launch chunks asynchronously
    for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        int s = (offset / CHUNK_SIZE) % 2;  // ping-pong buffer/stream
        int threads = 256;
        int blocks = (chunk + threads - 1) / threads;

        u32* d_P_chunk = d_P + offset;
        u32* d_out_chunk = d_out + offset;

        remainders_via_crt_32<<<blocks, threads, 0, streams[s]>>>(
            d_P_chunk, d_out_chunk, chunk, crt_data.d_c, crt_data.d_m, k_used
        );
        CUDA_CHECK(cudaGetLastError());

        // async D2H copy (overlaps next kernel)
        CUDA_CHECK(cudaMemcpyAsync(
            host_buffers[s], d_out_chunk, chunk * sizeof(u32),
            cudaMemcpyDeviceToHost, streams[s]
        ));
    }

    CUDA_CHECK(cudaEventRecord(global_end));
    CUDA_CHECK(cudaEventSynchronize(global_end));
    float total_gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, global_start, global_end));
    results.kernel_ms = total_gpu_ms;   // combined overlapped time
    results.d2h_chunks_ms = 0.0;        // absorbed into kernel_ms
    results.total_gpu_ms = total_gpu_ms;

    // --- CPU verification (sequential) ---
    auto t_cpu_start = now_tp();
    std::vector<u32> cpu_res(M);
    if (GPU_GENERATE_DIVISORS) {
        for (int i = 0; i < M; ++i)
            cpu_res[i] = (u32)(N % cpp_int(divisor_at_32(BASE_SEED_32, i)));
    } else {
        for (int i = 0; i < M; ++i)
            cpu_res[i] = (u32)(N % divisors[i]);
    }
    results.total_cpu_ms = ms_since(t_cpu_start);

    CUDA_CHECK(cudaDeviceSynchronize()); // ensure all streams done

    // --- Validation ---
    int mism = 0;
    int buf_idx = 0;
    for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        const u32* gpu_chunk = host_buffers[buf_idx];
        for (int i = 0; i < chunk; ++i)
            if (cpu_res[offset + i] != gpu_chunk[i])
                if (++mism <= 5)
                    fprintf(stderr, "Mismatch at %d: cpu=%u gpu=%u\n",
                            offset + i, cpu_res[offset + i], gpu_chunk[i]);
        buf_idx ^= 1;
    }
    results.total_mism = mism;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(global_start));
    CUDA_CHECK(cudaEventDestroy(global_end));
    for (int i = 0; i < BUFFER_COUNT; ++i) cudaFreeHost(host_buffers[i]);
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_out));
}
void run_crt_benchmark_64(int M, const cpp_int& N,
    const std::vector<cpp_int>& divisors,
    const DeviceCRTData& crt_data, int k_used,
    BenchmarkResults& results,
    const u64 BASE_SEED_64, const u32 BASE_SEED_32) {
const int CHUNK_SIZE = 10000000;

u64 *d_P = nullptr, *d_out = nullptr;
CUDA_CHECK(cudaMalloc(&d_P, M * sizeof(u64)));
CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(u64)));

if (GPU_GENERATE_DIVISORS) {
int threads = 256;
int blocks = (M + threads - 1) / threads;
generate_divisors_kernel_64<<<blocks, threads>>>(BASE_SEED_64, M, d_P);
CUDA_CHECK(cudaDeviceSynchronize());
} else {
std::vector<u64> P_cpu(M);
for (int i = 0; i < M; ++i) P_cpu[i] = (u64)divisors[i];
CUDA_CHECK(cudaMemcpy(d_P, P_cpu.data(), M * sizeof(u64), cudaMemcpyHostToDevice));
}

// --- Dual-stream setup ---
cudaStream_t streams[2];
CUDA_CHECK(cudaStreamCreate(&streams[0]));
CUDA_CHECK(cudaStreamCreate(&streams[1]));

const int BUFFER_COUNT = 2;
u64* host_buffers[BUFFER_COUNT];
for (int i = 0; i < BUFFER_COUNT; ++i)
CUDA_CHECK(cudaHostAlloc(&host_buffers[i], CHUNK_SIZE * sizeof(u64), cudaHostAllocDefault));

// Aggregate timers
cudaEvent_t global_start, global_end;
CUDA_CHECK(cudaEventCreate(&global_start));
CUDA_CHECK(cudaEventCreate(&global_end));
CUDA_CHECK(cudaEventRecord(global_start));

// Launch chunks asynchronously
for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
int chunk = std::min(CHUNK_SIZE, M - offset);
int s = (offset / CHUNK_SIZE) % 2;
int threads = 256;
int blocks = (chunk + threads - 1) / threads;

u64* d_P_chunk = d_P + offset;
u64* d_out_chunk = d_out + offset;

remainders_via_crt_64<<<blocks, threads, 0, streams[s]>>>(
d_P_chunk, d_out_chunk, chunk, crt_data.d_c, crt_data.d_m, k_used
);
CUDA_CHECK(cudaGetLastError());

CUDA_CHECK(cudaMemcpyAsync(
host_buffers[s], d_out_chunk, chunk * sizeof(u64),
cudaMemcpyDeviceToHost, streams[s]
));
}

CUDA_CHECK(cudaEventRecord(global_end));
CUDA_CHECK(cudaEventSynchronize(global_end));
float total_gpu_ms = 0.f;
CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, global_start, global_end));
results.kernel_ms = total_gpu_ms;
results.d2h_chunks_ms = 0.0;
results.total_gpu_ms = total_gpu_ms;

// --- CPU verification (sequential) ---
auto t_cpu_start = now_tp();
std::vector<u64> cpu_res(M);
if (GPU_GENERATE_DIVISORS) {
for (int i = 0; i < M; ++i)
cpu_res[i] = (u64)(N % cpp_int(divisor_at_64(BASE_SEED_64, i)));
} else {
for (int i = 0; i < M; ++i)
cpu_res[i] = (u64)(N % divisors[i]);
}
results.total_cpu_ms = ms_since(t_cpu_start);

CUDA_CHECK(cudaDeviceSynchronize()); // wait for all copies

// --- Validation ---
int mism = 0;
int buf_idx = 0;
for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
int chunk = std::min(CHUNK_SIZE, M - offset);
const u64* gpu_chunk = host_buffers[buf_idx];
for (int i = 0; i < chunk; ++i)
if (cpu_res[offset + i] != gpu_chunk[i])
if (++mism <= 5)
fprintf(stderr, "Mismatch at %d: cpu=%llu gpu=%llu\n",
       offset + i,
       (unsigned long long)cpu_res[offset + i],
       (unsigned long long)gpu_chunk[i]);
buf_idx ^= 1;
}
results.total_mism = mism;

// Cleanup
CUDA_CHECK(cudaEventDestroy(global_start));
CUDA_CHECK(cudaEventDestroy(global_end));
for (int i = 0; i < BUFFER_COUNT; ++i) cudaFreeHost(host_buffers[i]);
CUDA_CHECK(cudaStreamDestroy(streams[0]));
CUDA_CHECK(cudaStreamDestroy(streams[1]));
CUDA_CHECK(cudaFree(d_P));
CUDA_CHECK(cudaFree(d_out));
}
void print_benchmark_results(int M, const BenchmarkResults& results, int k_used, int k) {
    double gpu_mps = (results.total_gpu_ms > 0) ? (M / 1e6) / (results.total_gpu_ms / 1000.0) : 0.0;
    double cpu_mps = (results.total_cpu_ms > 0) ? (M / 1e6) / (results.total_cpu_ms / 1000.0) : 0.0;

    double ms_per_cand_gpu = (results.total_gpu_ms > 0) ? results.total_gpu_ms / double(M) : std::numeric_limits<double>::quiet_NaN();
    double ms_per_cand_cpu = (results.total_cpu_ms > 0) ? results.total_cpu_ms / double(M) : std::numeric_limits<double>::quiet_NaN();
    double Mcross = (ms_per_cand_gpu > 0) ? (ms_per_cand_cpu / ms_per_cand_gpu) : -1.0;

    printf("\n=== Results for M=%d (%d-bit divisors) ===\n", M, DIVISOR_BITS);
    printf("M=%d: ms per candidate: GPU=%.9f, CPU=%.9f | CPU/GPU crossover @ M=%.9f\n",
           M, ms_per_cand_gpu, ms_per_cand_cpu, Mcross);
    printf("[profile] runtime: h2d_chunks=%.2fms kernel=%.2fms d2h_chunks=%.2fms | external: pgen=%.2fms\n",
           results.h2d_chunks_ms, results.kernel_ms, results.d2h_chunks_ms, results.pgen_ms);

    if (DIVISOR_BITS <= 64) {
        double kernel_mps = (results.kernel_ms > 0) ? (M / 1e6) / (results.kernel_ms / 1000.0) : 0.0;
        printf("k=%d%s candidates=%d | CRT-GPU %.3f ms (%.2f M/s) | "
               "CPU %.3f ms (%.2f M/s) | mismatches=%d\n",
               k_used, (k < 0 ? " (dyn)" : ""), M,
               results.total_gpu_ms, gpu_mps, results.total_cpu_ms, cpu_mps,
               results.total_mism);
        printf("CRT Kernel-only: %.2f M/s\n", kernel_mps);
    }
}

#ifndef NO_CGBN
void run_cgbn_benchmark(int M, const cpp_int& N,
                       const std::vector<cpp_int>& divisors,
                       cgbn_bn_mem_t* d_N_single) {
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

    if (nbits > CGBN_BITS || max_div_bits > CGBN_BITS) {
        size_t needed = std::max(nbits, max_div_bits);
        fprintf(stderr, "[CGBN div_rem] Skipping: Need %zu bits, CGBN_BITS=%d. "
                "Rebuild with -DCGBN_BITS=%zu\n",
                needed, CGBN_BITS, ((needed + 31) / 32) * 32);
        return;
    }

    printf("[CGBN div_rem] Running comparison benchmark\n");

    cgbn_bn_mem_t *h_divs = nullptr, *h_r = nullptr;
    cudaHostAlloc(&h_divs, M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_r, M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);

    for (int i = 0; i < M; ++i) { cppint_to_cgbn_mem(divisors[i], h_divs[i]); }

    cgbn_bn_mem_t *d_divs = nullptr, *d_qm = nullptr, *d_rm = nullptr;
    cgbn_error_report_t *report = nullptr;

#ifdef CGBN_ALLOCATE_ERROR_REPORT
    report = CGBN_ALLOCATE_ERROR_REPORT();
#else
    CUDA_CHECK(cudaMallocManaged(&report, sizeof(cgbn_error_report_t)));
    host_cgbn_error_report_init(report);
#endif

    CUDA_CHECK(cudaMalloc(&d_divs, M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_qm, M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_rm, M * sizeof(cgbn_bn_mem_t)));

    auto t_h2d_start = now_tp();
    CUDA_CHECK(cudaMemcpy(d_divs, h_divs, M * sizeof(cgbn_bn_mem_t), cudaMemcpyHostToDevice));
    double cgbn_h2d_ms = ms_since(t_h2d_start);

     // Kernel timing with chunking
cudaEvent_t c0, c1;
CUDA_CHECK(cudaEventCreate(&c0));
CUDA_CHECK(cudaEventCreate(&c1));

auto t_launch_start = now_tp();
CUDA_CHECK(cudaEventRecord(c0));

    // CHUNKED PROCESSING: Process 500k divisions at a time
    const int CGBN_CHUNK_SIZE = 500000;

    for (int offset = 0; offset < M; offset += CGBN_CHUNK_SIZE) {
        int chunk = std::min(CGBN_CHUNK_SIZE, M - offset);
        
        int threads = 128;
        if (threads % CGBN_TPI) threads += (CGBN_TPI - (threads % CGBN_TPI));
        int instances_per_block = threads / CGBN_TPI;
        int blocks = (chunk + instances_per_block - 1) / instances_per_block;
        
        cgbn_divrem_kernel<<<blocks, threads>>>(
            report, 
            d_N_single, 
            d_divs + offset,    // offset into divisors array
            d_qm + offset,      // offset into quotients array
            d_rm + offset,      // offset into remainders array
            chunk               // number of divisions in this chunk
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(c1));
    CUDA_CHECK(cudaEventSynchronize(c1));
    CUDA_CHECK(cudaDeviceSynchronize());

    double cgbn_launch_ms = ms_since(t_launch_start);
    float cgbn_divrem_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&cgbn_divrem_ms, c0, c1));


    auto t_d2h_start = now_tp();
    CUDA_CHECK(cudaMemcpy(h_r, d_rm, M * sizeof(cgbn_bn_mem_t), cudaMemcpyDeviceToHost));
    double cgbn_d2h_ms = ms_since(t_d2h_start);

    int mism_cgbn = 0;
    for (int i = 0; i < M; ++i) {
        cpp_int cgbn_rem = cgbn_mem_to_cppint(h_r[i]);
        cpp_int expect = N % divisors[i];
        if (cgbn_rem != expect) {
            if (mism_cgbn < 5) {
                fprintf(stderr, "[CGBN div_rem] Mismatch at %d: cpu=%s cgbn=%s\n", i, expect.str().c_str(), cgbn_rem.str().c_str());
            }
            ++mism_cgbn;
        }
    }

    double cgbn_mps = (cgbn_divrem_ms > 0) ? (M / 1e6) / (cgbn_divrem_ms / 1000.0) : 0.0;
    printf("CGBN div_rem | %d divisions | %.3f ms (%.2f M/s) | mismatches=%d\n", M, cgbn_divrem_ms, cgbn_mps, mism_cgbn);

    double cgbn_full_ms = cgbn_h2d_ms + (double)cgbn_divrem_ms + cgbn_d2h_ms;
    double cgbn_full_mps = (cgbn_full_ms > 0.0) ? ((M / 1e6) / (cgbn_full_ms / 1000.0)) : 0.0;
    printf("CGBN div_rem full | %d divisions | %.3f ms (%.2f M/s) | h2d=%.2f ms kernel=%.2f ms d2h=%.2f ms\n", M, cgbn_full_ms, cgbn_full_mps, cgbn_h2d_ms, (double)cgbn_divrem_ms, cgbn_d2h_ms);

    CUDA_CHECK(cudaFree(d_divs));
    CUDA_CHECK(cudaFree(d_qm));
    CUDA_CHECK(cudaFree(d_rm));
#ifdef CGBN_FREE_ERROR_REPORT
    CGBN_FREE_ERROR_REPORT(report);
#else
    CUDA_CHECK(cudaFree(report));
#endif
    CUDA_CHECK(cudaEventDestroy(c0));
    CUDA_CHECK(cudaEventDestroy(c1));
    cudaFreeHost(h_divs);
    cudaFreeHost(h_r);
}
#endif

int main(int argc, char** argv) {
    check_gpu_capability();
    printf("=== CRT GPU with %d-bit divisors ===\n", DIVISOR_BITS);

    warmup_gpu_context();

    const u64 BASE_SEED_64 = 1234567ull;
    const u32 BASE_SEED_32 = (u32)BASE_SEED_64;

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string Ns;
    std::ifstream fin(argv[1]);
    if (fin) { std::getline(fin, Ns); fin.close(); } else { Ns = argv[1]; }

    int k = -1;
    int argi = 2;
    if (argc >= 5 && std::string(argv[2]) == "-k") {
        char* endptr = nullptr;
        long kval = std::strtol(argv[3], &endptr, 10);
        if (*endptr != '\0' || kval <= 0) { fprintf(stderr, "Error: invalid K for -k (got '%s').\n", argv[3]); return 1; }
        k = (int)kval;
        argi = 4;
    }

    if (argc - argi < 1) { fprintf(stderr, "Error: At least one M value required.\n"); return 1; }

    std::vector<int> M_list;
    for (int i = argi; i < argc; ++i) { M_list.push_back(std::stoi(argv[i])); }

    cpp_int N = read_big_decimal(Ns);
    const int SAFETY_BITS = 128;

    // Determine required k for table selection
    int required_k;
    if (k > 0) {
        required_k = k;
    } else {
        // Estimate k needed dynamically
        size_t nbits = bitlen_cppint(N);
        double target_bits = (double)nbits + (double)SAFETY_BITS;
        required_k = (int)(target_bits / 20.0) + 100;
        if (required_k < 100) required_k = 100;
    }

    // Select appropriate table file based on required_k
    std::string table_file;
    if (required_k <= 1000) {
        table_file = "data/garner_k1k.bin";
    } else if (required_k <= 4000) {
        table_file = "data/garner_k4k.bin";
    } else if (required_k <= 10000) {
        table_file = "data/garner_k10k.bin";
    } else if (required_k <= 20000) {
        table_file = "data/garner_k20k.bin";
    } else {
        table_file = "data/garner_k40k.bin";
        if (required_k > 40000) {
            fprintf(stderr, "Warning: Need k=%d but max table is k=40000\n", required_k);
            required_k = 40000;
        }
    }

    printf("[Setup] Loading Garner table: %s (k=%d)\n", table_file.c_str(), required_k);
    GarnerTable G = load_garner_table(table_file, required_k);

    SetupData setup = perform_crt_setup(N, k, SAFETY_BITS, G);

    DeviceCRTData crt_data;
    crt_data.allocate_and_upload(setup.c32, setup.m, setup.k_used);

    double total_setup_ms = setup.choose_ms + setup.residues_ms +
                           setup.garner_ms + crt_data.h2d_ms;

    printf("\n=== One-time Setup ===\n");
    printf("choose=%.2fms residues=%.2fms garner=%.2fms h2d_static=%.2fms | total=%.2fms\n",
           setup.choose_ms, setup.residues_ms, setup.garner_ms,
           crt_data.h2d_ms, total_setup_ms);
    printf("\n");

    if (DIVISOR_BITS == 32) {
        warmup_crt_kernel_32(crt_data.d_c, crt_data.d_m, setup.k_used);
    } else if (DIVISOR_BITS == 64) {
        warmup_crt_kernel_64(crt_data.d_c, crt_data.d_m, setup.k_used);
    }

#ifndef NO_CGBN
    cgbn_bn_mem_t N_mem;
    cppint_to_cgbn_mem(N, N_mem);
    cgbn_bn_mem_t *d_N_single = nullptr;
    CUDA_CHECK(cudaMalloc(&d_N_single, sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N_single, &N_mem, sizeof(cgbn_bn_mem_t), cudaMemcpyHostToDevice));
#endif

    double total_runtime_ms = 0.0;
    long long total_divs = 0;

    for (int M : M_list) {
        BenchmarkResults results;

        auto t_pgen_start = now_tp();
        std::unordered_set<u32> mset(setup.m.begin(), setup.m.end());
        std::vector<cpp_int> divisors = generate_divisors(M, DIVISOR_BITS, mset);
        results.pgen_ms = ms_since(t_pgen_start);

        if (DIVISOR_BITS == 32) {
            run_crt_benchmark_32(M, N, divisors, crt_data, setup.k_used, results, BASE_SEED_64, BASE_SEED_32);
        } else if (DIVISOR_BITS == 64) {
            run_crt_benchmark_64(M, N, divisors, crt_data, setup.k_used, results, BASE_SEED_64, BASE_SEED_32);
        } else {
#ifndef NO_CGBN
            fprintf(stderr, "Error: CGBN-CRT kernel is not available in this build.\n"); continue;
#else
            fprintf(stderr, "Error: DIVISOR_BITS=%d requires CGBN, but NO_CGBN is defined\n", DIVISOR_BITS); continue;
#endif
        }

        print_benchmark_results(M, results, setup.k_used, k);

        total_divs += M;
        total_runtime_ms += results.h2d_chunks_ms + results.kernel_ms + results.d2h_chunks_ms;

        double total_elapsed_ms = total_setup_ms + total_runtime_ms;
        double crt_full_mps = (total_elapsed_ms > 0.0)
            ? (total_divs / 1e6) / (total_elapsed_ms / 1000.0)
            : 0.0;

        total_runtime_ms += perM_runtime_ms;
        total_divs += M;

#ifndef NO_CGBN
        run_cgbn_benchmark(M, N, divisors, d_N_single);
#endif
    }

    double crt_full_ms_total = total_setup_ms + total_runtime_ms;
    double crt_full_Mps_amortized = (total_divs > 0 && crt_full_ms_total > 0) ? (total_divs / 1e6) / (crt_full_ms_total / 1000.0) : 0.0;

    printf("\n=== Amortized CRT Full Throughput Across All M ===\n");
    printf("Total setup: %.2f ms | Total runtime: %.2f ms | Total divisions: %lld\n",
           total_setup_ms, total_runtime_ms, total_divs);
    printf("Amortized CRT-Full Throughput: %.2f M/s\n", crt_full_Mps_amortized);

    crt_data.free();
#ifndef NO_CGBN
    if (d_N_single) CUDA_CHECK(cudaFree(d_N_single));
#endif

    return 0;
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

