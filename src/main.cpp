#include "../include/crt_config.hpp"
#include "../include/crt_utils.hpp"
#include "../include/crt_math.hpp"
#include "../include/crt_garner.hpp"
#include "../include/crt_product_tree.hpp"
#include "../include/crt_gpu_kernels.cuh"
#include "../include/crt_host_helpers.hpp"
#include "../include/crt_cpu.hpp"
#include "../include/cgbn_benchmark.hpp"

#include <unordered_set>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

/*** forward declarations needed by this TU ***/
#include <boost/multiprecision/cpp_int.hpp>

std::vector<uint32_t> choose_moduli_dynamic(const std::vector<uint32_t>& primes,
                                            const boost::multiprecision::cpp_int& N,
                                            int bit_limit,
                                            int* k_dyn);



int main(int argc, char** argv) {
  // Basic device sanity
  int device=0; cudaGetDevice(&device);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
  if (prop.major < 7)
    std::cerr << "Warning: device architecture may not support fast 128-bit math\n";
  printf("=== CRT GPU with %d-bit divisors ===\n", DIVISOR_BITS);

  // Warm-up
  auto t_warm_start = now_tp();
  cudaFree(0);
  cudaDeviceSynchronize();
#if FAST_REMAINDER_TREE
  {
    dim3 warm_blocks(1), warm_threads(1);
    remainders_via_crt_32<<<warm_blocks, warm_threads>>>(
      (u32*)nullptr, (u32*)nullptr, 0, (u32*)nullptr, (u32*)nullptr, 0);
    cudaDeviceSynchronize();
  }
#endif
  printf("[warmup] GPU ready in %.2f ms\n\n", ms_since(t_warm_start));

  const u64 BASE_SEED_64 = 1234567ull;
  const u32 BASE_SEED_32 = (u32)BASE_SEED_64;

  if (argc < 3) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s N M1 [M2 M3 ...]\n", argv[0]);
    fprintf(stderr, "  %s N -k K M1 [M2 M3 ...]   (explicitly set number of CRT moduli)\n", argv[0]);
    fprintf(stderr, "\nDivisor size: DIVISOR_BITS=%d (compile with -DDIVISOR_BITS=X)\n", DIVISOR_BITS);
    fprintf(stderr, "CGBN_BITS=%d (compile with -DCGBN_BITS=X for larger numbers)\n", CGBN_BITS);
    return 1;
  }

  // Read N (file path or literal)
  std::string Ns;
  {
    std::ifstream fin(argv[1]);
    if (fin) { std::getline(fin, Ns); fin.close(); }
    else     { Ns = argv[1]; }
  }

  int k = -1;
  int argi = 2;

  if (argc >= 5 && std::string(argv[2]) == "-k") {
    char* endptr = nullptr;
    long kval = std::strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || kval <= 0) {
      fprintf(stderr, "Error: invalid K for -k (got '%s').\n", argv[3]);
      return 1;
    }
    k = (int)kval;
    argi = 4;
  }
  if (argc - argi < 1) {
    fprintf(stderr, "Error: At least one M value required.\n");
    return 1;
  }

  std::vector<int> M_list;
  for (int i = argi; i < argc; ++i) {
    int Mi = std::stoi(argv[i]);
    if (Mi <= 0) {
      fprintf(stderr, "Error: M values must be positive (got %d).\n", Mi);
      return 1;
    }
    M_list.push_back(Mi);
  }

  double choose_ms=0.0, residues_ms=0.0, garner_ms=0.0, h2d_static_ms=0.0;
  cpp_int N = read_big_decimal(Ns);

  auto t_choose_start = now_tp();
  std::vector<u32> m;
  int k_used = k;
  if (k < 0) {
    int k_dyn = 0;
    m = choose_moduli_dynamic(global_primes(), N, SAFETY_BITS, &k_dyn);
    k_used = k_dyn;
  } else {
    const auto& gp = global_primes();
    if (k > (int)gp.size()) k_used = (int)gp.size(); else k_used = k;
    m.assign(gp.begin(), gp.begin() + k_used);
  }
  choose_ms = ms_since(t_choose_start);

  auto t_residues_start = now_tp();
  std::vector<u32> r(k_used);
#if FAST_REMAINDER_TREE
  {
    CRTProductTree PT = build_product_tree(m);
    std::vector<cpp_int> leaf_rems;
    remainder_tree_down(PT, N, leaf_rems);
    for (int i=0;i<k_used;++i) r[i] = (u32)leaf_rems[i];
  }
#else
  for (int i=0;i<k_used;++i) r[i] = (u32)(N % cpp_int(m[i]));
#endif
  residues_ms = ms_since(t_residues_start);

  auto t_garner_start = now_tp();
  std::vector<u64> c64 = garner_from_residues(
    std::vector<u64>(r.begin(), r.end()),
    std::vector<u64>(m.begin(), m.end())
  );
  std::vector<u32> c32(c64.begin(), c64.end());
  garner_ms = ms_since(t_garner_start);

  // Upload CRT coeffs/moduli once
  auto t_h2d_static_start = now_tp();
  u32 *d_c=nullptr, *d_m=nullptr;
  CUDA_CHECK(cudaMalloc(&d_c, k_used * sizeof(u32)));
  CUDA_CHECK(cudaMalloc(&d_m, k_used * sizeof(u32)));
  CUDA_CHECK(cudaMemcpy(d_c, c32.data(), k_used*sizeof(u32), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m, m.data(),   k_used*sizeof(u32), cudaMemcpyHostToDevice));
  h2d_static_ms = ms_since(t_h2d_static_start);

  printf("\n=== One-time Setup ===\n");
  double total_setup_ms = choose_ms + residues_ms + garner_ms + h2d_static_ms;
  printf("choose=%.2fms residues=%.2fms garner=%.2fms h2d_static=%.2fms | total=%.2fms\n",
         choose_ms, residues_ms, garner_ms, h2d_static_ms, total_setup_ms);

  const int CHUNK_SIZE = 10000000;
  std::unordered_set<u32> mset(m.begin(), m.end());

  double total_runtime_ms = 0.0;
  long long total_divs = 0;

  for (size_t m_idx=0; m_idx<M_list.size(); ++m_idx) {
    int M = M_list[m_idx];
    double pgen_ms=0.0, h2d_chunks_ms=0.0, kernel_ms=0.0, d2h_chunks_ms=0.0;
    double total_gpu_ms = 0.0, total_cpu_ms = 0.0;
    int total_mism = 0;

    auto t_pgen_start = now_tp();
    std::vector<cpp_int> divisors = generate_divisors(M, DIVISOR_BITS, mset);
    if ((int)divisors.size() < M) {
      fprintf(stderr, "Warning: Only generated %zu divisors, clamping M to %zu\n",
              divisors.size(), divisors.size());
      M = (int)divisors.size();
    }
    pgen_ms = ms_since(t_pgen_start);

    void *d_P_void=nullptr, *d_out_void=nullptr;
    if (DIVISOR_BITS == 32) {
      CUDA_CHECK(cudaMalloc(&d_P_void,  M*sizeof(u32)));
      CUDA_CHECK(cudaMalloc(&d_out_void,M*sizeof(u32)));
    } else if (DIVISOR_BITS == 64) {
      CUDA_CHECK(cudaMalloc(&d_P_void,  M*sizeof(u64)));
      CUDA_CHECK(cudaMalloc(&d_out_void,M*sizeof(u64)));
    }

    if (DIVISOR_BITS <= 64 && GPU_GENERATE_DIVISORS) {
      int threads=256, blocks=(M+threads-1)/threads;
      if (DIVISOR_BITS == 32) {
        generate_divisors_kernel_32<<<blocks,threads>>>(BASE_SEED_32, M, (u32*)d_P_void);
      } else {
        generate_divisors_kernel_64<<<blocks,threads>>>(BASE_SEED_64, M, (u64*)d_P_void);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      pgen_ms = ms_since(t_pgen_start);
    } else if (DIVISOR_BITS <= 64) {
      if (DIVISOR_BITS == 32) {
        std::vector<u32> P_cpu(M);
        for (int i=0;i<M;++i) P_cpu[i] = (u32)divisors[i];
        CUDA_CHECK(cudaMemcpy(d_P_void, P_cpu.data(), M*sizeof(u32), cudaMemcpyHostToDevice));
      } else {
        std::vector<u64> P_cpu(M);
        for (int i=0;i<M;++i) P_cpu[i] = (u64)divisors[i];
        CUDA_CHECK(cudaMemcpy(d_P_void, P_cpu.data(), M*sizeof(u64), cudaMemcpyHostToDevice));
      }
    }

    if (DIVISOR_BITS == 32) {
      std::vector<u32> gpu_chunk, cpu_chunk;
      for (int offset=0; offset<M; offset+=CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        u32 *d_P32   = reinterpret_cast<u32*>(d_P_void)   + offset;
        u32 *d_out32 = reinterpret_cast<u32*>(d_out_void) + offset;

        // kernel timing
        cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        int threads=256, blocks=(chunk+threads-1)/threads;
        remainders_via_crt_32<<<blocks,threads>>>(d_P32, d_out32, chunk, d_c, d_m, k_used);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float gpu_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
        kernel_ms += gpu_ms; total_gpu_ms += gpu_ms;
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));

        // D2H
        auto t_d2h_chunk_start = now_tp();
        gpu_chunk.resize(chunk);
        CUDA_CHECK(cudaMemcpy(gpu_chunk.data(), d_out32, chunk*sizeof(u32), cudaMemcpyDeviceToHost));
        d2h_chunks_ms += ms_since(t_d2h_chunk_start);

        // CPU check
        auto tstart = now_tp();
        cpu_chunk.resize(chunk);
        if (GPU_GENERATE_DIVISORS) {
          for (int i=0;i<chunk;++i) {
            u32 p = divisor_at_32(BASE_SEED_32, offset + i);
            cpu_chunk[i] = (u32)( (u64)(N % cpp_int(p)) );
          }
        } else {
          for (int i=0;i<chunk;++i) cpu_chunk[i] = (u32)(N % divisors[offset+i]);
        }
        total_cpu_ms += ms_since(tstart);

        for (int i=0;i<chunk;++i) if (cpu_chunk[i] != gpu_chunk[i]) {
          if (total_mism < 5)
            fprintf(stderr, "Mismatch at %d: cpu=%u gpu=%u\n",
                    offset+i, cpu_chunk[i], gpu_chunk[i]);
          ++total_mism;
        }
      }
      CUDA_CHECK(cudaFree(d_P_void)); CUDA_CHECK(cudaFree(d_out_void));
      
    } else if (DIVISOR_BITS == 64) {
      std::vector<u64> gpu_chunk, cpu_chunk;
      for (int offset=0; offset<M; offset+=CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        u64 *d_P64   = reinterpret_cast<u64*>(d_P_void)   + offset;
        u64 *d_out64 = reinterpret_cast<u64*>(d_out_void) + offset;

        cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        int threads=256, blocks=(chunk+threads-1)/threads;
        remainders_via_crt_64<<<blocks,threads>>>(d_P64, d_out64, chunk, d_c, d_m, k_used);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float gpu_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
        kernel_ms += gpu_ms; total_gpu_ms += gpu_ms;
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));

        auto t_d2h_chunk_start = now_tp();
        gpu_chunk.resize(chunk);
        CUDA_CHECK(cudaMemcpy(gpu_chunk.data(), d_out64, chunk*sizeof(u64), cudaMemcpyDeviceToHost));
        d2h_chunks_ms += ms_since(t_d2h_chunk_start);

        auto tstart = now_tp();
        cpu_chunk.resize(chunk);
        if (GPU_GENERATE_DIVISORS) {
          for (int i=0;i<chunk;++i) {
            u64 p = divisor_at_64(BASE_SEED_64, offset + i);
            cpu_chunk[i] = (u64)( (u64)(N % cpp_int(p)) );
          }
        } else {
          for (int i=0;i<chunk;++i) cpu_chunk[i] = (u64)(N % divisors[offset+i]);
        }
        total_cpu_ms += ms_since(tstart);

        for (int i=0;i<chunk;++i) if (cpu_chunk[i] != gpu_chunk[i]) {
          if (total_mism < 5)
            fprintf(stderr, "Mismatch at %d: cpu=%llu gpu=%llu\n",
                    offset+i, (unsigned long long)cpu_chunk[i], (unsigned long long)gpu_chunk[i]);
          ++total_mism;
        }
      }
      CUDA_CHECK(cudaFree(d_P_void)); CUDA_CHECK(cudaFree(d_out_void));
      
    } else {
      // 128+ bit divisors: compute CPU baseline only (no CRT-GPU kernel)
      auto tstart = now_tp();
      for (int i = 0; i < M; ++i) {
        volatile cpp_int rem = N % divisors[i];  // prevent optimization
        (void)rem;
      }
      total_cpu_ms = ms_since(tstart);
    }

    // Calculate metrics (works for all divisor sizes)
    double gpu_mps = (total_gpu_ms>0) ? (M/1e6) / (total_gpu_ms/1000.0) : 0.0;
    double cpu_mps = (total_cpu_ms>0) ? (M/1e6) / (total_cpu_ms/1000.0) : 0.0;

    double ms_per_cand_gpu = (total_gpu_ms>0) ? total_gpu_ms / double(M) : std::numeric_limits<double>::quiet_NaN();
    double ms_per_cand_cpu = (total_cpu_ms>0) ? total_cpu_ms / double(M) : std::numeric_limits<double>::quiet_NaN();
    double Mcross = (ms_per_cand_gpu>0) ? (ms_per_cand_cpu / ms_per_cand_gpu) : -1.0;

    double perM_runtime_ms = h2d_chunks_ms + kernel_ms + d2h_chunks_ms;
    double amortized_setup_ms = (total_setup_ms * (double(M) / double(total_divs + M)));
    double crt_full_ms = amortized_setup_ms + perM_runtime_ms;
    double crt_full_mps = (M / 1e6) / (crt_full_ms / 1000.0);

    total_runtime_ms += perM_runtime_ms;
    total_divs += M;

    // Print CRT results
    printf("\n=== Results for M=%d (%d-bit divisors) ===\n", M, DIVISOR_BITS);
    printf("M=%d: ms per candidate: GPU=%.9f, CPU=%.9f | CPU/GPU crossover @ M=%.9f\n",
           M, ms_per_cand_gpu, ms_per_cand_cpu, Mcross);
    printf("[profile] runtime: h2d_chunks=%.2fms kernel=%.2fms d2h_chunks=%.2fms | external: pgen=%.2fms\n",
           h2d_chunks_ms, kernel_ms, d2h_chunks_ms, pgen_ms);

    if (DIVISOR_BITS <= 64) {
      printf("k=%d%s candidates=%d | CRT-GPU %.3f ms (%.2f M/s) | CPU %.3f ms (%.2f M/s) | mismatches=%d\n",
             k_used, (k < 0 ? " (dyn)":""), M,
             total_gpu_ms, gpu_mps, total_cpu_ms, cpu_mps, total_mism);
      printf("CRT Kernel-only: %.2f M/s\n", gpu_mps);
      printf("CRT Full (amortized): %.2f M/s\n", crt_full_mps);
    } else {
      printf("k=%d%s candidates=%d | Big-int divisors | CPU %.3f ms (%.2f M/s)\n",
             k_used, (k < 0 ? " (dyn)":""), M, total_cpu_ms, cpu_mps);
    }

    // ✅ ALWAYS run CGBN benchmark for comparison (at ALL divisor sizes)
    run_cgbn_benchmark(N, divisors, M, ms_per_cand_cpu);
    
  } // end M_list loop

  // Final summary
  double crt_full_ms_total = total_setup_ms + total_runtime_ms;
  double crt_full_Mps_amortized = (total_divs / 1e6) / (crt_full_ms_total / 1000.0);
  printf("\n=== Amortized CRT Full Throughput Across All M ===\n");
  printf("Total setup: %.2f ms | Total runtime: %.2f ms | Total divisions: %lld\n",
         total_setup_ms, total_runtime_ms, total_divs);
  printf("Amortized CRT-Full Throughput: %.2f M/s\n", crt_full_Mps_amortized);

  // Cleanup
  if (d_c) CUDA_CHECK(cudaFree(d_c));
  if (d_m) CUDA_CHECK(cudaFree(d_m));

  return 0;
}