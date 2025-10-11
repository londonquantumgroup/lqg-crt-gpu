#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>
#include "crt_config.hpp"

using boost::multiprecision::cpp_int;

inline std::chrono::high_resolution_clock::time_point now_tp() {
  return std::chrono::high_resolution_clock::now();
}
inline double ms_since(std::chrono::high_resolution_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - t0)
      .count();
}

inline size_t bitlen_cppint(const cpp_int &x) {
  if (x == 0) return 1;
  cpp_int t = x;
  size_t bits = 0;
  while (t > 0) {
    t >>= 1;
    ++bits;
  }
  return bits;
}

inline cpp_int read_big_decimal(const std::string &s) {
  cpp_int N = 0;
  for (char ch : s)
    if (ch >= '0' && ch <= '9') N = N * 10 + (ch - '0');
  return N;
}

// CUDA check
#include <cuda_runtime.h>
#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t err = (stmt);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #stmt, __FILE__,        \
              __LINE__, cudaGetErrorString(err));                               \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)