#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(stmt) do {                                     \
    cudaError_t err = (stmt);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr,"CUDA error %s at %s:%d -> %s\n",          \
                #stmt, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                  \
    }                                                             \
} while(0)
