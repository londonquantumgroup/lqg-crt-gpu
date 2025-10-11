#pragma once
#include "types.hpp"

#ifndef NO_CGBN
#include <gmp.h>
#include <gmpxx.h>
#define CGBN_USE_GMP 1

// --- Fix: prevent CGBN error-report symbols from being defined in multiple .cu files ---
#ifndef CGBN_NO_IMPLEMENTATION
#define CGBN_NO_IMPLEMENTATION
#endif

#include <cgbn/cgbn.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstring>

using boost::multiprecision::cpp_int;

typedef cgbn_context_t<CGBN_TPI>               cgbn_context;
typedef cgbn_env_t<cgbn_context, CGBN_BITS>    cgbn_env;
typedef cgbn_env::cgbn_t                       cgbn_bn_t;
typedef cgbn_mem_t<CGBN_BITS>                  cgbn_bn_mem_t;

// Use __host__ __device__ to allow use in both CPU and GPU code
__host__ __device__ static inline void cppint_to_cgbn_mem_impl(const cpp_int &x, cgbn_bn_mem_t &out) {
    for (size_t i = 0; i < sizeof(out._limbs) / sizeof(out._limbs[0]); ++i) {
        out._limbs[i] = 0u;
    }
    cpp_int t = x;
    size_t i = 0;
    while (t > 0 && i < sizeof(out._limbs) / sizeof(out._limbs[0])) {
        uint32_t limb = (uint32_t)(t & cpp_int(0xFFFFFFFFu));
        out._limbs[i] = limb;
        t >>= 32;
        ++i;
    }
}

// Wrapper to avoid multiple definitions
static inline void cppint_to_cgbn_mem(const cpp_int &x, cgbn_bn_mem_t &out) {
    cppint_to_cgbn_mem_impl(x, out);
}

__host__ __device__ static inline cpp_int cgbn_mem_to_cppint_impl(const cgbn_bn_mem_t &x) {
    cpp_int result = 0;
    for (int i = (sizeof(x._limbs) / sizeof(x._limbs[0])) - 1; i >= 0; --i) {
        result = (result << 32) | cpp_int(x._limbs[i]);
    }
    return result;
}

static inline cpp_int cgbn_mem_to_cppint(const cgbn_bn_mem_t &x) {
    return cgbn_mem_to_cppint_impl(x);
}

static inline void host_cgbn_error_report_init(cgbn_error_report_t *report) {
    if (report) {
        memset(report, 0, sizeof(cgbn_error_report_t));
    }
}
#endif