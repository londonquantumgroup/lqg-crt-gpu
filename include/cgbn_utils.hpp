#pragma once
#include "types.hpp"

#ifndef NO_CGBN

// DON'T include cgbn.h here! Each .cu file must include it with proper defines.
// This header only provides utility functions that work with CGBN types.

#include <boost/multiprecision/cpp_int.hpp>
#include <cstring>
#include <gmp.h>
#include <gmpxx.h>

// Forward declare the cgbn_mem_t template
template<uint32_t BITS> struct cgbn_mem_t;

// Forward declare error report type
struct cgbn_error_report_t;

using boost::multiprecision::cpp_int;

// Utility functions that work with cgbn_mem_t
// Note: cgbn_mem_t<CGBN_BITS> must be fully defined before calling these

template<uint32_t BITS>
static inline void cppint_to_cgbn_mem(const cpp_int &x, cgbn_mem_t<BITS> &out) {
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

template<uint32_t BITS>
static inline cpp_int cgbn_mem_to_cppint(const cgbn_mem_t<BITS> &x) {
    cpp_int result = 0;
    for (int i = (sizeof(x._limbs) / sizeof(x._limbs[0])) - 1; i >= 0; --i) {
        result = (result << 32) | cpp_int(x._limbs[i]);
    }
    return result;
}

static inline void host_cgbn_error_report_init(cgbn_error_report_t *report) {
    if (report) {
        memset(report, 0, sizeof(cgbn_error_report_t));
    }
}

#endif

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