#pragma once
#include "crt_config.hpp"
#include "crt_utils.hpp"

#ifdef ENABLE_CGBN
#include <gmp.h>
#include <gmpxx.h>
#define CGBN_USE_GMP 1
#include <cgbn/cgbn.h>

typedef cgbn_context_t<CGBN_TPI>               cgbn_context;
typedef cgbn_env_t<cgbn_context, CGBN_BITS>    cgbn_env;
typedef cgbn_env::cgbn_t                       cgbn_bn_t;
typedef cgbn_mem_t<CGBN_BITS>                  cgbn_bn_mem_t;

// Convert cpp_int to CGBN memory format
inline void cppint_to_cgbn_mem(const cpp_int &x, cgbn_bn_mem_t &out) {
    for(size_t i=0; i<sizeof(out._limbs)/sizeof(out._limbs[0]); ++i) 
        out._limbs[i] = 0u;
    cpp_int t = x;
    size_t i = 0;
    while(t > 0 && i < sizeof(out._limbs)/sizeof(out._limbs[0])) {
        uint32_t limb = (uint32_t)(t & cpp_int(0xFFFFFFFFu));
        out._limbs[i] = limb;
        t >>= 32;
        ++i;
    }
}

// Convert CGBN memory format to cpp_int
inline cpp_int cgbn_mem_to_cppint(const cgbn_bn_mem_t &x) {
    cpp_int result = 0;
    for(int i = (sizeof(x._limbs)/sizeof(x._limbs[0])) - 1; i >= 0; --i) {
        result = (result << 32) | cpp_int(x._limbs[i]);
    }
    return result;
}

// Initialize CGBN error report
inline void host_cgbn_error_report_init(cgbn_error_report_t *report) {
    if (report) {
        memset(report, 0, sizeof(cgbn_error_report_t));
    }
}

#endif // ENABLE_CGBN