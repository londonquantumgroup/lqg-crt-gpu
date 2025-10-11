#pragma once
#include "crt_cgbn.hpp"

#ifdef ENABLE_CGBN

// CGBN div_rem kernel
__global__ void cgbn_divrem_kernel(
    cgbn_error_report_t *report,
    const cgbn_bn_mem_t *d_N_single,
    cgbn_bn_mem_t *divs,
    cgbn_bn_mem_t *qouts,
    cgbn_bn_mem_t *routs,
    int count);

#endif // ENABLE_CGBN