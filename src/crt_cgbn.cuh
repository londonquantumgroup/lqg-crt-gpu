#include "../include/crt_cgbn_kernels.cuh"

#ifdef ENABLE_CGBN

__global__ void cgbn_divrem_kernel(
    cgbn_error_report_t *report,
    const cgbn_bn_mem_t *d_N_single,
    cgbn_bn_mem_t *divs,
    cgbn_bn_mem_t *qouts,
    cgbn_bn_mem_t *routs,
    int count)
{
    cgbn_context ctx(cgbn_no_checks, report, 0);
    cgbn_env env(ctx);
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / CGBN_TPI;
    if(instance >= count) return;
    
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

#endif // ENABLE_CGBN