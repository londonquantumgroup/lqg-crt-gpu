// src/cgbn_impl.cu
// This is the ONLY file that should have the full CGBN implementation
// All other .cu files should have CGBN_NO_IMPLEMENTATION defined

#include <gmp.h>
#include <gmpxx.h>
#define CGBN_USE_GMP 1

// DO NOT define CGBN_NO_IMPLEMENTATION here - this file needs the implementation
#include <cgbn/cgbn.h>
