#pragma once

// ===== Project-wide switches (same as your monolith) =====
using u32  = unsigned int;
using u64  = unsigned long long;
using u128 = unsigned __int128;

#ifndef MAX_K
#define MAX_K 128
#endif

#ifndef FAST_REMAINDER_TREE
#define FAST_REMAINDER_TREE 1
#endif

// Default to 128 (you can override with -DDIVISOR_BITS=32 or 64 or 128+)
#ifndef DIVISOR_BITS
#define DIVISOR_BITS 128
#endif

// CRT safety margin in bits
#ifndef SAFETY_BITS
#define SAFETY_BITS 32
#endif

// GPU divisor generation toggle
#ifndef GPU_GENERATE_DIVISORS
#define GPU_GENERATE_DIVISORS 1
#endif

// CGBN knobs (guarded in code; ENABLE_CGBN set by CMake option)
#ifndef CGBN_TPI
#define CGBN_TPI 32
#endif
#ifndef CGBN_BITS
#define CGBN_BITS 256
#endif