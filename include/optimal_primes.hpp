#pragma once
#include "types.hpp"
#include <vector>
#include <cmath>

// Optimal 32-bit primes for CRT
// For extreme post-quantum security and large-scale tests, we need up to 50,000 primes
// Strategy: Use the LARGEST primes below 2^32 working backwards

namespace OptimalPrimes {

// Maximum number of 32-bit primes we cache
// 50k primes × 32 bits ≈ 1.6 million bits of representation
constexpr int MAX_CACHED_PRIMES = 50000;

// Top 50,000 primes below 2^32 (largest first)
// This gives us up to ~1.6 million bits of representation
// Stored in a separate data file to avoid bloating this header
extern const u32 TOP_PRIMES_32BIT[MAX_CACHED_PRIMES];

// How many bits of entropy does k primes give us?
inline double bits_from_primes(int k) {
    if (k > MAX_CACHED_PRIMES) k = MAX_CACHED_PRIMES;
    double bits = 0.0;
    for (int i = 0; i < k; ++i) {
        bits += std::log2((double)TOP_PRIMES_32BIT[i]);
    }
    return bits;
}

// Get k optimal primes (the k largest primes below 2^32)
inline std::vector<u32> get_optimal_primes(int k) {
    if (k > MAX_CACHED_PRIMES) {
        fprintf(stderr, "Warning: requested k=%d, only have %d cached primes\n", 
                k, MAX_CACHED_PRIMES);
        k = MAX_CACHED_PRIMES;
    }
    return std::vector<u32>(TOP_PRIMES_32BIT, TOP_PRIMES_32BIT + k);
}

// Get enough primes to represent target_bits with safety margin
inline std::vector<u32> get_primes_for_bits(size_t target_bits, int safety_bits = 32, int* out_k = nullptr) {
    double needed_bits = (double)target_bits + (double)safety_bits;
    
    // Each 32-bit prime gives ~32 bits, but account for non-uniformity
    int k = (int)((needed_bits / 31.9) + 2); // Slightly conservative
    
    // Refine by actually computing the product
    double acc_bits = 0.0;
    for (int i = 0; i < MAX_CACHED_PRIMES && i < k + 100; ++i) {
        acc_bits += std::log2((double)TOP_PRIMES_32BIT[i]);
        if (acc_bits >= needed_bits) {
            k = i + 1;
            break;
        }
    }
    
    if (k > MAX_CACHED_PRIMES) {
        fprintf(stderr, "Warning: need k=%d but only have %d primes\n", k, MAX_CACHED_PRIMES);
        k = MAX_CACHED_PRIMES;
    }
    
    if (out_k) *out_k = k;
    return get_optimal_primes(k);
}

// Check if we have enough cached primes
inline bool has_sufficient_primes(size_t target_bits, int safety_bits = 32) {
    double needed_bits = (double)target_bits + (double)safety_bits;
    double available_bits = bits_from_primes(MAX_CACHED_PRIMES);
    return available_bits >= needed_bits;
}

// For post-quantum: how many primes do we need?
inline int primes_needed_for_security_level(int security_bits) {
    // Classical security: need ~2*security_bits of CRT product
    // Post-quantum: need ~3*security_bits (conservative estimate)
    int target_bits = 3 * security_bits;
    int k = 0;
    get_primes_for_bits(target_bits, 64, &k);
    return k;
}

// Statistics about our prime cache
inline void print_cache_info() {
    printf("Prime cache info:\n");
    printf("  Total primes cached: %d\n", MAX_CACHED_PRIMES);
    printf("  Largest prime: %u\n", TOP_PRIMES_32BIT[0]);
    printf("  Smallest cached: %u\n", TOP_PRIMES_32BIT[MAX_CACHED_PRIMES - 1]);
    double total_bits = bits_from_primes(MAX_CACHED_PRIMES);
    printf("  Total representation: ~%.0f bits (~%d bytes)\n", total_bits, (int)(total_bits / 8));
    printf("\nSecurity levels supported:\n");
    for (int sec : {128, 192, 256, 384, 512}) {
        int k = primes_needed_for_security_level(sec);
        printf("  %d-bit post-quantum: %d primes\n", sec, k);
    }
}

} // namespace OptimalPrimes