#pragma once
#include "types.hpp"
#include "bigint_utils.hpp"
#include <vector>
#include <string>

// Forward declarations
struct CRTProductTree {
    std::vector<std::vector<cpp_int>> levels;
};

// Garner precomputed table structure
struct GarnerTable {
    uint32_t max_k;
    std::vector<uint32_t> primes;
    std::vector<uint64_t> inv_flat;
};

// File header structure (matches generator)
struct GarnerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t max_k;
    uint32_t num_primes;
    uint64_t primes_offset;
    uint64_t inv_table_offset;
    uint64_t file_size;
};

// Function declarations
CRTProductTree build_product_tree(const std::vector<u32>& m);

void remainder_tree_down(const CRTProductTree &T, const cpp_int &N, 
                        std::vector<cpp_int> &leaf_rems);

std::vector<u32> choose_moduli_dynamic(const std::vector<u32>& primes, 
                                       const cpp_int& N, int safety_bits, 
                                       int* out_k = nullptr);

// Load Garner table from file
GarnerTable load_garner_table(const std::string& filename, int required_k);

// Garner algorithm with precomputed inverses
std::vector<u64> garner_from_residues(const std::vector<u64>& r, 
                                      const std::vector<u64>& m,
                                      const GarnerTable& G);