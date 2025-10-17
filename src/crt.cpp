#include "crt.hpp"
#include "modular_arithmetic.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>

GarnerTable load_garner_table(const std::string& filename, int required_k) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + filename);

    GarnerHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));

    if (h.magic != 0x47415442)
        throw std::runtime_error("Invalid magic number in Garner file");
    if (required_k > (int)h.max_k)
        throw std::runtime_error("Requested k larger than table size");

    GarnerTable G;
    G.max_k = h.max_k;

    // Read primes
    f.seekg(h.primes_offset);
    G.primes.resize(required_k);
    f.read(reinterpret_cast<char*>(G.primes.data()), required_k * sizeof(uint32_t));

    std::cout << "[Garner] First 10 primes from table: ";
    for (int i = 0; i < 10 && i < required_k; i++) {
        std::cout << G.primes[i] << " ";
    }
    std::cout << std::endl;

    // Read flattened inverse matrix
    f.seekg(h.inv_table_offset);
    G.inv_flat.resize((size_t)required_k * required_k);

    if (required_k == (int)h.max_k) {
        // Read entire matrix at once
        f.read(reinterpret_cast<char*>(G.inv_flat.data()),
               (size_t)required_k * required_k * sizeof(uint64_t));
    } else {
        // CRITICAL FIX: When reading a submatrix, the file contains max_k columns per row
        // but we're storing only required_k columns per row
        std::vector<uint64_t> temp_row(h.max_k);
        for (int row = 0; row < required_k; ++row) {
            // Seek to the start of this row in the file
            f.seekg(h.inv_table_offset + (size_t)row * h.max_k * sizeof(uint64_t));
            
            // Read the full row from file
            f.read(reinterpret_cast<char*>(temp_row.data()), 
                   h.max_k * sizeof(uint64_t));
            
            // Copy only the first required_k elements to our matrix
            std::copy(temp_row.begin(), temp_row.begin() + required_k,
                     G.inv_flat.begin() + row * required_k);
        }
    }

    std::cout << "[Garner] Loaded " << required_k << "×" << required_k
              << " inverse matrix from " << filename << std::endl;

    // DEBUG: Print first few diagonal elements
    std::cout << "[Garner] First 5 diagonal inverses: ";
    for (int i = 0; i < 5 && i < required_k; i++) {
        std::cout << G.inv_flat[i * required_k + i] << " ";
    }
    std::cout << std::endl;

    return G;
}

// Product tree for CRT
CRTProductTree build_product_tree(const std::vector<u32>& m) {
    CRTProductTree T;
    T.levels.clear();

    std::vector<cpp_int> level0;
    level0.reserve(m.size());
    for (u32 mi : m) {
        level0.emplace_back(cpp_int(mi));
    }
    T.levels.push_back(std::move(level0));

    while (T.levels.back().size() > 1) {
        const auto &cur = T.levels.back();
        std::vector<cpp_int> nxt;
        nxt.reserve((cur.size() + 1) / 2);

        for (size_t i = 0; i < cur.size(); i += 2) {
            if (i + 1 < cur.size()) {
                nxt.push_back(cur[i] * cur[i + 1]);
            } else {
                nxt.push_back(cur[i]);
            }
        }
        T.levels.push_back(std::move(nxt));
    }
    return T;
}

void remainder_tree_down(const CRTProductTree &T, const cpp_int &N,
                         std::vector<cpp_int> &leaf_rems) {
    const size_t L = T.levels.size();
    if (L == 0) {
        leaf_rems.clear();
        return;
    }

    std::vector<std::vector<cpp_int>> rems(L);
    rems[L - 1].resize(1);
    rems[L - 1][0] = N % T.levels[L - 1][0];

    for (size_t level = L - 1; level > 0; --level) {
        const auto &cur_products = T.levels[level - 1];
        const auto &parent_rems = rems[level];
        auto &cur_rems = rems[level - 1];
        cur_rems.resize(cur_products.size());

        size_t idx_child = 0;
        for (size_t i = 0; i < rems[level].size(); ++i) {
            const cpp_int &R = parent_rems[i];
            const cpp_int &Lprod = cur_products[idx_child];
            cur_rems[idx_child] = R % Lprod;
            ++idx_child;

            if (idx_child < cur_products.size()) {
                const cpp_int &Rprod = cur_products[idx_child];
                cur_rems[idx_child] = R % Rprod;
                ++idx_child;
            }
        }
    }
    leaf_rems = std::move(rems[0]);
}

std::vector<u32> choose_moduli_dynamic(const std::vector<u32>& primes,
    const cpp_int& N, int safety_bits,
    int* out_k) {
 // ADD THIS DEBUG
 printf("[choose_moduli_dynamic] Called with %zu primes\n", primes.size());
 printf("[choose_moduli_dynamic] First 10 primes: ");
 for (int i = 0; i < 10 && i < (int)primes.size(); i++) {
     printf("%u ", primes[i]);
 }
 printf("\n");
 printf("[choose_moduli_dynamic] Last 10 primes: ");
 for (int i = std::max(0, (int)primes.size() - 10); i < (int)primes.size(); i++) {
     printf("%u ", primes[i]);
 }
 printf("\n");
size_t nbits = bitlen_cppint(N);
double target_bits = (double)nbits + (double)safety_bits;

std::vector<u32> m;
m.reserve(128);
double acc_bits = 0.0;

// Iterate forward through largest primes
for (size_t idx = 0; idx < primes.size(); ++idx) {
u32 p = primes[idx];
acc_bits += std::log2((double)p);
m.push_back(p);
if (acc_bits >= target_bits) break;
}

// No reverse needed - already in correct order
if (out_k) *out_k = (int)m.size();
return m;
}


std::vector<u64> garner_from_residues(const std::vector<u64>& r,
    const std::vector<u64>& m, const GarnerTable& G) {
    const size_t k = m.size();
    if (r.size() != k) {
        throw std::runtime_error("garner: size mismatch");
    }

    // ADD THIS DEBUG
    printf("[garner_from_residues] k=%zu, G.max_k=%u, G.inv_flat.size()=%zu\n", 
           k, G.max_k, G.inv_flat.size());
    printf("[garner_from_residues] Expected inv_flat size for k=%zu: %zu\n", 
           k, k * k);
    
    std::vector<u64> a(k);
    a[0] = r[0] % m[0];

    for (size_t j = 1; j < k; ++j) {
        u64 sum = 0;
        u64 prod = 1;
        
        for (size_t i = 0; i < j; ++i) {
            u64 ai_mod = a[i] % m[j];
            u64 term = (u64)(((__uint128_t)ai_mod * (__uint128_t)prod) % (__uint128_t)m[j]);
            sum = (sum + term) % m[j];
            
            u64 mi = m[i] % m[j];
            prod = (u64)(((__uint128_t)prod * (__uint128_t)mi) % (__uint128_t)m[j]);
        }
        
        u64 rj_mod = r[j] % m[j];
        u64 diff;
        if (rj_mod >= sum) {
            diff = rj_mod - sum;
        } else {
            diff = rj_mod + m[j] - sum;
        }
        
        // ADD THIS DEBUG FOR FIRST FEW ITERATIONS
        if (j < 3) {
            size_t idx = j * k + j;
            printf("[garner j=%zu] Accessing G.inv_flat[%zu], m[j]=%llu, prod=%llu, diff=%llu\n",
                   j, idx, (unsigned long long)m[j], (unsigned long long)prod, (unsigned long long)diff);
        }
        
        u64 inv_jj = G.inv_flat[j * G.max_k + j];
        
        if (j < 3) {
            printf("[garner j=%zu] inv_jj=%llu, computing a[j]...\n", 
                   j, (unsigned long long)inv_jj);
        }
        
        a[j] = (u64)(((__uint128_t)diff * (__uint128_t)inv_jj) % (__uint128_t)m[j]);
        
        if (j < 3) {
            printf("[garner j=%zu] a[j]=%llu\n", j, (unsigned long long)a[j]);
        }
    }

    // ADD THIS
    printf("[garner_from_residues] First 5 coefficients: ");
    for (size_t i = 0; i < 5 && i < k; i++) {
        printf("%llu ", (unsigned long long)a[i]);
    }
    printf("\n");

    return a;
}