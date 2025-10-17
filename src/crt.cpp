#include "crt.hpp"
#include "modular_arithmetic.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>

// --- Garner precomputed table ---
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

// loads the file and returns table (only first required_k entries)
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

    // Read flattened inverse matrix (only first k×k)
    f.seekg(h.inv_table_offset);
    G.inv_flat.resize((size_t)required_k * required_k);
    f.read(reinterpret_cast<char*>(G.inv_flat.data()),
           (size_t)required_k * required_k * sizeof(uint64_t));

    std::cout << "[Garner] Loaded " << required_k << "×" << required_k
              << " inverse matrix from " << filename << std::endl;

    return G;
}

// --- Product tree for CRT ---
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
    size_t nbits = bitlen_cppint(N);
    double target_bits = (double)nbits + (double)safety_bits;

    std::vector<u32> m;
    m.reserve(128);
    double acc_bits = 0.0;

    for (size_t idx = primes.size(); idx-- > 0; ) {
        u32 p = primes[idx];
        acc_bits += std::log2((double)p);
        m.push_back(p);
        if (acc_bits >= target_bits) break;
    }

    std::reverse(m.begin(), m.end());
    if (out_k) *out_k = (int)m.size();
    return m;
}

// --- Garner reconstruction using precomputed inverse table ---
std::vector<u64> garner_from_residues(const std::vector<u64>& r,
                                      const std::vector<u64>& m) {
    const size_t k = m.size();
    if (r.size() != k) {
        throw std::runtime_error("garner: size mismatch");
    }

    // --- choose which precomputed table to load ---
    std::string table_file;
    if (k <= 1000) table_file = "data/garner_k1k.bin";
    else if (k <= 4000) table_file = "data/garner_k4k.bin";
    else if (k <= 10000) table_file = "data/garner_k10k.bin";
    else if (k <= 20000) table_file = "data/garner_k20k.bin";
    else table_file = "data/garner_k40k.bin";

    GarnerTable G = load_garner_table(table_file, (int)k);

    std::vector<u64> c(k);
    c[0] = r[0] % m[0];

    for (size_t i = 1; i < k; ++i) {
        u64 sum = c[0] % m[i];
        for (size_t j = 1; j < i; ++j) {
            // FIX: The index for M_j^{-1} mod m_i must be (j, i) in the
            // row-major flattened matrix, NOT (i, j).
            u64 inv_ij = G.inv_flat[j * k + i]; 
            
            u64 term = (u64)(((__uint128_t)(c[j] % m[i]) * inv_ij) % m[i]);
            sum = (sum + term) % m[i];
        }

        u64 diff = (r[i] % m[i] >= sum)
                       ? (r[i] % m[i] - sum)
                       : (r[i] % m[i] + m[i] - sum);

        // This diagonal index (i, i) was already correct.
        u64 inv_ii = G.inv_flat[i * k + i];
        c[i] = (u64)(((__uint128_t)diff * inv_ii) % m[i]);
    }

    return c;
}