#include "crt.hpp"
#include "modular_arithmetic.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>

// Load Garner table from file
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

// --- Garner reconstruction using O(k^2) Basis Transformation ---
std::vector<u64> garner_from_residues(const std::vector<u64>& r,
    const std::vector<u64>& m, const GarnerTable& G) {
    const size_t k = m.size();
    if (r.size() != k) {
        throw std::runtime_error("garner: size mismatch");
    }

    std::vector<u64> c(k); // Mixed-radix coefficients

    for (size_t i = 0; i < k; ++i) {
        // V_i is the current remaining residue, initially r_i mod m_i
        u64 V_i = r[i] % m[i];

        // --- Inner loop: Basis Transformation (O(1) per step) ---
        for (size_t j = 0; j < i; ++j) {
            u64 cj_mod = c[j] % m[i];
            V_i = (V_i >= cj_mod) ?
                  (V_i - cj_mod) :
                  (V_i + m[i] - cj_mod);

            u64 inv_ji = G.inv_flat[j * k + i];
            V_i = (u64)(((__uint128_t)V_i * (__uint128_t)inv_ji) % (__uint128_t)m[i]);
        }
        c[i] = V_i;
    }

    return c;
}