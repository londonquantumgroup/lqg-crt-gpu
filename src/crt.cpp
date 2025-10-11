#include "crt.hpp"
#include "modular_arithmetic.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

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

std::vector<u64> garner_from_residues(const std::vector<u64>& r, 
                                      const std::vector<u64>& m) {
    const size_t k = m.size();
    if (r.size() != k) {
        throw std::runtime_error("garner: size mismatch");
    }

    // inv[i][j] = (m[0] * m[1] * ... * m[i-1])^{-1} mod m[j]
    std::vector<std::vector<u64>> inv(k, std::vector<u64>(k, 1));
    for (size_t i = 1; i < k; ++i) {
        for (size_t j = i; j < k; ++j) {
            u64 m_inv = modinv_u64(m[i - 1] % m[j], m[j]);
            inv[i][j] = (u64)(((__uint128_t)inv[i - 1][j] * 
                              (__uint128_t)m_inv) % (__uint128_t)m[j]);
        }
    }

    std::vector<u64> c(k);
    c[0] = r[0] % m[0];

    for (size_t i = 1; i < k; ++i) {
        u64 sum = c[0] % m[i];
        u64 prod = 1;

        for (size_t j = 1; j < i; ++j) {
            prod = (u64)(((__uint128_t)prod * 
                         (__uint128_t)(m[j - 1] % m[i])) % (__uint128_t)m[i]);
            u64 term = (u64)(((__uint128_t)(c[j] % m[i]) * 
                            (__uint128_t)prod) % (__uint128_t)m[i]);
            sum = (sum + term) % m[i];
        }

        u64 diff = (r[i] % m[i] >= sum) ? 
                   (r[i] % m[i] - sum) : 
                   (r[i] % m[i] + m[i] - sum);
        c[i] = (u64)(((__uint128_t)diff * 
                     (__uint128_t)inv[i][i]) % (__uint128_t)m[i]);
    }

    return c;
}