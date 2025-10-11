#pragma once
#include "crt_math.hpp"

inline std::vector<u64> garner_from_residues(const std::vector<u64>& r,
                                             const std::vector<u64>& m) {
  const size_t k = m.size();
  if (r.size() != k) throw std::runtime_error("garner: size mismatch");

  std::vector<std::vector<u64>> inv(k, std::vector<u64>(k, 1));
  for (size_t i = 1; i < k; ++i) {
    for (size_t j = i; j < k; ++j) {
      u64 m_inv = modinv_u64(m[i-1] % m[j], m[j]);
      inv[i][j] = (u64)(((__uint128_t)inv[i-1][j] * (__uint128_t)m_inv) % (__uint128_t)m[j]);
    }
  }

  std::vector<u64> c(k);
  c[0] = r[0] % m[0];

  for (size_t i = 1; i < k; ++i) {
    u64 sum = c[0] % m[i];
    u64 prod = 1; // product of previous moduli, progressively
    for (size_t j = 1; j < i; ++j) {
      prod = (u64)(((__uint128_t)prod * (__uint128_t)(m[j-1] % m[i])) % (__uint128_t)m[i]);
      u64 term = (u64)(((__uint128_t)(c[j] % m[i]) * (__uint128_t)prod) % (__uint128_t)m[i]);
      sum = (sum + term) % m[i];
    }
    u64 ri = r[i] % m[i];
    u64 diff = (ri >= sum) ? (ri - sum) : (ri + m[i] - sum);
    c[i] = (u64)(((__uint128_t)diff * (__uint128_t)inv[i][i]) % (__uint128_t)m[i]);
  }
  return c;
}