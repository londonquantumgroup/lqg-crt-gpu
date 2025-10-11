#include "../include/crt_utils.hpp"
#include "../include/crt_host_helpers.hpp"

// choose CRT moduli to cover |N| + safety bits
std::vector<u32> choose_moduli_dynamic(const std::vector<u32>& primes,
                                       const cpp_int& N,
                                       int safety_bits,
                                       int* out_k=nullptr) {
  size_t nbits = bitlen_cppint(N);
  double target_bits = (double)nbits + (double)safety_bits;
  std::vector<u32> m; m.reserve(128);
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

// generate divisors (32/64/128+) excluding CRT moduli
std::vector<cpp_int> generate_divisors(int M, int divisor_bits, const std::unordered_set<u32>& exclude_set) {
  std::vector<cpp_int> divisors; divisors.reserve(M);

  if (divisor_bits == 32) {
    const auto& gp = global_primes();
    for (size_t i=0; i<gp.size() && (int)divisors.size()<M; ++i) {
      u32 p = gp[i];
      if (exclude_set.find(p) == exclude_set.end())
        divisors.emplace_back(cpp_int(p));
    }
  } else if (divisor_bits == 64) {
    const auto& gp = global_primes();
    for (size_t i=0; i+1<gp.size() && (int)divisors.size()<M; i += 2) {
      u64 val = ((u64)gp[i] << 32) | gp[i+1];
      divisors.emplace_back(cpp_int(val));
    }
  } else {
    cpp_int base = cpp_int(1) << (divisor_bits - 1);
    for (int i=0;i<M;++i) {
      cpp_int offset = cpp_int(i * 104729 + 1234567);
      cpp_int d = base + offset;
      if ((d & 1) == 0) d += 1;
      divisors.emplace_back(d);
    }
  }
  return divisors;
}