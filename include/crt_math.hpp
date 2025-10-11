#pragma once
#include "crt_utils.hpp"

inline long long egcd(long long a, long long b, long long &x, long long &y) {
  if (b == 0) { x = 1; y = 0; return a; }
  long long x1, y1;
  long long g = egcd(b, a % b, x1, y1);
  x = y1; y = x1 - (a / b) * y1;
  return g;
}

inline u64 modinv_u64(u64 a, u64 m) {
  if (m == 0) throw std::runtime_error("modinv_u64(): divide by zero");
  long long x, y; long long g = egcd((long long)a, (long long)m, x, y);
  if (g != 1) throw std::runtime_error("modinv_u64(): inverse does not exist");
  long long inv = x % (long long)m;
  if (inv < 0) inv += (long long)m;
  return (u64)inv;
}

// Inverse modulo 2^64 for odd a using Newton iteration
inline u64 modinv64(u64 a) {
  u64 x = 1;
  x *= 2 - a * x;
  x *= 2 - a * x;
  x *= 2 - a * x;
  x *= 2 - a * x;
  x *= 2 - a * x;
  x *= 2 - a * x;
  return x;
}