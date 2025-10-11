#pragma once
#include "types.hpp"
#include <stdexcept>

long long egcd(long long a, long long b, long long &x, long long &y);
u64 modinv_u64(u64 a, u64 m);
u64 modinv64(u64 a);