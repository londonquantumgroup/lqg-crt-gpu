#pragma once
#include "types.hpp"

// AUTO-GENERATED: Precomputed Garner inverse diagonal
// Do not edit manually - regenerate with precompute_garner_inverses
// Storage: 128 bytes for k=16

namespace PrecomputedInverses {

// Garner inverse diagonal for k=16
// inv_diag[i] = (m[0]*m[1]*...*m[i-1])^{-1} mod m[i]
constexpr size_t K_16 = 16;
constexpr u64 INV_DIAG_16[16] = {
    1ull, 357913940ull, 1146815903ull, 1822974893ull, 2648723231ull, 1142029989ull, 1401891989ull, 2309466289ull,
    4071227427ull, 822361013ull, 1917678692ull, 1782334110ull, 1086130749ull, 3977513848ull, 4047754231ull, 1704937495ull
};

} // namespace PrecomputedInverses
