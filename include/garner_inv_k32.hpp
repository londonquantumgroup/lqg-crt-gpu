#pragma once
#include "types.hpp"

// AUTO-GENERATED: Precomputed Garner inverse diagonal
// Do not edit manually - regenerate with precompute_garner_inverses
// Storage: 256 bytes for k=32

namespace PrecomputedInverses {

// Garner inverse diagonal for k=32
// inv_diag[i] = (m[0]*m[1]*...*m[i-1])^{-1} mod m[i]
constexpr size_t K_32 = 32;
constexpr u64 INV_DIAG_32[32] = {
    1ull, 357913940ull, 1146815903ull, 1822974893ull, 2648723231ull, 1142029989ull, 1401891989ull, 2309466289ull,
    4071227427ull, 822361013ull, 1917678692ull, 1782334110ull, 1086130749ull, 3977513848ull, 4047754231ull, 1704937495ull,
    3596520247ull, 2132102508ull, 1833086972ull, 2760004605ull, 3174238409ull, 2842082017ull, 4046293999ull, 3572966423ull,
    1695412035ull, 1382636218ull, 789519626ull, 3059901629ull, 1458000307ull, 4015703733ull, 1442033695ull, 3237936451ull
};

} // namespace PrecomputedInverses
