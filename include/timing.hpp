#pragma once
#include <chrono>

static inline std::chrono::high_resolution_clock::time_point now_tp() {
    return std::chrono::high_resolution_clock::now();
}

static inline double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0
    ).count();
}