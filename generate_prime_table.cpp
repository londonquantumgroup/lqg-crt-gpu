#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

using u32 = uint32_t;

// Miller-Rabin primality test for deterministic checking
bool miller_rabin(uint64_t n, uint64_t a) {
    if (n == 2) return true;
    if (n < 2 || n % 2 == 0) return false;
    
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }
    
    // Compute a^d mod n
    auto mod_pow = [](uint64_t base, uint64_t exp, uint64_t mod) -> uint64_t {
        uint64_t result = 1;
        base %= mod;
        while (exp > 0) {
            if (exp & 1) {
                result = ((__uint128_t)result * base) % mod;
            }
            base = ((__uint128_t)base * base) % mod;
            exp >>= 1;
        }
        return result;
    };
    
    uint64_t x = mod_pow(a, d, n);
    if (x == 1 || x == n - 1) return true;
    
    for (int i = 0; i < r - 1; i++) {
        x = ((__uint128_t)x * x) % n;
        if (x == n - 1) return true;
    }
    return false;
}

// Deterministic primality test for 32-bit numbers
bool is_prime(u32 n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    // For 32-bit numbers, checking these witnesses is deterministic
    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t a : witnesses) {
        if (a >= n) continue;
        if (!miller_rabin(n, a)) return false;
    }
    return true;
}

// Generate the top N primes below 2^32
std::vector<u32> generate_top_primes(int count) {
    std::vector<u32> primes;
    primes.reserve(count);
    
    u32 candidate = 0xFFFFFFFF; // 2^32 - 1
    
    std::cout << "Finding top " << count << " primes below 2^32...\n";
    std::cout << "This will take several minutes for 50,000 primes...\n\n";
    
    int progress = 0;
    int last_percent = 0;
    
    while (primes.size() < (size_t)count && candidate > 2) {
        if (is_prime(candidate)) {
            primes.push_back(candidate);
            progress++;
            
            int percent = (100 * progress) / count;
            if (percent > last_percent) {
                std::cout << "Progress: " << percent << "% (" << progress 
                         << "/" << count << " primes found, current: " 
                         << candidate << ")\n";
                last_percent = percent;
            }
        }
        candidate--;
    }
    
    return primes;
}

void write_primes_to_cpp(const std::vector<u32>& primes, 
                         const std::string& filename) {
    std::ofstream out(filename);
    
    out << "#include \"optimal_primes.hpp\"\n\n";
    out << "// AUTO-GENERATED: Top " << primes.size() << " 32-bit primes\n";
    out << "// Do not edit manually - regenerate with generate_prime_table\n";
    out << "// Generated with deterministic Miller-Rabin primality test\n\n";
    out << "namespace OptimalPrimes {\n\n";
    out << "const u32 TOP_PRIMES_32BIT[" << primes.size() << "] = {\n";
    
    for (size_t i = 0; i < primes.size(); ++i) {
        if (i % 8 == 0) out << "    ";
        out << primes[i] << "u";
        if (i < primes.size() - 1) out << ",";
        if (i % 8 == 7 || i == primes.size() - 1) out << "\n";
        else out << " ";
    }
    
    out << "};\n\n";
    out << "} // namespace OptimalPrimes\n";
    out.close();
}

int main(int argc, char** argv) {
    const int NUM_PRIMES = 50000;
    
    std::cout << "=== Generating Top 50,000 32-bit Primes ===\n\n";
    std::cout << "This is a one-time generation that will take 5-10 minutes\n";
    std::cout << "The generated file will be ~1.5 MB\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto primes = generate_top_primes(NUM_PRIMES);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "\n=== Generation Complete ===\n";
    std::cout << "Time taken: " << duration << " seconds\n\n";
    std::cout << "Generated " << primes.size() << " primes\n";
    std::cout << "Largest: " << primes[0] << "\n";
    std::cout << "Smallest: " << primes.back() << "\n";
    
    // Calculate total entropy
    double total_bits = 0.0;
    for (u32 p : primes) {
        total_bits += log2((double)p);
    }
    std::cout << "Total bits of representation: " << (int)total_bits << " bits\n";
    std::cout << "Can represent numbers up to ~" << (int)(total_bits / 8) << " kilobytes\n\n";
    
    // Write to .cpp file
    write_primes_to_cpp(primes, "src/optimal_primes_data.cpp");
    std::cout << "Generated src/optimal_primes_data.cpp (~1.5 MB)\n\n";
    
    // Print security level info
    std::cout << "=== Security Levels Supported ===\n";
    std::cout << "(Using conservative 3× multiplier for post-quantum)\n\n";
    
    struct Level { int bits; const char* name; };
    Level levels[] = {
        {128, "Standard Post-Quantum"},
        {192, "High Post-Quantum"},
        {256, "Very High Post-Quantum"},
        {384, "Extreme Post-Quantum"},
        {512, "Paranoid Post-Quantum"}
    };
    
    for (const auto& level : levels) {
        int target_bits = 3 * level.bits;
        int k_needed = (target_bits + 31) / 32;
        double actual_bits = 0.0;
        for (int i = 0; i < k_needed && i < (int)primes.size(); ++i) {
            actual_bits += log2((double)primes[i]);
        }
        std::cout << level.bits << "-bit " << level.name << ":\n";
        std::cout << "  Needs: " << k_needed << " primes\n";
        std::cout << "  Provides: ~" << (int)actual_bits << " bits\n";
        std::cout << "  Status: " << (k_needed <= (int)primes.size() ? "✓ SUPPORTED" : "✗ INSUFFICIENT") << "\n\n";
    }
    
    std::cout << "=== Next Steps ===\n";
    std::cout << "1. Add src/optimal_primes_data.cpp to your CMakeLists.txt\n";
    std::cout << "2. Run: ./precompute_garner_inverses to generate inverse tables\n";
    std::cout << "3. Rebuild your project with zero-setup-time CRT!\n";
    
    return 0;
}