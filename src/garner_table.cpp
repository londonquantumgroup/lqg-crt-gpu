#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "crt.hpp"  // for GarnerTable

struct GarnerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t max_k;
    uint32_t num_primes;
    uint64_t primes_offset;
    uint64_t inv_table_offset;
    uint64_t file_size;
};

GarnerTable load_garner_table(const std::string& filename, int required_k) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + filename);

    GarnerHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));

    if (h.magic != 0x47415442)
        throw std::runtime_error("Invalid magic number in Garner file");

    if (required_k > (int)h.max_k)
        throw std::runtime_error("Requested k larger than table size");

    // Read primes
    f.seekg(h.primes_offset);
    GarnerTable G;
    G.max_k = h.max_k;
    G.primes.resize(required_k);
    f.read(reinterpret_cast<char*>(G.primes.data()), required_k * sizeof(uint32_t));

    // Read flattened inverse matrix (only first k×k region)
    f.seekg(h.inv_table_offset);
    G.inv_flat.resize((size_t)required_k * required_k);
    f.read(reinterpret_cast<char*>(G.inv_flat.data()),
           (size_t)required_k * required_k * sizeof(uint64_t));

    std::cout << "Loaded Garner table: " << required_k
              << " primes from " << filename << std::endl;
    return G;
}