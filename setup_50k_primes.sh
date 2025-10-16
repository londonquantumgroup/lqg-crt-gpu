#!/bin/bash
set -e

echo "=========================================="
echo "CRT GPU Benchmark - 50K Prime Setup"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Generate 50,000 optimal 32-bit primes (~5-10 min)"
echo "  2. Precompute Garner inverse tables (~20-30 min)"
echo "  3. Build the main benchmark project"
echo ""
echo "Total time: ~30-40 minutes (one-time setup)"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check for required tools
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}ERROR: g++ not found${NC}"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}WARNING: nvcc (CUDA) not found. Skipping CUDA-dependent steps. CPU-only precomputation will continue.${NC}"
fi

echo -e "${GREEN}✓ Found required tools${NC}"
echo ""

# Step 1: Generate prime table
echo -e "${BLUE}=== Step 1/3: Generating 50,000 Primes ===${NC}"
echo "Building prime generator..."
g++ -o generate_prime_table generate_prime_table.cpp -std=c++17 -O3 -march=native

echo ""
echo "Generating primes (this will take 5-10 minutes)..."
./generate_prime_table

if [ ! -f "src/optimal_primes_data.cpp" ]; then
    echo -e "${RED}ERROR: Failed to generate prime data${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prime table generated${NC}"
echo ""

# Step 2: Precompute inverse tables
echo -e "${BLUE}=== Step 2/3: Precomputing Inverse Tables ===${NC}"
echo "Building inverse table generator..."
g++ -o precompute_garner_inverses precompute_garner_inverses.cpp \
    src/modular_arithmetic.cpp src/optimal_primes_data.cpp \
    -I include -std=c++17 -O3 -march=native

echo ""
echo "Computing inverse tables (this will take 20-30 minutes)..."
echo "Progress will be shown for each table size..."
echo ""
./precompute_garner_inverses

# Verify key files were generated
if [ ! -f "include/garner_inv_k50000.hpp" ]; then
    echo -e "${RED}ERROR: Failed to generate inverse tables${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Inverse tables precomputed${NC}"
echo ""
