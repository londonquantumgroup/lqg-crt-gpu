# Garner Precomputation Tables

This directory contains precomputed modular inverse tables for the Garner CRT algorithm.

## Included in Repository (Git LFS)

| File | Size | Max k | Max Bits (approx) |
|------|------|-------|-------------------|
| `garner_k1k.bin` | 8 MB | 1,000 | ~250K bits |
| `garner_k4k.bin` | 128 MB | 4,000 | ~1M bits |

## Download from S3 (Large Files)

| File | Size | Max k | Max Bits (approx) | Download |
|------|------|-------|-------------------|----------|
| `garner_k10k.bin` | 800 MB | 10,000 | ~2.5M bits | Use download script |
| `garner_k20k.bin` | 3.2 GB | 20,000 | ~5M bits | Use download script |
| `garner_k40k.bin` | 12.8 GB | 40,000 | ~10M bits | Use download script |

## Downloading Large Tables
```bash
cd data
./download_large_tables.sh
```

Or download directly:
```bash
cd data

# 10k table (800 MB)
wget https://lqg-crt-inverse-matrix-data.s3.eu-north-1.amazonaws.com/garner_k10k.bin

# 20k table (3.2 GB)
wget https://lqg-crt-inverse-matrix-data.s3.eu-north-1.amazonaws.com/garner_k20k.bin

# 40k table (12.8 GB)
wget https://lqg-crt-inverse-matrix-data.s3.eu-north-1.amazonaws.com/garner_k40k.bin
```

## Storage Requirements

To run benchmarks with different number sizes, you'll need:

| Target Number Size | Required Tables | Total Storage |
|-------------------|-----------------|---------------|
| Up to  2k | k1k only | 8 MB |
| Up to 16k bits | k1k, k4k | 136 MB |
| Up to 250k bits | k1k, k4k, k10k | ~936 MB |
| Up to 500K bits | k1k, k4k, k10k, k20k | ~4.1 GB |
| Up to 1M bits | All tables | ~16.9 GB |

## File Format

Each `.bin` file contains:
- Header with magic number and version
- Array of prime numbers used for CRT
- Precomputed modular inverse matrix (flattened)

See `src/crt.cpp` for the loading implementation.