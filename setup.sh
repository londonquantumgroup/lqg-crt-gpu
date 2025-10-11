#!/bin/bash
set -e

echo "=========================================="
echo "CRT GPU Benchmark - Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on a system with CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc (CUDA compiler) not found!${NC}"
    echo "Please install CUDA toolkit first."
    exit 1
fi

echo -e "${GREEN}✓ CUDA found: $(nvcc --version | grep release)${NC}"

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
    UPDATE_CMD="apt-get update -qq"
    INSTALL_CMD="apt-get install -y"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    UPDATE_CMD="yum check-update || true"
    INSTALL_CMD="yum install -y"
else
    echo -e "${RED}ERROR: No supported package manager found (apt-get or yum)${NC}"
    exit 1
fi

echo -e "${YELLOW}Using package manager: $PKG_MANAGER${NC}"

# Check for sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then 
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
        echo -e "${YELLOW}Using sudo for package installation${NC}"
    else
        echo -e "${RED}ERROR: Not running as root and sudo not found${NC}"
        exit 1
    fi
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
$SUDO $UPDATE_CMD

if [ "$PKG_MANAGER" = "apt-get" ]; then
    $SUDO $INSTALL_CMD \
        libgmp-dev \
        libgmpxx4ldbl \
        libboost-all-dev \
        build-essential \
        cmake \
        git
elif [ "$PKG_MANAGER" = "yum" ]; then
    $SUDO $INSTALL_CMD \
        gmp-devel \
        boost-devel \
        cmake \
        git \
        gcc-c++
fi

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Clone CGBN if not already present
echo ""
echo "Setting up CGBN library..."
if [ -d "CGBN" ]; then
    echo -e "${YELLOW}CGBN directory already exists, skipping clone${NC}"
else
    echo "Cloning CGBN from GitHub..."
    git clone https://github.com/NVlabs/CGBN.git
    echo -e "${GREEN}✓ CGBN cloned${NC}"
fi

# Create cmake directory and FindGMP.cmake
echo ""
echo "Creating CMake modules..."
mkdir -p cmake

cat > cmake/FindGMP.cmake << 'EOF'
# FindGMP.cmake - CMake module to find GMP library

find_path(GMP_INCLUDE_DIR 
    NAMES gmp.h 
    PATHS /usr/include /usr/local/include
)

find_library(GMP_LIBRARIES 
    NAMES gmp libgmp 
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib64
)

find_library(GMPXX_LIBRARIES 
    NAMES gmpxx libgmpxx 
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP 
    DEFAULT_MSG 
    GMP_INCLUDE_DIR 
    GMP_LIBRARIES
)

mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARIES GMPXX_LIBRARIES)

if(GMP_FOUND)
    set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
    if(NOT TARGET GMP::GMP)
        add_library(GMP::GMP UNKNOWN IMPORTED)
        set_target_properties(GMP::GMP PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
        )
    endif()
endif()
EOF

echo -e "${GREEN}✓ FindGMP.cmake created${NC}"

# Detect GPU architecture
echo ""
echo "Detecting GPU architecture..."
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "Detected GPU: $GPU_NAME"
    
    # Map common GPUs to compute capability
    if [[ $GPU_NAME == *"T4"* ]] || [[ $GPU_NAME == *"V100"* ]]; then
        GPU_ARCH="75"
    elif [[ $GPU_NAME == *"A100"* ]] || [[ $GPU_NAME == *"A30"* ]]; then
        GPU_ARCH="80"
    elif [[ $GPU_NAME == *"A10"* ]] || [[ $GPU_NAME == *"RTX 30"* ]] || [[ $GPU_NAME == *"RTX 40"* ]]; then
        GPU_ARCH="86"
    elif [[ $GPU_NAME == *"H100"* ]]; then
        GPU_ARCH="90"
    elif [[ $GPU_NAME == *"L4"* ]]; then
        GPU_ARCH="89"
    else
        GPU_ARCH="75"  # Default to Turing
        echo -e "${YELLOW}Unknown GPU, defaulting to compute capability 7.5${NC}"
    fi
else
    GPU_ARCH="75"  # Default
    echo -e "${YELLOW}nvidia-smi not found, defaulting to compute capability 7.5${NC}"
fi

echo "Detected CUDA architecture: $GPU_ARCH"

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Setup complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Build the project:"
echo "   mkdir -p build && cd build"
echo "   cmake .. -DCMAKE_MODULE_PATH=../cmake \\"
echo "            -DCGBN_INCLUDE_DIR=../CGBN/include \\"
echo "            -DDIVISOR_BITS=32 \\"
echo "            -DCGBN_BITS=128 \\"
echo "            -DCGBN_TPI=32 \\"
echo "            -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
echo "   make -j\$(nproc)"
echo ""
echo "2. Run benchmark:"
echo "   ./crt_benchmark 12345678901234567890 1000000"
echo ""
echo "Build options you can customize:"
echo "  -DDIVISOR_BITS=<32|64|128>  - Size of divisors"
echo "  -DCGBN_BITS=<96|128|256|512> - CGBN bit width"
echo "  -DCGBN_TPI=<4|8|16|32>       - CGBN threads per instance"
echo ""