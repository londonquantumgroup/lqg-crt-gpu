cmake_minimum_required(VERSION 3.18)
project(CRT_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(NO_CGBN "Build without CGBN support" OFF)
set(DIVISOR_BITS "128" CACHE STRING "Divisor bit width")
set(CGBN_BITS "256" CACHE STRING "CGBN bit width")
set(CGBN_TPI "32" CACHE STRING "CGBN threads per instance")
set(FAST_REMAINDER_TREE "1" CACHE STRING "Use fast remainder tree")
set(GPU_GENERATE_DIVISORS "1" CACHE STRING "Generate divisors on GPU")

# Find packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(GMP REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIRS}
)

# Source files
set(CPP_SOURCES
    src/primes.cpp
    src/modular_arithmetic.cpp
    src/crt.cpp
    src/divisors.cpp
)

set(CUDA_SOURCES
    src/gpu_kernels.cu
    src/main.cpp
)

# Compile definitions
add_compile_definitions(
    DIVISOR_BITS=${DIVISOR_BITS}
    CGBN_BITS=${CGBN_BITS}
    CGBN_TPI=${CGBN_TPI}
    FAST_REMAINDER_TREE=${FAST_REMAINDER_TREE}
    GPU_GENERATE_DIVISORS=${GPU_GENERATE_DIVISORS}
)

if(NO_CGBN)
    add_compile_definitions(NO_CGBN)
else()
    # Add CGBN include path (adjust as needed)
    include_directories(/path/to/cgbn/include)
endif()

# Executable
add_executable(crt_benchmark
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Set CUDA architecture
set_property(TARGET crt_benchmark PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Link libraries
target_link_libraries(crt_benchmark
    ${Boost_LIBRARIES}
    ${GMP_LIBRARIES}
    gmpxx
)

# Compiler flags
target_compile_options(crt_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -Wall>
)