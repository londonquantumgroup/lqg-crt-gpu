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
set(CGBN_INCLUDE_DIR "" CACHE PATH "Path to CGBN include directory")

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

# CGBN configuration
if(NO_CGBN)
    add_compile_definitions(NO_CGBN)
    message(STATUS "Building WITHOUT CGBN support")
else()
    # Try to find CGBN
    if(CGBN_INCLUDE_DIR)
        if(EXISTS "${CGBN_INCLUDE_DIR}/cgbn/cgbn.h")
            include_directories(${CGBN_INCLUDE_DIR})
            message(STATUS "CGBN found at: ${CGBN_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "CGBN_INCLUDE_DIR specified but cgbn.h not found at ${CGBN_INCLUDE_DIR}/cgbn/cgbn.h")
        endif()
    else()
        # Try common locations
        set(CGBN_SEARCH_PATHS
            "/content/cgbn/include"           # Google Colab
            "/usr/local/include"              # System-wide
            "${CMAKE_SOURCE_DIR}/../cgbn/include"  # Adjacent to project
            "$ENV{HOME}/cgbn/include"         # User home
        )
        
        foreach(SEARCH_PATH ${CGBN_SEARCH_PATHS})
            if(EXISTS "${SEARCH_PATH}/cgbn/cgbn.h")
                set(CGBN_INCLUDE_DIR ${SEARCH_PATH})
                include_directories(${CGBN_INCLUDE_DIR})
                message(STATUS "CGBN auto-detected at: ${CGBN_INCLUDE_DIR}")
                break()
            endif()
        endforeach()
        
        if(NOT CGBN_INCLUDE_DIR)
            message(WARNING "CGBN not found! Set CGBN_INCLUDE_DIR or build with -DNO_CGBN=ON")
            message(WARNING "Automatically enabling NO_CGBN")
            add_compile_definitions(NO_CGBN)
            set(NO_CGBN ON CACHE BOOL "Build without CGBN support" FORCE)
        endif()
    endif()
endif()

# Executable
add_executable(crt_benchmark
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Set CUDA architecture - make configurable
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()
set_property(TARGET crt_benchmark PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

message(STATUS "Target CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

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

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "CRT Benchmark Build Configuration")
message(STATUS "========================================")
message(STATUS "DIVISOR_BITS: ${DIVISOR_BITS}")
message(STATUS "CGBN_BITS: ${CGBN_BITS}")
message(STATUS "CGBN_TPI: ${CGBN_TPI}")
message(STATUS "NO_CGBN: ${NO_CGBN}")
message(STATUS "FAST_REMAINDER_TREE: ${FAST_REMAINDER_TREE}")
message(STATUS "GPU_GENERATE_DIVISORS: ${GPU_GENERATE_DIVISORS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT NO_CGBN AND CGBN_INCLUDE_DIR)
    message(STATUS "CGBN Include: ${CGBN_INCLUDE_DIR}")
endif()
message(STATUS "========================================")
message(STATUS "")