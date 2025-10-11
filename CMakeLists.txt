cmake_minimum_required(VERSION 3.20)
project(lqg_crt_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(ENABLE_CGBN "Enable CGBN/GMP paths (you must have deps installed)" OFF)

# You can tweak these to match your GPU
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

add_library(crt_headers INTERFACE)
target_include_directories(crt_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find Boost (header-only multiprecision)
find_package(Boost REQUIRED)

add_compile_options(-fpermissive)

# Source files
set(SOURCES
  src/main.cpp
  src/crt_cpu.cpp
  src/crt_gpu.cu
)

# Add CGBN sources if enabled
if (ENABLE_CGBN)
  list(APPEND SOURCES 
    src/crt_cgbn.cu
    src/cgbn_benchmark.cpp
  )
endif()

add_executable(lqg_crt_gpu ${SOURCES})

# Force CUDA compilation for .cpp files that use CUDA headers
set_source_files_properties(
    src/main.cpp
    src/cgbn_benchmark.cpp
    PROPERTIES LANGUAGE CUDA
)

target_link_libraries(lqg_crt_gpu PRIVATE crt_headers)

# CUDA build properties
set_target_properties(lqg_crt_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# DIVISOR_BITS support
if (NOT DEFINED DIVISOR_BITS)
  set(DIVISOR_BITS 128)
endif()
add_compile_definitions(DIVISOR_BITS=${DIVISOR_BITS})
message(STATUS "Using DIVISOR_BITS=${DIVISOR_BITS}")

# NVCC & CXX flags
target_compile_options(lqg_crt_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --use_fast_math
    -Xptxas=-O3,-v
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
  >
)

# Optional: Enable CGBN/GMP
if (ENABLE_CGBN)
  target_compile_definitions(lqg_crt_gpu PRIVATE ENABLE_CGBN=1)
  
  find_path(GMP_INCLUDE_DIR NAMES gmpxx.h gmp.h)
  find_library(GMP_LIB NAMES gmp)
  find_library(GMPXX_LIB NAMES gmpxx)
  
  if (GMP_INCLUDE_DIR AND GMP_LIB AND GMPXX_LIB)
    target_include_directories(lqg_crt_gpu PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(lqg_crt_gpu PRIVATE ${GMPXX_LIB} ${GMP_LIB})
    message(STATUS "CGBN enabled with GMP support")
  else()
    message(FATAL_ERROR "ENABLE_CGBN=ON but GMP not found")
  endif()
  
  # Find CGBN headers
  find_path(CGBN_INCLUDE_DIR 
    NAMES cgbn/cgbn.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/CGBN/include
          /usr/local/include
          /usr/include
  )
  
  if (CGBN_INCLUDE_DIR)
    target_include_directories(lqg_crt_gpu PRIVATE ${CGBN_INCLUDE_DIR})
    message(STATUS "Found CGBN at: ${CGBN_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "ENABLE_CGBN=ON but CGBN headers not found. "
                        "Run: git clone https://github.com/NVlabs/CGBN.git")
  endif()
else()
  message(STATUS "CGBN disabled (use -DENABLE_CGBN=ON to enable)")
endif()

# Boost headers
target_include_directories(lqg_crt_gpu PRIVATE ${Boost_INCLUDE_DIRS})