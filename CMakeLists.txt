cmake_minimum_required(VERSION 3.20)
project(lqg_crt_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(ENABLE_CGBN "Enable CGBN/GMP paths (you must have deps installed)" OFF)

# You can tweak these to match your card
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

add_library(crt_headers INTERFACE)
target_include_directories(crt_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find Boost (header-only multiprecision)
find_package(Boost REQUIRED)

add_executable(lqg_crt_gpu
  src/main.cpp
  src/crt_cpu.cpp
  src/crt_gpu.cu
)

target_link_libraries(lqg_crt_gpu PRIVATE crt_headers)

# CUDA
set_target_properties(lqg_crt_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# NVCC flags
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

if (ENABLE_CGBN)
  target_compile_definitions(lqg_crt_gpu PRIVATE CGBN_USE_GMP=1)
  # Users must provide include/link paths for GMP/CGBN if not system-wide
  find_path(GMP_INCLUDE_DIR NAMES gmpxx.h gmp.h)
  find_library(GMP_LIB NAMES gmp)
  find_library(GMPXX_LIB NAMES gmpxx)
  if (GMP_INCLUDE_DIR AND GMP_LIB AND GMPXX_LIB)
    target_include_directories(lqg_crt_gpu PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(lqg_crt_gpu PRIVATE ${GMPXX_LIB} ${GMP_LIB})
  else()
    message(FATAL_ERROR "ENABLE_CGBN=ON but GMP not found")
  endif()
endif()

# Boost headers
target_include_directories(lqg_crt_gpu PRIVATE ${Boost_INCLUDE_DIRS})