cmake_minimum_required(VERSION 3.20)
project(slmfs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# SIMD detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)

if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
    add_compile_definitions(SLM_HAS_AVX2)
elseif(COMPILER_SUPPORTS_SSE42)
    add_compile_options(-msse4.2)
    add_compile_definitions(SLM_HAS_SSE42)
endif()

# On ARM (Apple Silicon / Linux ARM), NEON is always available in AArch64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    add_compile_definitions(SLM_HAS_NEON)
endif()

add_subdirectory(src/slab)
add_subdirectory(src/metric)
add_subdirectory(src/langevin)
add_subdirectory(src/sheaf)
add_subdirectory(src/persist)
add_subdirectory(src/engine)

# Testing
option(SLM_BUILD_TESTS "Build tests" ON)
if(SLM_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()
