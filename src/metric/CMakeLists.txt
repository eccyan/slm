add_library(metric STATIC
    src/simd_ops.cpp
    src/fisher_rao.cpp
)

target_include_directories(metric PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(metric PUBLIC cxx_std_23)
