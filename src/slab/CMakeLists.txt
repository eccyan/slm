add_library(slab STATIC
    src/slab_allocator.cpp
)

target_include_directories(slab PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(slab PUBLIC cxx_std_23)
