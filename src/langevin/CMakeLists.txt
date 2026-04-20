add_library(langevin STATIC
    src/sde_stepper.cpp
)

target_include_directories(langevin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(langevin PUBLIC cxx_std_23)
