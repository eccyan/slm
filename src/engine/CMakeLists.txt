add_library(engine STATIC
    src/memory_graph.cpp
    src/scheduler.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC metric langevin slab sheaf persist)

target_compile_features(engine PUBLIC cxx_std_23)
