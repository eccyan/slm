add_library(engine STATIC
    src/memory_graph.cpp
    src/scheduler.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC metric langevin slab sheaf persist)

target_compile_features(engine PUBLIC cxx_std_23)

# Engine binary
add_executable(slmfs_engine
    src/main.cpp
)

target_link_libraries(slmfs_engine PRIVATE engine)

# POSIX shared memory requires -lrt on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(slmfs_engine PRIVATE rt)
endif()
