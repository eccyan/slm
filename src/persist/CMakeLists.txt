find_package(SQLite3 REQUIRED)

add_library(persist STATIC
    src/sqlite_store.cpp
)

target_include_directories(persist PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# persist needs engine headers (MemoryGraph) but must not link the engine
# library to avoid a circular dependency (engine -> persist -> engine).
# We expose engine's include dir and its transitive header deps instead.
target_include_directories(persist PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/include
)

target_link_libraries(persist PUBLIC metric langevin SQLite::SQLite3)

target_compile_features(persist PUBLIC cxx_std_23)
