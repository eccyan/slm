find_package(SQLite3 REQUIRED)

add_library(persist STATIC
    src/sqlite_store.cpp
)

target_include_directories(persist PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(persist PUBLIC engine SQLite::SQLite3)

target_compile_features(persist PUBLIC cxx_std_23)
