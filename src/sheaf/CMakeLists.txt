add_library(sheaf STATIC
    src/coboundary.cpp
    src/annotation.cpp
)

target_include_directories(sheaf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sheaf PUBLIC metric)

target_compile_features(sheaf PUBLIC cxx_std_23)
