add_executable(slab_tests
    test_header.cpp
    test_ring_buffer.cpp
)

target_link_libraries(slab_tests PRIVATE
    slab
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(slab_tests)
