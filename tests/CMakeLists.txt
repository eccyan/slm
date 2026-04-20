add_executable(slab_tests
    test_header.cpp
    test_ring_buffer.cpp
    test_slab_allocator.cpp
)

target_link_libraries(slab_tests PRIVATE
    slab
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(slab_tests)

add_executable(metric_tests
    test_metric.cpp
)

target_link_libraries(metric_tests PRIVATE
    metric
    GTest::gtest_main
)

gtest_discover_tests(metric_tests)
