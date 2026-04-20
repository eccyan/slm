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

add_executable(langevin_tests
    test_langevin.cpp
)

target_link_libraries(langevin_tests PRIVATE
    langevin
    GTest::gtest_main
)

gtest_discover_tests(langevin_tests)

add_executable(sheaf_tests
    test_sheaf.cpp
)

target_link_libraries(sheaf_tests PRIVATE
    sheaf
    GTest::gtest_main
)

gtest_discover_tests(sheaf_tests)

add_executable(engine_tests
    test_memory_graph.cpp
    test_scheduler.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest_main
)

gtest_discover_tests(engine_tests)

add_executable(persist_tests
    test_persist.cpp
)

target_link_libraries(persist_tests PRIVATE
    persist
    engine
    GTest::gtest_main
)

gtest_discover_tests(persist_tests)
