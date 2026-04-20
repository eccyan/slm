#include <gtest/gtest.h>
#include <slab/ring_buffer.hpp>
#include <thread>
#include <vector>

using namespace slm::slab;

TEST(SPSCRingBuffer, EmptyOnConstruction) {
    SPSCRingBuffer<uint32_t, 16> rb;
    uint32_t val;
    EXPECT_FALSE(rb.try_pop(val));
}

TEST(SPSCRingBuffer, PushAndPop) {
    SPSCRingBuffer<uint32_t, 16> rb;
    EXPECT_TRUE(rb.try_push(42));
    uint32_t val = 0;
    EXPECT_TRUE(rb.try_pop(val));
    EXPECT_EQ(val, 42u);
}

TEST(SPSCRingBuffer, FIFO_Order) {
    SPSCRingBuffer<uint32_t, 16> rb;
    for (uint32_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(rb.try_push(i));
    }
    for (uint32_t i = 0; i < 10; ++i) {
        uint32_t val = 999;
        EXPECT_TRUE(rb.try_pop(val));
        EXPECT_EQ(val, i);
    }
}

TEST(SPSCRingBuffer, FullReturnsFalse) {
    SPSCRingBuffer<uint32_t, 4> rb;
    EXPECT_TRUE(rb.try_push(1));
    EXPECT_TRUE(rb.try_push(2));
    EXPECT_TRUE(rb.try_push(3));
    // Capacity is 4, but usable slots = Capacity - 1 = 3 to distinguish full from empty
    EXPECT_FALSE(rb.try_push(4));
}

TEST(SPSCRingBuffer, EmptyAfterDrain) {
    SPSCRingBuffer<uint32_t, 8> rb;
    rb.try_push(1);
    rb.try_push(2);
    uint32_t v;
    rb.try_pop(v);
    rb.try_pop(v);
    EXPECT_FALSE(rb.try_pop(v));
}

TEST(SPSCRingBuffer, Wraparound) {
    SPSCRingBuffer<uint32_t, 4> rb;
    uint32_t v;

    for (int round = 0; round < 10; ++round) {
        EXPECT_TRUE(rb.try_push(round * 10 + 1));
        EXPECT_TRUE(rb.try_push(round * 10 + 2));
        EXPECT_TRUE(rb.try_push(round * 10 + 3));

        EXPECT_TRUE(rb.try_pop(v));
        EXPECT_EQ(v, static_cast<uint32_t>(round * 10 + 1));
        EXPECT_TRUE(rb.try_pop(v));
        EXPECT_EQ(v, static_cast<uint32_t>(round * 10 + 2));
        EXPECT_TRUE(rb.try_pop(v));
        EXPECT_EQ(v, static_cast<uint32_t>(round * 10 + 3));
    }
}

TEST(SPSCRingBuffer, Peek) {
    SPSCRingBuffer<uint32_t, 8> rb;
    EXPECT_FALSE(rb.peek());
    rb.try_push(99);
    EXPECT_TRUE(rb.peek());
    uint32_t v;
    rb.try_pop(v);
    EXPECT_FALSE(rb.peek());
}

TEST(SPSCRingBuffer, CacheLinePadding) {
    SPSCRingBuffer<uint32_t, 16> rb;
    auto base = reinterpret_cast<uintptr_t>(&rb);
    auto head_addr = reinterpret_cast<uintptr_t>(&rb.head_);
    auto tail_addr = reinterpret_cast<uintptr_t>(&rb.tail_);
    EXPECT_GE(std::abs(static_cast<long long>(tail_addr - head_addr)), 64);
}

TEST(SPSCRingBuffer, ConcurrentProducerConsumer) {
    constexpr int N = 100'000;
    SPSCRingBuffer<uint32_t, 256> rb;

    std::thread producer([&] {
        for (uint32_t i = 0; i < N; ++i) {
            while (!rb.try_push(i)) {
                // spin
            }
        }
    });

    std::vector<uint32_t> received;
    received.reserve(N);

    std::thread consumer([&] {
        uint32_t val;
        while (received.size() < N) {
            if (rb.try_pop(val)) {
                received.push_back(val);
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(received.size(), N);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(received[i], i) << "Mismatch at index " << i;
    }
}
