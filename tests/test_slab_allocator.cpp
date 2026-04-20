#include <gtest/gtest.h>
#include <slab/slab_allocator.hpp>
#include <slab/header.hpp>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace slm::slab;

namespace {

constexpr uint32_t TEST_SLAB_COUNT = 4;
constexpr uint32_t TEST_SLAB_SIZE = 1024;  // small for testing
constexpr uint32_t TEST_CTRL_SIZE = 4096;
constexpr uint32_t TEST_SHM_SIZE = TEST_CTRL_SIZE + TEST_SLAB_COUNT * TEST_SLAB_SIZE;

struct TestFixture : public ::testing::Test {
    alignas(64) std::array<std::byte, TEST_SHM_SIZE> shm_buf{};
    std::unique_ptr<SlabAllocator> alloc;

    void SetUp() override {
        std::memset(shm_buf.data(), 0, shm_buf.size());
        alloc = std::make_unique<SlabAllocator>(
            shm_buf.data(), TEST_SLAB_COUNT, TEST_SLAB_SIZE, TEST_CTRL_SIZE
        );
    }
};

} // namespace

TEST_F(TestFixture, AcquireReturnsValidIndex) {
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());
    EXPECT_LT(*idx, TEST_SLAB_COUNT);
}

TEST_F(TestFixture, AcquireReturnsUniqueIndices) {
    std::set<uint32_t> indices;
    for (uint32_t i = 0; i < TEST_SLAB_COUNT; ++i) {
        auto idx = alloc->acquire();
        ASSERT_TRUE(idx.has_value()) << "Failed to acquire slab " << i;
        EXPECT_TRUE(indices.insert(*idx).second) << "Duplicate index " << *idx;
    }
}

TEST_F(TestFixture, ExhaustsPool) {
    for (uint32_t i = 0; i < TEST_SLAB_COUNT; ++i) {
        ASSERT_TRUE(alloc->acquire().has_value());
    }
    EXPECT_FALSE(alloc->acquire().has_value());
}

TEST_F(TestFixture, ReleaseAllowsReacquire) {
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());
    alloc->release(*idx);
    auto idx2 = alloc->acquire();
    ASSERT_TRUE(idx2.has_value());
    EXPECT_EQ(*idx, *idx2);
}

TEST_F(TestFixture, GetReturnsCorrectSize) {
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());
    auto span = alloc->get(*idx);
    EXPECT_EQ(span.size(), TEST_SLAB_SIZE);
}

TEST_F(TestFixture, GetReturnsNonOverlappingSpans) {
    auto i0 = alloc->acquire();
    auto i1 = alloc->acquire();
    ASSERT_TRUE(i0.has_value());
    ASSERT_TRUE(i1.has_value());

    auto s0 = alloc->get(*i0);
    auto s1 = alloc->get(*i1);

    auto p0 = reinterpret_cast<uintptr_t>(s0.data());
    auto p1 = reinterpret_cast<uintptr_t>(s1.data());
    if (p0 < p1) {
        EXPECT_GE(p1, p0 + s0.size());
    } else {
        EXPECT_GE(p0, p1 + s1.size());
    }
}

TEST_F(TestFixture, HeaderAccessible) {
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());

    auto span = alloc->get(*idx);
    auto* hdr = reinterpret_cast<MemoryFSHeader*>(span.data());
    hdr->magic = MEMFS_MAGIC;
    hdr->command = CMD_WRITE_COMMIT;
    hdr->vector_dim = 384;
    hdr->parent_id = 5;
    hdr->depth = 2;

    const auto& hdr_ref = alloc->header(*idx);
    EXPECT_EQ(hdr_ref.magic, MEMFS_MAGIC);
    EXPECT_EQ(hdr_ref.command, CMD_WRITE_COMMIT);
    EXPECT_EQ(hdr_ref.vector_dim, 384u);
    EXPECT_EQ(hdr_ref.parent_id, 5u);
    EXPECT_EQ(hdr_ref.depth, 2u);
}

TEST_F(TestFixture, ZeroCopyWriteAndRead) {
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());

    auto span = alloc->get(*idx);

    auto* hdr = reinterpret_cast<MemoryFSHeader*>(span.data());
    hdr->magic = MEMFS_MAGIC;
    hdr->command = CMD_WRITE_COMMIT;
    hdr->text_offset = 64;
    hdr->text_length = 5;
    hdr->vector_offset = 128;
    hdr->vector_dim = 3;

    std::memcpy(span.data() + 64, "hello", 5);

    float vec[] = {1.0f, 2.0f, 3.0f};
    std::memcpy(span.data() + 128, vec, sizeof(vec));

    const auto& h = alloc->header(*idx);
    auto text_span = span.subspan(h.text_offset, h.text_length);
    EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(text_span.data()),
                               text_span.size()), "hello");

    auto vec_span = span.subspan(h.vector_offset, h.vector_dim * sizeof(float));
    auto* fptr = reinterpret_cast<const float*>(vec_span.data());
    EXPECT_FLOAT_EQ(fptr[0], 1.0f);
    EXPECT_FLOAT_EQ(fptr[1], 2.0f);
    EXPECT_FLOAT_EQ(fptr[2], 3.0f);
}

TEST_F(TestFixture, CmdQueueAccess) {
    auto& queue = alloc->cmd_queue();
    EXPECT_FALSE(queue.peek());

    uint32_t handle = (CMD_WRITE_COMMIT << 24) | 3;
    EXPECT_TRUE(queue.try_push(handle));

    uint32_t popped = 0;
    EXPECT_TRUE(queue.try_pop(popped));
    EXPECT_EQ(popped >> 24, CMD_WRITE_COMMIT);
    EXPECT_EQ(popped & 0x00FFFFFF, 3u);
}

TEST_F(TestFixture, FullPipelineSimulation) {
    // === Simulate Python FUSE side ===

    // 1. Acquire a slab
    auto idx = alloc->acquire();
    ASSERT_TRUE(idx.has_value());

    // 2. Prepare payload
    const std::string text = "The proxy IP is 10.0.0.5";
    const std::vector<float> vec = {0.1f, 0.2f, 0.3f, 0.4f};

    uint32_t text_offset = 64;
    uint32_t text_length = static_cast<uint32_t>(text.size());
    uint32_t vector_offset = align_up(text_offset + text_length, 64);
    uint32_t vector_dim = static_cast<uint32_t>(vec.size());
    uint64_t total_size = vector_offset + vector_dim * sizeof(float);

    // 3. Write into slab via span
    auto span = alloc->get(*idx);
    auto* hdr = reinterpret_cast<MemoryFSHeader*>(span.data());
    *hdr = MemoryFSHeader{};
    hdr->magic = MEMFS_MAGIC;
    hdr->command = CMD_WRITE_COMMIT;
    hdr->total_size = total_size;
    hdr->text_offset = text_offset;
    hdr->text_length = text_length;
    hdr->vector_offset = vector_offset;
    hdr->vector_dim = vector_dim;
    hdr->parent_id = 7;
    hdr->depth = 2;

    std::memcpy(span.data() + text_offset, text.data(), text_length);
    std::memcpy(span.data() + vector_offset, vec.data(), vec.size() * sizeof(float));

    // 4. Push handle to command queue
    uint32_t handle = encode_handle(CMD_WRITE_COMMIT, *idx);
    EXPECT_TRUE(alloc->cmd_queue().try_push(handle));

    // === Simulate C++ Engine side ===

    // 5. Pop handle from command queue
    uint32_t popped = 0;
    EXPECT_TRUE(alloc->cmd_queue().try_pop(popped));

    uint8_t cmd = decode_command(popped);
    uint32_t slab_idx = decode_slab_index(popped);
    EXPECT_EQ(cmd, CMD_WRITE_COMMIT);
    EXPECT_EQ(slab_idx, *idx);

    // 6. Zero-copy read from slab
    const auto& h = alloc->header(slab_idx);
    EXPECT_EQ(h.magic, MEMFS_MAGIC);
    EXPECT_EQ(h.command, CMD_WRITE_COMMIT);
    EXPECT_EQ(h.parent_id, 7u);
    EXPECT_EQ(h.depth, 2u);

    auto engine_span = alloc->get(slab_idx);

    // Read text (zero-copy)
    std::string_view txt(
        reinterpret_cast<const char*>(engine_span.data() + h.text_offset),
        h.text_length
    );
    EXPECT_EQ(txt, "The proxy IP is 10.0.0.5");

    // Read vector (zero-copy) — verify 64-byte alignment
    EXPECT_EQ(h.vector_offset % 64, 0u);
    auto* fptr = reinterpret_cast<const float*>(engine_span.data() + h.vector_offset);
    std::span<const float> vec_view(fptr, h.vector_dim);
    EXPECT_EQ(vec_view.size(), 4u);
    EXPECT_FLOAT_EQ(vec_view[0], 0.1f);
    EXPECT_FLOAT_EQ(vec_view[1], 0.2f);
    EXPECT_FLOAT_EQ(vec_view[2], 0.3f);
    EXPECT_FLOAT_EQ(vec_view[3], 0.4f);

    // 7. Release slab
    alloc->release(slab_idx);

    // 8. Verify slab is reusable
    auto reacquired = alloc->acquire();
    ASSERT_TRUE(reacquired.has_value());
    EXPECT_EQ(*reacquired, slab_idx);
}
