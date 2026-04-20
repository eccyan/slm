#include <gtest/gtest.h>
#include <slab/header.hpp>
#include <cstring>

using namespace slm::slab;

TEST(MemoryFSHeader, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(MemoryFSHeader), 64u);
}

TEST(MemoryFSHeader, AlignmentIs64Bytes) {
    EXPECT_EQ(alignof(MemoryFSHeader), 64u);
}

TEST(MemoryFSHeader, MagicConstant) {
    EXPECT_EQ(MEMFS_MAGIC, 0x4D454D46u);
}

TEST(MemoryFSHeader, CommandConstants) {
    EXPECT_EQ(CMD_READ, 0x01u);
    EXPECT_EQ(CMD_WRITE_COMMIT, 0x02u);
}

TEST(MemoryFSHeader, DoneResponseMagic) {
    EXPECT_EQ(MEMFS_DONE, 0x444F4E45u);
}

TEST(MemoryFSHeader, FieldOffsets) {
    MemoryFSHeader h{};
    auto base = reinterpret_cast<const char*>(&h);

    EXPECT_EQ(reinterpret_cast<const char*>(&h.magic) - base, 0);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.command) - base, 4);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.total_size) - base, 8);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.text_offset) - base, 16);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.text_length) - base, 20);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.vector_offset) - base, 24);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.vector_dim) - base, 28);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.parent_id) - base, 32);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.depth) - base, 36);
}

TEST(MemoryFSHeader, DefaultInitialization) {
    MemoryFSHeader h{};
    EXPECT_EQ(h.magic, 0u);
    EXPECT_EQ(h.command, 0u);
    EXPECT_EQ(h.total_size, 0u);
    EXPECT_EQ(h.text_offset, 0u);
    EXPECT_EQ(h.text_length, 0u);
    EXPECT_EQ(h.vector_offset, 0u);
    EXPECT_EQ(h.vector_dim, 0u);
    EXPECT_EQ(h.parent_id, 0u);
    EXPECT_EQ(h.depth, 0u);
}

TEST(MemoryFSHeader, AlignUp) {
    EXPECT_EQ(align_up(0, 64), 0u);
    EXPECT_EQ(align_up(1, 64), 64u);
    EXPECT_EQ(align_up(63, 64), 64u);
    EXPECT_EQ(align_up(64, 64), 64u);
    EXPECT_EQ(align_up(65, 64), 128u);
    EXPECT_EQ(align_up(100, 64), 128u);
}

TEST(MemoryFSHeader, VectorOffsetAlignment) {
    uint32_t text_end = 64 + 50;
    uint32_t vec_off = align_up(text_end, 64);
    EXPECT_EQ(vec_off, 128u);
    EXPECT_EQ(vec_off % 64, 0u);
}

#include <slab/control_block.hpp>

TEST(ControlBlock, FitsInOnePage) {
    // ControlBlock must fit within the 4KB control region
    EXPECT_LE(sizeof(ControlBlock), 4096u);
}

TEST(ControlBlock, FreeBitmaskInitiallyAllFree) {
    ControlBlock cb{};
    cb.init(63);  // 63 slabs
    // All 63 bits set = all slabs free
    uint64_t expected = (1ULL << 63) - 1;
    EXPECT_EQ(cb.free_bitmask.load(), expected);
}

TEST(ControlBlock, EngineStatusDefault) {
    ControlBlock cb{};
    cb.init(63);
    EXPECT_EQ(cb.engine_status.load(), ControlBlock::STATUS_IDLE);
}

TEST(ControlBlock, SlabCountAndSize) {
    ControlBlock cb{};
    cb.init(63, 65536);
    EXPECT_EQ(cb.slab_count, 63u);
    EXPECT_EQ(cb.slab_size, 65536u);
}

TEST(HandleEncoding, EncodeWriteCommit) {
    uint32_t handle = slm::slab::encode_handle(CMD_WRITE_COMMIT, 42);
    EXPECT_EQ(handle, (0x02u << 24) | 42u);
}

TEST(HandleEncoding, EncodeRead) {
    uint32_t handle = slm::slab::encode_handle(CMD_READ, 0);
    EXPECT_EQ(handle, (0x01u << 24));
}

TEST(HandleEncoding, DecodeCommand) {
    uint32_t handle = slm::slab::encode_handle(CMD_WRITE_COMMIT, 100);
    EXPECT_EQ(slm::slab::decode_command(handle), CMD_WRITE_COMMIT);
}

TEST(HandleEncoding, DecodeSlabIndex) {
    uint32_t handle = slm::slab::encode_handle(CMD_READ, 12345);
    EXPECT_EQ(slm::slab::decode_slab_index(handle), 12345u);
}

TEST(HandleEncoding, RoundTrip) {
    for (uint8_t cmd : {CMD_READ, CMD_WRITE_COMMIT}) {
        for (uint32_t idx : {0u, 1u, 62u, 0x00FFFFFFu}) {
            uint32_t h = slm::slab::encode_handle(cmd, idx);
            EXPECT_EQ(slm::slab::decode_command(h), cmd);
            EXPECT_EQ(slm::slab::decode_slab_index(h), idx);
        }
    }
}

TEST(HandleEncoding, MaxSlabIndex) {
    uint32_t h = slm::slab::encode_handle(CMD_READ, 0x00FFFFFF);
    EXPECT_EQ(slm::slab::decode_slab_index(h), 0x00FFFFFFu);
}
