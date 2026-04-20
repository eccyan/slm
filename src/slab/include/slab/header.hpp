#pragma once

#include <cstdint>

namespace slm::slab {

/// Magic bytes: 'MEMF' in little-endian
inline constexpr uint32_t MEMFS_MAGIC = 0x4D454D46;

/// Response magic: 'DONE' in little-endian
inline constexpr uint32_t MEMFS_DONE = 0x444F4E45;

/// Command types
inline constexpr uint8_t CMD_READ = 0x01;
inline constexpr uint8_t CMD_WRITE_COMMIT = 0x02;

/// Round `offset` up to the next multiple of `alignment`.
/// `alignment` must be a power of 2.
constexpr uint32_t align_up(uint32_t offset, uint32_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

/// Binary payload header at the start of every slab.
/// Exactly 64 bytes, 64-byte aligned for SIMD compatibility.
///
/// Layout within a slab:
///   [0..64)          MemoryFSHeader
///   [text_offset..)  UTF-8 text payload
///   [vector_offset..)  float32 array (64-byte aligned start)
struct alignas(64) MemoryFSHeader {
    uint32_t magic;           // MEMFS_MAGIC or MEMFS_DONE
    uint8_t  command;         // CMD_READ or CMD_WRITE_COMMIT
    uint8_t  padding[3];
    uint64_t total_size;      // total payload size in bytes
    uint32_t text_offset;     // byte offset to text (always 64)
    uint32_t text_length;     // text length in bytes
    uint32_t vector_offset;   // byte offset to vector (multiple of 64)
    uint32_t vector_dim;      // number of float32 elements
    uint32_t parent_id;       // scene graph parent (0 = root)
    uint8_t  depth;           // heading level (0 = preamble)
    uint8_t  reserved[27];
};

static_assert(sizeof(MemoryFSHeader) == 64, "MemoryFSHeader must be exactly 64 bytes");
static_assert(alignof(MemoryFSHeader) == 64, "MemoryFSHeader must be 64-byte aligned");

/// Encode a command and slab index into a 32-bit handle.
/// Upper 8 bits: command type. Lower 24 bits: slab index.
constexpr uint32_t encode_handle(uint8_t command, uint32_t slab_index) {
    return (static_cast<uint32_t>(command) << 24) | (slab_index & 0x00FFFFFF);
}

/// Extract the command type from a 32-bit handle.
constexpr uint8_t decode_command(uint32_t handle) {
    return static_cast<uint8_t>(handle >> 24);
}

/// Extract the slab index from a 32-bit handle.
constexpr uint32_t decode_slab_index(uint32_t handle) {
    return handle & 0x00FFFFFF;
}

} // namespace slm::slab
