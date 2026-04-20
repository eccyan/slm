#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <slab/control_block.hpp>
#include <slab/header.hpp>

namespace slm::slab {

/// Manages a pool of fixed-size memory slabs in a shared memory region.
///
/// Memory layout:
///   [0, ctrl_size)         ControlBlock
///   [ctrl_size, ...)       Slab[0], Slab[1], ..., Slab[slab_count-1]
///
/// Acquire/release are lock-free via atomic CAS on the ControlBlock bitmask.
/// get() returns a std::span directly into shared memory — zero copies.
class SlabAllocator {
public:
    SlabAllocator(void* shm_base, uint32_t slab_count,
                  uint32_t slab_size, uint32_t ctrl_size = 4096);

    std::optional<uint32_t> acquire();
    void release(uint32_t index);
    std::span<std::byte> get(uint32_t index);
    const MemoryFSHeader& header(uint32_t index);
    SPSCRingBuffer<uint32_t, 256>& cmd_queue();
    std::atomic<uint32_t>& engine_status();

private:
    std::byte* base_;
    ControlBlock* ctrl_;
    std::byte* data_base_;
    uint32_t slab_count_;
    uint32_t slab_size_;
};

} // namespace slm::slab
