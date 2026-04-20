#pragma once

#include <atomic>
#include <cstdint>
#include <slab/ring_buffer.hpp>

namespace slm::slab {

/// Control block placed at the beginning of shared memory (offset 0).
/// Contains the free-list bitmask, command queue, and engine status.
///
/// Total size must fit within the 4KB control region.
struct ControlBlock {
    static constexpr uint32_t STATUS_IDLE = 0;
    static constexpr uint32_t STATUS_BUSY = 1;
    static constexpr uint32_t STATUS_SHUTDOWN = 2;

    /// Bitmask tracking free slabs. Bit i set = slab i is free.
    std::atomic<uint64_t> free_bitmask{0};

    /// Lock-free command queue. Producer: Python FUSE. Consumer: C++ engine.
    /// Each entry is a 32-bit handle (upper 8 = command, lower 24 = slab index).
    SPSCRingBuffer<uint32_t, 256> cmd_queue;

    /// Engine lifecycle status.
    std::atomic<uint32_t> engine_status{STATUS_IDLE};

    /// Number of slabs in the pool.
    uint32_t slab_count{0};

    /// Size of each slab in bytes.
    uint32_t slab_size{0};

    /// Initialize the control block for `count` slabs of `size` bytes each.
    void init(uint32_t count, uint32_t size = 65536) {
        slab_count = count;
        slab_size = size;
        engine_status.store(STATUS_IDLE, std::memory_order_relaxed);

        // Set all `count` bits to 1 (all slabs free)
        uint64_t mask = (count >= 64) ? ~uint64_t{0} : ((uint64_t{1} << count) - 1);
        free_bitmask.store(mask, std::memory_order_relaxed);
    }
};

} // namespace slm::slab
