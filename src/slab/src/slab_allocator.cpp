#include <slab/slab_allocator.hpp>
#include <cassert>
#include <bit>

namespace slm::slab {

SlabAllocator::SlabAllocator(void* shm_base, uint32_t slab_count,
                             uint32_t slab_size, uint32_t ctrl_size)
    : base_(static_cast<std::byte*>(shm_base))
    , ctrl_(reinterpret_cast<ControlBlock*>(base_))
    , data_base_(base_ + ctrl_size)
    , slab_count_(slab_count)
    , slab_size_(slab_size)
{
    assert(slab_count <= 64 && "Bitmask supports at most 64 slabs");
    ctrl_->init(slab_count, slab_size);
}

std::optional<uint32_t> SlabAllocator::acquire() {
    while (true) {
        uint64_t current = ctrl_->free_bitmask.load(std::memory_order_relaxed);
        if (current == 0) {
            return std::nullopt;
        }

        uint32_t idx = static_cast<uint32_t>(std::countr_zero(current));

        uint64_t new_val = current & ~(uint64_t{1} << idx);
        if (ctrl_->free_bitmask.compare_exchange_weak(
                current, new_val,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return idx;
        }
    }
}

void SlabAllocator::release(uint32_t index) {
    assert(index < slab_count_);
    ctrl_->free_bitmask.fetch_or(uint64_t{1} << index, std::memory_order_release);
}

std::span<std::byte> SlabAllocator::get(uint32_t index) {
    assert(index < slab_count_);
    return {data_base_ + static_cast<std::size_t>(index) * slab_size_, slab_size_};
}

const MemoryFSHeader& SlabAllocator::header(uint32_t index) {
    auto span = get(index);
    return *reinterpret_cast<const MemoryFSHeader*>(span.data());
}

SPSCRingBuffer<uint32_t, 256>& SlabAllocator::cmd_queue() {
    return ctrl_->cmd_queue;
}

std::atomic<uint32_t>& SlabAllocator::engine_status() {
    return ctrl_->engine_status;
}

} // namespace slm::slab
