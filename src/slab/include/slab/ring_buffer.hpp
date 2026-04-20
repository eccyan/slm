#pragma once

#include <array>
#include <atomic>
#include <cstddef>

namespace slm::slab {

/// Lock-free Single-Producer Single-Consumer ring buffer.
///
/// - `Capacity` must be a power of 2.
/// - Usable slots = Capacity - 1 (one slot reserved to distinguish full from empty).
/// - `head_` and `tail_` are on separate cache lines to prevent false sharing.
/// - Cached counters (`cached_tail_`, `cached_head_`) reduce atomic read frequency.
///
/// Memory ordering:
///   - Producer: `release` store on head_ (makes payload visible to consumer)
///   - Consumer: `acquire` load on head_ (sees producer's writes)
template <typename T, std::size_t Capacity>
    requires (Capacity > 0 && (Capacity & (Capacity - 1)) == 0)
struct alignas(64) SPSCRingBuffer {
    static constexpr std::size_t kMask = Capacity - 1;
    static constexpr std::size_t kMaxUsable = Capacity - 1;

    /// Writer index (only modified by producer).
    alignas(64) std::atomic<std::size_t> head_{0};
    /// Producer's local cache of tail_ to avoid frequent atomic reads.
    alignas(64) std::size_t cached_tail_{0};
    /// Reader index (only modified by consumer).
    alignas(64) std::atomic<std::size_t> tail_{0};
    /// Consumer's local cache of head_ to avoid frequent atomic reads.
    alignas(64) std::size_t cached_head_{0};
    /// Ring storage.
    alignas(64) std::array<T, Capacity> buffer_{};

    /// Push a value (producer side). Returns false if full.
    bool try_push(T value) {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto next_head = (head + 1) & kMask;

        // Check if full using cached tail
        if (next_head == cached_tail_) {
            // Refresh cache from the actual atomic tail
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if (next_head == cached_tail_) {
                return false; // genuinely full
            }
        }

        buffer_[head & kMask] = value;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /// Pop a value (consumer side). Returns false if empty.
    bool try_pop(T& value) {
        const auto tail = tail_.load(std::memory_order_relaxed);

        // Check if empty using cached head
        if (tail == cached_head_) {
            // Refresh cache from the actual atomic head
            cached_head_ = head_.load(std::memory_order_acquire);
            if (tail == cached_head_) {
                return false; // genuinely empty
            }
        }

        value = buffer_[tail & kMask];
        tail_.store((tail + 1) & kMask, std::memory_order_release);
        return true;
    }

    /// Check if there are items available without consuming.
    /// Note: result may be stale by the time caller acts on it.
    bool peek() const {
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);
        return tail != head;
    }
};

} // namespace slm::slab
