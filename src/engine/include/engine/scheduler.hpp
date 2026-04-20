#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <random>
#include <atomic>
#include <thread>
#include <engine/memory_graph.hpp>
#include <metric/fisher_rao.hpp>
#include <sheaf/coboundary.hpp>
#include <sheaf/annotation.hpp>
#include <sheaf/neighborhood.hpp>
#include <langevin/sde_stepper.hpp>
#include <slab/slab_allocator.hpp>
#include <persist/store.hpp>

namespace slm::engine {

class Scheduler {
public:
    struct Config {
        std::chrono::microseconds tier1_poll_interval{100};
        std::chrono::milliseconds tier2_time_budget{50};
        std::chrono::seconds      tier3_tick_interval{5};
        std::chrono::seconds      checkpoint_interval{60};
        float contradiction_threshold{0.5f};
        uint32_t search_top_k{10};
        float active_radius{0.3f};
    };

    Scheduler(
        slab::SlabAllocator& slab,
        slab::SPSCRingBuffer<uint32_t, 256>& queue,
        MemoryGraph& graph,
        metric::FisherRaoMetric& metric,
        sheaf::CoboundaryOperator& sheaf,
        langevin::LangevinStepper& langevin,
        persist::Store& persist,
        Config config
    );

    /// Main loop — runs until request_stop() is called.
    void run();

    /// Request cooperative stop (thread-safe).
    void request_stop() { stop_requested_.store(true, std::memory_order_release); }

private:
    void process_tier1();
    void process_tier2();
    void process_tier3();
    void checkpoint();

    void handle_write_commit(uint32_t slab_idx);
    void handle_read(uint32_t slab_idx);

    double current_time() const;

    slab::SlabAllocator& slab_;
    slab::SPSCRingBuffer<uint32_t, 256>& queue_;
    MemoryGraph& graph_;
    metric::FisherRaoMetric& metric_;
    sheaf::CoboundaryOperator& sheaf_;
    langevin::LangevinStepper& langevin_;
    persist::Store& persist_;
    Config config_;

    std::deque<uint32_t> cohomology_pending_;
    std::mt19937 rng_{42};

    std::atomic<bool> stop_requested_{false};

    std::chrono::steady_clock::time_point last_tier3_tick_;
    std::chrono::steady_clock::time_point last_checkpoint_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace slm::engine
