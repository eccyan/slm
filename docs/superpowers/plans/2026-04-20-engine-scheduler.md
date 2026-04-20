# Engine Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-tier game-engine-style scheduler that orchestrates I/O command processing, cohomology consistency checks, and Langevin drift simulation — plus the main.cpp binary entry point with SIGTERM graceful shutdown.

**Architecture:** The Scheduler class owns no data — it references all components (SlabAllocator, MemoryGraph, FisherRaoMetric, CoboundaryOperator, LangevinStepper, Store) via references passed at construction. Its `run(std::stop_token)` method implements the three-tier frame loop: Tier 1 drains the SPSC queue (READ/WRITE_COMMIT), Tier 2 processes pending cohomology checks with a time budget, Tier 3 fires the Langevin SDE tick at fixed intervals. The main.cpp binary creates POSIX shared memory, constructs all components, and runs the scheduler in a `std::jthread` with SIGTERM handling.

**Tech Stack:** C++23, `std::jthread`/`std::stop_token`, POSIX shared memory (`shm_open`/`mmap`), all project libraries

---

## File Map

| File | Responsibility |
|---|---|
| `src/engine/include/engine/scheduler.hpp` | Scheduler class: Config, constructor, run() |
| `src/engine/src/scheduler.cpp` | Three-tier frame loop implementation |
| `src/engine/src/main.cpp` | Binary entry point: shared memory setup, component wiring, SIGTERM |
| `src/engine/CMakeLists.txt` | Modify: add scheduler.cpp, main.cpp, link all libraries |
| `tests/test_scheduler.cpp` | Scheduler unit tests (using local memory buffer, not real shm) |
| `tests/CMakeLists.txt` | Modify: add scheduler tests to engine_tests |

---

### Task 1: Scheduler Class — Tier 1 WRITE_COMMIT

**Files:**
- Create: `src/engine/include/engine/scheduler.hpp`
- Create: `src/engine/src/scheduler.cpp`
- Modify: `src/engine/CMakeLists.txt` (add scheduler.cpp, link slab/sheaf/persist)
- Create: `tests/test_scheduler.cpp`
- Modify: `tests/CMakeLists.txt` (add test_scheduler.cpp to engine_tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scheduler.cpp`:

```cpp
#include <gtest/gtest.h>
#include <engine/scheduler.hpp>
#include <engine/memory_graph.hpp>
#include <slab/slab_allocator.hpp>
#include <slab/header.hpp>
#include <metric/fisher_rao.hpp>
#include <sheaf/coboundary.hpp>
#include <langevin/sde_stepper.hpp>
#include <persist/sqlite_store.hpp>
#include <cstring>
#include <filesystem>
#include <thread>

using namespace slm::engine;
using namespace slm::slab;
using namespace slm::metric;
using namespace slm::sheaf;
using namespace slm::langevin;
using namespace slm::persist;

namespace {

constexpr uint32_t TEST_SLAB_COUNT = 8;
constexpr uint32_t TEST_SLAB_SIZE = 4096;
constexpr uint32_t TEST_CTRL_SIZE = 4096;
constexpr uint32_t TEST_SHM_SIZE = TEST_CTRL_SIZE + TEST_SLAB_COUNT * TEST_SLAB_SIZE;

struct SchedulerFixture : public ::testing::Test {
    alignas(64) std::array<std::byte, TEST_SHM_SIZE> shm_buf{};
    std::unique_ptr<SlabAllocator> slab;
    MemoryGraph graph;
    FisherRaoMetric metric;
    CoboundaryOperator sheaf;
    LangevinStepper langevin{{.dt = 5.0f, .lambda_decay = 0.01f,
                               .noise_scale = 0.0f, .archive_threshold = 0.95f}};
    std::filesystem::path db_path;
    std::unique_ptr<SqliteStore> store;

    void SetUp() override {
        std::memset(shm_buf.data(), 0, shm_buf.size());
        slab = std::make_unique<SlabAllocator>(
            shm_buf.data(), TEST_SLAB_COUNT, TEST_SLAB_SIZE, TEST_CTRL_SIZE
        );
        db_path = std::filesystem::temp_directory_path() / "test_scheduler.db";
        std::filesystem::remove(db_path);
        store = std::make_unique<SqliteStore>(db_path);
    }

    void TearDown() override {
        store.reset();
        std::filesystem::remove(db_path);
        std::filesystem::remove(db_path.string() + "-wal");
        std::filesystem::remove(db_path.string() + "-shm");
    }

    // Helper: write a WRITE_COMMIT payload into a slab and push the handle
    void submit_write(const std::string& text, uint32_t parent_id = 0,
                      uint8_t depth = 0) {
        auto idx = slab->acquire();
        ASSERT_TRUE(idx.has_value());

        auto span = slab->get(*idx);
        auto* hdr = reinterpret_cast<MemoryFSHeader*>(span.data());
        *hdr = MemoryFSHeader{};
        hdr->magic = MEMFS_MAGIC;
        hdr->command = CMD_WRITE_COMMIT;
        hdr->text_offset = 64;
        hdr->text_length = static_cast<uint32_t>(text.size());
        hdr->parent_id = parent_id;
        hdr->depth = depth;

        // Simple fake vector: 4 floats
        hdr->vector_offset = align_up(64 + hdr->text_length, 64);
        hdr->vector_dim = 4;
        float fake_vec[] = {1.0f, 0.0f, 0.0f, 0.0f};

        std::memcpy(span.data() + hdr->text_offset, text.data(), text.size());
        std::memcpy(span.data() + hdr->vector_offset, fake_vec, sizeof(fake_vec));

        hdr->total_size = hdr->vector_offset + hdr->vector_dim * sizeof(float);

        auto handle = encode_handle(CMD_WRITE_COMMIT, *idx);
        ASSERT_TRUE(slab->cmd_queue().try_push(handle));
    }
};

} // namespace

TEST_F(SchedulerFixture, WriteCommitInsertsNode) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("Hello from the agent");

    // Run one frame
    std::stop_source ss;
    std::jthread t([&](std::stop_token token) {
        scheduler.run(token);
    });

    // Give it time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ss.request_stop();
    t.request_stop();
    t.join();

    // Node should have been inserted into the graph
    EXPECT_EQ(graph.size(), 1u);

    auto ids = graph.all_ids();
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(graph.text(ids[0]), "Hello from the agent");
    EXPECT_EQ(graph.parent_id(ids[0]), 0u);

    // Node should be at the center of the Poincaré disk
    EXPECT_FLOAT_EQ(graph.state(ids[0]).pos.x, 0.0f);
    EXPECT_FLOAT_EQ(graph.state(ids[0]).pos.y, 0.0f);
}

TEST_F(SchedulerFixture, MultipleWriteCommits) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("First memory");
    submit_write("Second memory");
    submit_write("Third memory");

    std::jthread t([&](std::stop_token token) {
        scheduler.run(token);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t.request_stop();
    t.join();

    EXPECT_EQ(graph.size(), 3u);
}

TEST_F(SchedulerFixture, WriteCommitWithParentId) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("Parent node", 0, 0);

    std::jthread t([&](std::stop_token token) {
        scheduler.run(token);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t.request_stop();
    t.join();

    ASSERT_EQ(graph.size(), 1u);
    auto parent_id = graph.all_ids()[0];

    // Submit a child
    submit_write("Child node", parent_id, 1);

    std::jthread t2([&](std::stop_token token) {
        scheduler.run(token);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t2.request_stop();
    t2.join();

    ASSERT_EQ(graph.size(), 2u);
    auto siblings = graph.siblings(parent_id);
    EXPECT_EQ(siblings.size(), 1u);
}

TEST_F(SchedulerFixture, GracefulShutdownFlushes) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("Persisted on shutdown");

    std::jthread t([&](std::stop_token token) {
        scheduler.run(token);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t.request_stop();
    t.join();

    // After shutdown, graph should have been flushed to SQLite
    MemoryGraph loaded;
    store->load(loaded);
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded.text(loaded.all_ids()[0]), "Persisted on shutdown");
}
```

- [ ] **Step 2: Modify src/engine/CMakeLists.txt**

Replace `src/engine/CMakeLists.txt`:

```cmake
add_library(engine STATIC
    src/memory_graph.cpp
    src/scheduler.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC metric langevin slab sheaf persist)

target_compile_features(engine PUBLIC cxx_std_23)
```

- [ ] **Step 3: Add test_scheduler.cpp to tests/CMakeLists.txt**

Modify the existing `engine_tests` executable in `tests/CMakeLists.txt` to include the new test file:

```cmake
add_executable(engine_tests
    test_memory_graph.cpp
    test_scheduler.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest_main
)

gtest_discover_tests(engine_tests)
```

- [ ] **Step 4: Implement Scheduler header**

Create `src/engine/include/engine/scheduler.hpp`:

```cpp
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <random>
#include <stop_token>
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

/// Three-tier game-engine-style scheduler.
///
/// Tier 1 (Frame-Critical): Drain SPSC queue — READ/WRITE_COMMIT
/// Tier 2 (Asynchronous): Cohomology consistency checks (time-budgeted)
/// Tier 3 (Fixed Tick): Langevin drift simulation
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
        Config config = {}
    );

    /// Main loop — runs until stop token is triggered.
    void run(std::stop_token token);

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

    std::chrono::steady_clock::time_point last_tier3_tick_;
    std::chrono::steady_clock::time_point last_checkpoint_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace slm::engine
```

- [ ] **Step 5: Implement Scheduler**

Create `src/engine/src/scheduler.cpp`:

```cpp
#include <engine/scheduler.hpp>
#include <slab/header.hpp>
#include <metric/gaussian_node.hpp>
#include <cstring>
#include <string>
#include <string_view>

namespace slm::engine {

Scheduler::Scheduler(
    slab::SlabAllocator& slab,
    slab::SPSCRingBuffer<uint32_t, 256>& queue,
    MemoryGraph& graph,
    metric::FisherRaoMetric& metric,
    sheaf::CoboundaryOperator& sheaf,
    langevin::LangevinStepper& langevin,
    persist::Store& persist,
    Config config
)
    : slab_(slab), queue_(queue), graph_(graph), metric_(metric),
      sheaf_(sheaf), langevin_(langevin), persist_(persist), config_(config)
{}

double Scheduler::current_time() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

void Scheduler::run(std::stop_token token) {
    start_time_ = std::chrono::steady_clock::now();
    last_tier3_tick_ = start_time_;
    last_checkpoint_ = start_time_;

    while (!token.stop_requested()) {
        // Tier 1: Always drain the queue first
        process_tier1();

        auto now = std::chrono::steady_clock::now();

        // Tier 2: Amortized cohomology (time-budgeted)
        if (!cohomology_pending_.empty()) {
            process_tier2();
        }

        // Tier 3: Fixed-tick Langevin
        now = std::chrono::steady_clock::now();
        if (now - last_tier3_tick_ >= config_.tier3_tick_interval) {
            process_tier3();
            last_tier3_tick_ = now;
        }

        // Periodic checkpoint
        now = std::chrono::steady_clock::now();
        if (now - last_checkpoint_ >= config_.checkpoint_interval) {
            checkpoint();
            last_checkpoint_ = now;
        }

        // Backpressure: sleep only if all tiers are idle
        if (cohomology_pending_.empty() && !queue_.peek()) {
            std::this_thread::sleep_for(config_.tier1_poll_interval);
        }
    }

    // Graceful shutdown: drain + flush
    process_tier1();
    persist_.flush(graph_);
}

void Scheduler::process_tier1() {
    uint32_t handle;
    while (queue_.try_pop(handle)) {
        uint8_t cmd = slab::decode_command(handle);
        uint32_t slab_idx = slab::decode_slab_index(handle);

        switch (cmd) {
        case slab::CMD_WRITE_COMMIT:
            handle_write_commit(slab_idx);
            break;
        case slab::CMD_READ:
            handle_read(slab_idx);
            break;
        }

        slab_.release(slab_idx);
    }
}

void Scheduler::handle_write_commit(uint32_t slab_idx) {
    auto span = slab_.get(slab_idx);
    const auto& hdr = slab_.header(slab_idx);

    // Extract text
    std::string text(
        reinterpret_cast<const char*>(span.data() + hdr.text_offset),
        hdr.text_length
    );

    // Extract vector (zero-copy view for now, copy into owned storage)
    const float* vec_ptr = reinterpret_cast<const float*>(
        span.data() + hdr.vector_offset);
    std::vector<float> mu(vec_ptr, vec_ptr + hdr.vector_dim);

    // Initial sigma = SIGMA_MAX (fresh node)
    std::vector<float> sigma(hdr.vector_dim, metric::SIGMA_MAX);

    // Insert at Poincaré center
    langevin::NodeState state{};
    state.pos = {0.0f, 0.0f};
    state.last_access_time = current_time();
    state.access_count = 0;

    auto node_id = graph_.insert(
        std::move(mu), std::move(sigma), std::move(text),
        hdr.parent_id, hdr.depth, state
    );

    // Enqueue for Tier 2 cohomology check
    cohomology_pending_.push_back(node_id);
}

void Scheduler::handle_read(uint32_t slab_idx) {
    auto span = slab_.get(slab_idx);
    const auto& hdr = slab_.header(slab_idx);

    // Build result Markdown from active nodes (r < active_radius)
    std::string result;
    for (auto id : graph_.all_ids()) {
        if (graph_.state(id).pos.radius() < config_.active_radius) {
            result += graph_.text(id);
            result += "\n";
            const auto& ann = graph_.annotation(id);
            if (!ann.empty()) {
                result += ann;
                result += "\n";
            }
        }
    }

    // If a query vector was provided, also do top_k search
    if (hdr.vector_dim > 0) {
        const float* query_ptr = reinterpret_cast<const float*>(
            span.data() + hdr.vector_offset);

        // Build GaussianNodes for all graph nodes
        std::vector<metric::GaussianNode> candidates;
        std::vector<uint32_t> candidate_ids;
        for (auto id : graph_.all_ids()) {
            candidates.push_back({graph_.mu(id), graph_.sigma(id),
                                  graph_.state(id).access_count});
            candidate_ids.push_back(id);
        }

        if (!candidates.empty()) {
            std::vector<float> query_sigma(hdr.vector_dim, metric::SIGMA_MAX);
            metric::GaussianNode query{
                std::span<const float>(query_ptr, hdr.vector_dim),
                query_sigma, 0
            };

            auto top_indices = metric_.top_k(query, candidates,
                                              config_.search_top_k);

            for (auto idx : top_indices) {
                auto id = candidate_ids[idx];
                // Activate retrieved nodes
                langevin::LangevinStepper::activate(graph_.state(id),
                                                     current_time());
                graph_.state(id).access_count++;

                // Add to result if not already included
                if (graph_.state(id).pos.radius() >= config_.active_radius) {
                    result += graph_.text(id);
                    result += "\n";
                }
            }
        }
    }

    // Write result back to slab
    auto* resp_hdr = reinterpret_cast<slab::MemoryFSHeader*>(span.data());
    resp_hdr->magic = slab::MEMFS_DONE;
    resp_hdr->text_offset = 64;
    resp_hdr->text_length = static_cast<uint32_t>(result.size());

    uint32_t max_text = slab_.get(slab_idx).size() - 64;
    uint32_t copy_len = std::min(resp_hdr->text_length, max_text);
    std::memcpy(span.data() + 64, result.data(), copy_len);
    resp_hdr->text_length = copy_len;
}

void Scheduler::process_tier2() {
    auto deadline = std::chrono::steady_clock::now() + config_.tier2_time_budget;

    while (!cohomology_pending_.empty()
           && std::chrono::steady_clock::now() < deadline) {
        auto node_id = cohomology_pending_.front();
        cohomology_pending_.pop_front();

        // Skip if node was removed
        if (!graph_.contains(node_id)) continue;

        // Build neighborhood from MemoryGraph
        sheaf::Neighborhood hood;
        hood.new_node_mu = graph_.mu(node_id);
        hood.new_node_text = graph_.text(node_id);

        // Primary: structural siblings
        auto sibs = graph_.siblings(graph_.parent_id(node_id));
        for (auto sib_id : sibs) {
            if (sib_id == node_id) continue;

            uint32_t neighbor_idx = static_cast<uint32_t>(
                hood.neighbor_mus.size());
            auto sib_mu = graph_.mu(sib_id);
            hood.neighbor_mus.emplace_back(sib_mu.begin(), sib_mu.end());
            hood.neighbor_texts.push_back(graph_.text(sib_id));

            // Zero relation for same-topic siblings
            std::vector<float> zero_rel(sib_mu.size(), 0.0f);
            hood.edges.push_back({neighbor_idx,
                                  sheaf::EdgeType::Structural,
                                  std::move(zero_rel)});
        }

        auto result = sheaf_.compute_local(hood,
                                            config_.contradiction_threshold);

        if (result.norm > config_.contradiction_threshold) {
            // Apply drift penalty to conflicting nodes
            for (auto neighbor_idx : result.conflicting) {
                // Map neighbor_idx back to graph ID
                auto& sibs_list = sibs;
                uint32_t actual_idx = 0;
                for (auto sib_id : sibs_list) {
                    if (sib_id == node_id) continue;
                    if (actual_idx == neighbor_idx) {
                        graph_.state(sib_id).pos = {0.0f, 0.93f};
                        break;
                    }
                    actual_idx++;
                }
            }

            // Attach annotation to new node
            sheaf::Annotation ann;
            if (!result.conflicting.empty()) {
                uint32_t first_conflict = result.conflicting[0];
                if (first_conflict < hood.neighbor_texts.size()) {
                    ann.superseded_text = hood.neighbor_texts[first_conflict];
                }
            }
            ann.superseding_text = graph_.text(node_id);
            ann.delta_norm = result.norm;
            graph_.set_annotation(node_id, sheaf::format_annotation(ann));
        }

        // Yield to Tier 1 if new commands arrived
        if (queue_.peek()) break;
    }
}

void Scheduler::process_tier3() {
    auto archived = langevin_.step(graph_.all_states(), current_time(), rng_);

    // Archive and remove in reverse order to avoid index invalidation
    // (swap-and-pop can shift indices)
    std::vector<uint32_t> archive_ids;
    auto all_ids = graph_.all_ids();
    for (auto state_idx : archived) {
        if (state_idx < all_ids.size()) {
            archive_ids.push_back(all_ids[state_idx]);
        }
    }

    for (auto id : archive_ids) {
        auto snap = graph_.snapshot(id);
        persist_.archive_node(snap);
        graph_.remove(id);
    }
}

void Scheduler::checkpoint() {
    persist_.checkpoint(graph_);
}

} // namespace slm::engine
```

- [ ] **Step 6: Build and run tests**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && cd build && ctest --output-on-failure -R engine
```

Expected: All engine tests PASS (11 MemoryGraph + 4 Scheduler).

- [ ] **Step 7: Commit**

```bash
git add src/engine/ tests/test_scheduler.cpp tests/CMakeLists.txt
git commit -m "feat(engine): add three-tier Scheduler with WRITE_COMMIT, cohomology, and Langevin"
```

---

### Task 2: main.cpp — Engine Binary Entry Point

**Files:**
- Create: `src/engine/src/main.cpp`
- Modify: `src/engine/CMakeLists.txt` (add memory_engine executable)

- [ ] **Step 1: Create main.cpp**

Create `src/engine/src/main.cpp`:

```cpp
#include <engine/scheduler.hpp>
#include <engine/memory_graph.hpp>
#include <slab/slab_allocator.hpp>
#include <metric/fisher_rao.hpp>
#include <sheaf/coboundary.hpp>
#include <langevin/sde_stepper.hpp>
#include <persist/sqlite_store.hpp>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

// Global jthread pointer for signal handler
static std::jthread* g_engine_thread = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_engine_thread) {
        g_engine_thread->request_stop();
    }
}

int main(int argc, char* argv[]) {
    // Configuration defaults
    std::string shm_name = "superlocal_shm";
    std::filesystem::path db_path = ".superlocal/memory.db";
    uint32_t shm_size = 4 * 1024 * 1024;  // 4MB
    uint32_t slab_size = 64 * 1024;        // 64KB
    uint32_t ctrl_size = 4096;

    // Parse CLI args (simple key=value)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.starts_with("--shm-name=")) {
            shm_name = arg.substr(11);
        } else if (arg.starts_with("--db-path=")) {
            db_path = arg.substr(10);
        }
    }

    // Ensure DB directory exists
    db_path.parent_path().empty() ||
        std::filesystem::create_directories(db_path.parent_path());

    uint32_t slab_count = (shm_size - ctrl_size) / slab_size;

    std::cout << "Superlocal Engine starting...\n"
              << "  shm_name:   " << shm_name << "\n"
              << "  db_path:    " << db_path << "\n"
              << "  slab_count: " << slab_count << "\n"
              << "  slab_size:  " << slab_size << "\n";

    // Create POSIX shared memory
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        std::cerr << "Failed to create shared memory: " << shm_name << "\n";
        return 1;
    }

    if (ftruncate(shm_fd, shm_size) < 0) {
        std::cerr << "Failed to resize shared memory\n";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        return 1;
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap shared memory\n";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        return 1;
    }

    // Construct components
    slm::slab::SlabAllocator slab(shm_ptr, slab_count, slab_size, ctrl_size);
    slm::engine::MemoryGraph graph;
    slm::metric::FisherRaoMetric metric;
    slm::sheaf::CoboundaryOperator sheaf;
    slm::langevin::LangevinStepper langevin({
        .dt = 5.0f,
        .lambda_decay = 0.01f,
        .noise_scale = 0.001f,
        .archive_threshold = 0.95f,
    });
    slm::persist::SqliteStore store(db_path);

    // Load existing graph from SQLite
    store.load(graph);
    std::cout << "Loaded " << graph.size() << " nodes from " << db_path << "\n";

    // Wire up scheduler
    slm::engine::Scheduler scheduler(
        slab, slab.cmd_queue(), graph, metric, sheaf, langevin, store
    );

    // Set up SIGTERM handler
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    // Run in jthread
    std::jthread engine_thread([&](std::stop_token token) {
        scheduler.run(token);
    });
    g_engine_thread = &engine_thread;

    std::cout << "Engine running. PID=" << getpid() << "\n";

    // Block until the thread finishes (SIGTERM triggers request_stop)
    engine_thread.join();

    // Cleanup shared memory
    munmap(shm_ptr, shm_size);
    close(shm_fd);
    shm_unlink(shm_name.c_str());

    std::cout << "Engine stopped.\n";
    return 0;
}
```

- [ ] **Step 2: Add memory_engine executable to CMakeLists.txt**

Append to `src/engine/CMakeLists.txt`:

```cmake

# Engine binary
add_executable(memory_engine
    src/main.cpp
)

target_link_libraries(memory_engine PRIVATE engine)
```

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build
```

Expected: `build/src/engine/memory_engine` binary is produced.

- [ ] **Step 4: Verify the binary runs and exits cleanly**

Run:
```bash
cd build && timeout 2 src/engine/memory_engine --db-path=/tmp/test_engine.db 2>&1 || true
```

Expected: Prints startup message and exits after timeout (or Ctrl+C).

- [ ] **Step 5: Commit**

```bash
git add src/engine/src/main.cpp src/engine/CMakeLists.txt
git commit -m "feat(engine): add memory_engine binary with POSIX shm, SIGTERM handling, and CLI args"
```

---

### Task 3: systemd Service File

**Files:**
- Create: `config/superlocal-engine.service`

- [ ] **Step 1: Create the systemd unit file**

Create `config/superlocal-engine.service`:

```ini
[Unit]
Description=Superlocal Memory C++ Backend Engine
Before=superlocal-fuse.service

[Service]
Type=simple
ExecStart=%h/.local/bin/memory_engine --db-path=%h/.superlocal/memory.db --shm-name=superlocal_shm
Restart=on-failure
LimitMEMLOCK=infinity
KillSignal=SIGTERM
TimeoutStopSec=10

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Commit**

```bash
git add config/superlocal-engine.service
git commit -m "feat(engine): add systemd user service configuration"
```

---

## Summary

After completing all 3 tasks, the engine provides:

| Component | What it does |
|---|---|
| `Scheduler` | Three-tier frame loop: Tier 1 I/O drain, Tier 2 cohomology, Tier 3 Langevin |
| `handle_write_commit()` | Parses slab payload, inserts node at r=0, enqueues for cohomology |
| `handle_read()` | Gathers active nodes, runs top_k search, writes Markdown result to slab |
| `process_tier2()` | Builds Neighborhood from MemoryGraph, runs CoboundaryOperator, applies drift penalties and annotations |
| `process_tier3()` | Runs LangevinStepper::step(), archives nodes past threshold |
| `main.cpp` | POSIX shared memory setup, component wiring, SIGTERM → graceful shutdown with flush |
| `superlocal-engine.service` | systemd user service for daemon management |
