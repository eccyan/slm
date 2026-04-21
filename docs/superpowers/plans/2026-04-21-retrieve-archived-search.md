# Retrieve Archived Nodes via Cold-Storage Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable search (`handle_read`) to find and reactivate archived nodes from SQLite, so that memories beyond the Poincaré boundary are dormant rather than invisible.

**Architecture:** Add `retrieve_archived()` to the `Store` interface. `SqliteStore` implements it by loading all archived mu/sigma BLOBs, scoring them with `FisherRaoMetric::top_k`, and returning the top-k snapshots. `Scheduler::handle_read` merges active + archived candidates, re-inserts matched archived nodes into `MemoryGraph` at the Poincaré center, and deletes them from the archived set in SQLite.

**Tech Stack:** C++23, SQLite3, Google Test

---

### Task 1: Add `retrieve_archived` to `Store` interface

**Files:**
- Modify: `src/persist/include/persist/store.hpp`

- [ ] **Step 1: Add the virtual method declaration**

In `src/persist/include/persist/store.hpp`, add below the `archive_node` declaration:

```cpp
virtual std::vector<engine::MemoryGraph::NodeSnapshot> retrieve_archived(
    const metric::GaussianNode& query,
    const metric::FisherRaoMetric& metric,
    uint32_t k
) = 0;

virtual void reactivate_node(uint32_t id) = 0;
```

Also add the required include at the top:

```cpp
#include <metric/fisher_rao.hpp>
```

- [ ] **Step 2: Verify it compiles (expect failure — pure virtual not yet implemented)**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | head -30`
Expected: Compile errors in `SqliteStore` about unimplemented pure virtual methods.

- [ ] **Step 3: Commit**

```bash
git add src/persist/include/persist/store.hpp
git commit -m "feat(persist): add retrieve_archived and reactivate_node to Store interface"
```

---

### Task 2: Implement `retrieve_archived` in `SqliteStore`

**Files:**
- Modify: `src/persist/include/persist/sqlite_store.hpp`
- Modify: `src/persist/src/sqlite_store.cpp`

- [ ] **Step 1: Add method declarations to the header**

In `src/persist/include/persist/sqlite_store.hpp`, add inside the `public` section after `archive_node`:

```cpp
std::vector<engine::MemoryGraph::NodeSnapshot> retrieve_archived(
    const metric::GaussianNode& query,
    const metric::FisherRaoMetric& metric,
    uint32_t k
) override;

void reactivate_node(uint32_t id) override;
```

- [ ] **Step 2: Implement `retrieve_archived` in sqlite_store.cpp**

Add at the end of `sqlite_store.cpp` (before the closing namespace brace):

```cpp
std::vector<engine::MemoryGraph::NodeSnapshot> SqliteStore::retrieve_archived(
    const metric::GaussianNode& query,
    const metric::FisherRaoMetric& metric,
    uint32_t k
) {
    // Load all archived node mu/sigma/metadata from SQLite
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "SELECT id, parent_id, depth, text, mu, sigma, access_count, "
        "       pos_x, pos_y, last_access, annotation "
        "FROM memory_nodes WHERE status = 1",
        -1, &stmt, nullptr);

    std::vector<engine::MemoryGraph::NodeSnapshot> snapshots;
    std::vector<metric::GaussianNode> candidates;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        engine::MemoryGraph::NodeSnapshot snap;
        snap.id = static_cast<uint32_t>(sqlite3_column_int(stmt, 0));
        snap.parent_id = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));
        snap.depth = static_cast<uint8_t>(sqlite3_column_int(stmt, 2));

        const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        snap.text = text ? text : "";

        const float* mu_data = static_cast<const float*>(sqlite3_column_blob(stmt, 4));
        int mu_bytes = sqlite3_column_bytes(stmt, 4);
        int mu_count = mu_bytes / static_cast<int>(sizeof(float));
        snap.mu.assign(mu_data, mu_data + mu_count);

        const float* sigma_data = static_cast<const float*>(sqlite3_column_blob(stmt, 5));
        int sigma_bytes = sqlite3_column_bytes(stmt, 5);
        int sigma_count = sigma_bytes / static_cast<int>(sizeof(float));
        snap.sigma.assign(sigma_data, sigma_data + sigma_count);

        snap.access_count = static_cast<uint32_t>(sqlite3_column_int(stmt, 6));
        snap.pos_x = static_cast<float>(sqlite3_column_double(stmt, 7));
        snap.pos_y = static_cast<float>(sqlite3_column_double(stmt, 8));
        snap.last_access = sqlite3_column_double(stmt, 9);

        const char* ann = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        snap.annotation = ann ? ann : "";

        candidates.push_back({snap.mu, snap.sigma, snap.access_count});
        snapshots.push_back(std::move(snap));
    }

    sqlite3_finalize(stmt);

    if (candidates.empty()) {
        return {};
    }

    auto top_indices = metric.top_k(query, candidates, k);

    std::vector<engine::MemoryGraph::NodeSnapshot> results;
    results.reserve(top_indices.size());
    for (auto idx : top_indices) {
        results.push_back(std::move(snapshots[idx]));
    }
    return results;
}

void SqliteStore::reactivate_node(uint32_t id) {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "DELETE FROM memory_nodes WHERE id = ? AND status = 1",
        -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, static_cast<int>(id));
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}
```

- [ ] **Step 3: Verify compilation succeeds**

Run: `cmake --build build -j$(sysctl -n hw.ncpu)`
Expected: Clean build with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/persist/include/persist/sqlite_store.hpp src/persist/src/sqlite_store.cpp
git commit -m "feat(persist): implement retrieve_archived and reactivate_node in SqliteStore"
```

---

### Task 3: Write tests for `retrieve_archived`

**Files:**
- Modify: `tests/test_persist.cpp`

- [ ] **Step 1: Write the test for retrieve_archived**

Add the following test at the end of `tests/test_persist.cpp`:

```cpp
TEST_F(PersistFixture, RetrieveArchivedFindsMatchingNodes) {
    using namespace slm::metric;

    MemoryGraph graph;
    // Insert 3 nodes with distinct mu vectors
    std::vector<float> mu_a = {1.0f, 0.0f, 0.0f};
    std::vector<float> mu_b = {0.0f, 1.0f, 0.0f};
    std::vector<float> mu_c = {0.9f, 0.1f, 0.0f};  // close to mu_a
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f};

    auto id_a = graph.insert(mu_a, sigma, "Node A", 0, 0,
                              NodeState{.pos = {0.96f, 0.0f}});
    auto id_b = graph.insert(mu_b, sigma, "Node B", 0, 0,
                              NodeState{.pos = {0.97f, 0.0f}});
    auto id_c = graph.insert(mu_c, sigma, "Node C", 0, 0,
                              NodeState{.pos = {0.98f, 0.0f}});

    SqliteStore store(db_path);

    // Archive all 3 nodes
    store.archive_node(graph.snapshot(id_a));
    store.archive_node(graph.snapshot(id_b));
    store.archive_node(graph.snapshot(id_c));

    // Query with a vector close to mu_a
    std::vector<float> query_mu = {0.95f, 0.05f, 0.0f};
    std::vector<float> query_sigma(3, SIGMA_MAX);
    GaussianNode query{query_mu, query_sigma, 0};
    FisherRaoMetric metric;

    auto results = store.retrieve_archived(query, metric, 2);

    ASSERT_EQ(results.size(), 2u);
    // Top-2 should be Node A and Node C (closest to query)
    bool found_a = false, found_c = false;
    for (const auto& snap : results) {
        if (snap.text == "Node A") found_a = true;
        if (snap.text == "Node C") found_c = true;
    }
    EXPECT_TRUE(found_a) << "Node A should be in top-2";
    EXPECT_TRUE(found_c) << "Node C should be in top-2";
}

TEST_F(PersistFixture, RetrieveArchivedEmptyArchive) {
    using namespace slm::metric;

    SqliteStore store(db_path);

    std::vector<float> query_mu = {1.0f, 0.0f, 0.0f};
    std::vector<float> query_sigma(3, SIGMA_MAX);
    GaussianNode query{query_mu, query_sigma, 0};
    FisherRaoMetric metric;

    auto results = store.retrieve_archived(query, metric, 5);
    EXPECT_TRUE(results.empty());
}

TEST_F(PersistFixture, ReactivateNodeRemovesFromArchive) {
    using namespace slm::metric;

    MemoryGraph graph;
    std::vector<float> mu = {1.0f, 0.0f, 0.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f};
    auto id = graph.insert(mu, sigma, "To reactivate", 0, 0,
                            NodeState{.pos = {0.96f, 0.0f}});

    SqliteStore store(db_path);
    store.archive_node(graph.snapshot(id));

    // Verify it's in the archive
    GaussianNode query{mu, std::vector<float>(3, SIGMA_MAX), 0};
    FisherRaoMetric metric;
    auto before = store.retrieve_archived(query, metric, 10);
    ASSERT_EQ(before.size(), 1u);

    // Reactivate it
    store.reactivate_node(id);

    // Verify it's gone from the archive
    auto after = store.retrieve_archived(query, metric, 10);
    EXPECT_TRUE(after.empty());
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) && cd build && ctest -R persist --output-on-failure`
Expected: All persist tests pass, including the 3 new ones.

- [ ] **Step 3: Commit**

```bash
git add tests/test_persist.cpp
git commit -m "test(persist): add tests for retrieve_archived and reactivate_node"
```

---

### Task 4: Update `handle_read` to search archived nodes

**Files:**
- Modify: `src/engine/src/scheduler.cpp`

- [ ] **Step 1: Write the failing scheduler integration test**

Add to `tests/test_scheduler.cpp`:

```cpp
TEST_F(SchedulerFixture, SearchFindsArchivedNodes) {
    // Pre-populate an archived node directly in SQLite
    MemoryGraph temp_graph;
    std::vector<float> arch_mu = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> arch_sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    auto arch_id = temp_graph.insert(arch_mu, arch_sigma, "Archived memory", 0, 0,
                                      NodeState{.pos = {0.96f, 0.0f}, .access_count = 3});
    store->archive_node(temp_graph.snapshot(arch_id));

    // No active nodes in the graph — search must hit the archive
    ASSERT_EQ(graph.size(), 0u);

    Scheduler::Config cfg{};
    cfg.search_top_k = 5;
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, cfg);

    // Submit a search read with the same embedding
    auto idx = slab->acquire();
    ASSERT_TRUE(idx.has_value());

    auto span = slab->get(*idx);
    auto* hdr = reinterpret_cast<MemoryFSHeader*>(span.data());
    std::memset(hdr, 0, sizeof(MemoryFSHeader));
    hdr->magic = MEMFS_MAGIC;
    hdr->command = CMD_READ;
    hdr->text_offset = 64;
    hdr->text_length = 0;
    hdr->vector_offset = 64;
    hdr->vector_dim = 4;

    float query_vec[] = {1.0f, 0.0f, 0.0f, 0.0f};
    std::memcpy(span.data() + hdr->vector_offset, query_vec, sizeof(query_vec));

    auto handle = encode_handle(CMD_READ, *idx);
    ASSERT_TRUE(slab->cmd_queue().try_push(handle));

    std::thread t([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    scheduler.request_stop();
    t.join();

    // The archived node should have been reactivated into the graph
    EXPECT_GE(graph.size(), 1u);

    // Check the response slab contains the archived text
    auto resp_span = slab->get(*idx);
    auto* resp_hdr = reinterpret_cast<const MemoryFSHeader*>(resp_span.data());
    EXPECT_EQ(resp_hdr->magic, MEMFS_DONE);
    EXPECT_GT(resp_hdr->text_length, 0u);

    std::string response(
        reinterpret_cast<const char*>(resp_span.data() + resp_hdr->text_offset),
        resp_hdr->text_length);
    EXPECT_NE(response.find("Archived memory"), std::string::npos)
        << "Response was: " << response;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) && cd build && ctest -R engine --output-on-failure`
Expected: `SearchFindsArchivedNodes` FAILS — `graph.size()` is 0, response is empty.

- [ ] **Step 3: Update `handle_read` to merge active + archived results**

Replace the `handle_read` method in `src/engine/src/scheduler.cpp` with:

```cpp
void Scheduler::handle_read(uint32_t slab_idx) {
    // Search read: top-k retrieval from active graph + archived cold storage.
    auto span = slab_.get(slab_idx);
    const auto& hdr = slab_.header(slab_idx);

    std::string result;

    if (hdr.vector_dim > 0) {
        const float* query_ptr = reinterpret_cast<const float*>(
            span.data() + hdr.vector_offset);

        // Phase 1: Search active nodes in-memory
        std::vector<metric::GaussianNode> candidates;
        std::vector<uint32_t> candidate_ids;
        for (auto id : graph_.all_ids()) {
            candidates.push_back({graph_.mu(id), graph_.sigma(id),
                                  graph_.state(id).access_count});
            candidate_ids.push_back(id);
        }

        std::vector<uint32_t> active_top;
        if (!candidates.empty()) {
            std::vector<float> query_sigma(hdr.vector_dim, metric::SIGMA_MAX);
            metric::GaussianNode query{
                std::span<const float>(query_ptr, hdr.vector_dim),
                query_sigma, 0
            };
            active_top = metric_.top_k(query, candidates,
                                        config_.search_top_k);
        }

        // Phase 2: Search archived nodes via SQLite cold storage
        std::vector<float> query_sigma(hdr.vector_dim, metric::SIGMA_MAX);
        metric::GaussianNode query{
            std::span<const float>(query_ptr, hdr.vector_dim),
            query_sigma, 0
        };
        auto archived_hits = persist_.retrieve_archived(
            query, metric_, config_.search_top_k);

        // Phase 3: Collect active results
        struct ScoredResult {
            float distance;
            std::string text;
            std::string annotation;
            uint32_t active_id;       // non-zero if from active graph
            size_t archived_idx;      // index into archived_hits if from archive
            bool is_archived;
        };
        std::vector<ScoredResult> scored;

        for (auto idx : active_top) {
            auto id = candidate_ids[idx];
            float dist = metric_.distance(query, candidates[idx]);
            scored.push_back({dist, graph_.text(id), graph_.annotation(id),
                              id, 0, false});
        }

        // Phase 4: Score archived results against the same query
        for (size_t i = 0; i < archived_hits.size(); ++i) {
            const auto& snap = archived_hits[i];
            metric::GaussianNode arch_node{snap.mu, snap.sigma, snap.access_count};
            float dist = metric_.distance(query, arch_node);
            scored.push_back({dist, snap.text, snap.annotation,
                              0, i, true});
        }

        // Phase 5: Sort by distance, take top-k
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.distance < b.distance;
                  });

        uint32_t limit = std::min(static_cast<uint32_t>(scored.size()),
                                   config_.search_top_k);

        for (uint32_t i = 0; i < limit; ++i) {
            const auto& hit = scored[i];

            result += hit.text;
            result += "\n";
            if (!hit.annotation.empty()) {
                result += hit.annotation;
                result += "\n";
            }

            if (hit.is_archived) {
                // Reactivate: re-insert into graph at Poincaré center
                auto& snap = archived_hits[hit.archived_idx];
                snap.pos_x = 0.0f;
                snap.pos_y = 0.0f;
                snap.access_count += 1;
                snap.last_access = current_time();
                graph_.insert_from_snapshot(snap);
                persist_.reactivate_node(snap.id);
            } else {
                // Activate in-memory node: pull to center
                langevin::LangevinStepper::activate(
                    graph_.state(hit.active_id), current_time());
            }
        }
    }

    write_response(slab_idx, span, result);
}
```

Also add the required include at the top of `scheduler.cpp`:

```cpp
#include <algorithm>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) && cd build && ctest -R "engine|persist" --output-on-failure`
Expected: All engine and persist tests pass, including `SearchFindsArchivedNodes`.

- [ ] **Step 5: Commit**

```bash
git add src/engine/src/scheduler.cpp tests/test_scheduler.cpp
git commit -m "feat(engine): search archived nodes via cold-storage retrieval

handle_read now queries both the active MemoryGraph and SQLite archive,
merges results by Fisher-Rao distance, and reactivates matched archived
nodes at the Poincaré center."
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Rebuild and run full test suite**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) && cd build && ctest --output-on-failure`
Expected: All tests pass (slab, metric, langevin, sheaf, engine, persist).

- [ ] **Step 2: Live test with the running engine**

Restart the engine (or wait for checkpoint), then test search retrieval:

```bash
# Search for something that should be in the archive
cat ~/.agent_memory/search/vulkan_game.md
```

Expected: Returns relevant archived memories, not just the single active node.

- [ ] **Step 3: Verify reactivation via analyze**

```bash
.venv/bin/python -m slmfs analyze
```

Expected: Active node count should increase after the search pulled archived nodes back to the center.

- [ ] **Step 4: Commit any remaining changes and push**

```bash
git push
```
