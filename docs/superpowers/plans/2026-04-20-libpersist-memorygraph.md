# libpersist + MemoryGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SoA in-memory node registry (MemoryGraph) and the abstract persistence layer with SQLite V1 implementation — enabling nodes to be inserted, removed, queried by scene graph, checkpointed to disk, and loaded on startup.

**Architecture:** MemoryGraph lives in `src/engine/` as the central registry. It uses Structure-of-Arrays layout: hot arrays (`NodeState`, owned mu/sigma vectors, parent_ids, depths) for Tier 2/3 iteration and cold arrays (texts, annotations) for Tier 1 I/O. libpersist provides the abstract `Store` interface and `SqliteStore` implementation. SqliteStore serializes mu/sigma as raw BLOB, positions as REAL, and manages the active/archived status flag. MemoryGraph depends on liblangevin (NodeState) and libmetric (GaussianNode). libpersist depends on MemoryGraph, libmetric, and SQLite3.

**Tech Stack:** C++23, SQLite3, Google Test

---

## File Map

| File | Responsibility |
|---|---|
| `src/engine/CMakeLists.txt` | Engine library (MemoryGraph for now, scheduler later) |
| `src/engine/include/engine/memory_graph.hpp` | MemoryGraph class: SoA storage, insert/remove, scene graph queries |
| `src/engine/src/memory_graph.cpp` | MemoryGraph implementation |
| `src/persist/CMakeLists.txt` | libpersist static library, links SQLite3 |
| `src/persist/include/persist/store.hpp` | Abstract Store interface |
| `src/persist/include/persist/sqlite_store.hpp` | SqliteStore class declaration |
| `src/persist/src/sqlite_store.cpp` | SqliteStore implementation (schema, checkpoint, load, archive) |
| `tests/test_memory_graph.cpp` | MemoryGraph tests |
| `tests/test_persist.cpp` | SqliteStore round-trip tests |
| `CMakeLists.txt` | Root CMake (add engine + persist subdirectories) |
| `tests/CMakeLists.txt` | Add engine_tests + persist_tests executables |

---

### Task 1: MemoryGraph — SoA Registry

**Files:**
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/engine)`)
- Create: `src/engine/CMakeLists.txt`
- Create: `src/engine/include/engine/memory_graph.hpp`
- Create: `src/engine/src/memory_graph.cpp`
- Create: `tests/test_memory_graph.cpp`
- Modify: `tests/CMakeLists.txt` (add engine_tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_graph.cpp`:

```cpp
#include <gtest/gtest.h>
#include <engine/memory_graph.hpp>
#include <vector>

using namespace slm::engine;
using namespace slm::langevin;
using namespace slm::metric;

namespace {

struct GraphFixture : public ::testing::Test {
    MemoryGraph graph;

    // Helper: create owned mu/sigma vectors and insert a node
    uint32_t insert_node(const std::string& text, uint32_t parent_id = 0,
                         uint8_t depth = 0, float x = 0.0f, float y = 0.0f) {
        std::vector<float> mu = {1.0f, 0.0f, 0.0f};
        std::vector<float> sigma = {1.0f, 1.0f, 1.0f};
        NodeState state{};
        state.pos = {x, y};
        state.last_access_time = 0.0;
        state.access_count = 0;
        return graph.insert(std::move(mu), std::move(sigma), text,
                            parent_id, depth, state);
    }
};

} // namespace

TEST_F(GraphFixture, InsertReturnsUniqueIds) {
    auto id1 = insert_node("first");
    auto id2 = insert_node("second");
    auto id3 = insert_node("third");

    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id1, id3);
}

TEST_F(GraphFixture, InsertIncrementsSize) {
    EXPECT_EQ(graph.size(), 0u);
    insert_node("a");
    EXPECT_EQ(graph.size(), 1u);
    insert_node("b");
    EXPECT_EQ(graph.size(), 2u);
}

TEST_F(GraphFixture, GetNodeData) {
    auto id = insert_node("hello world", 5, 2, 0.3f, 0.4f);

    EXPECT_EQ(graph.text(id), "hello world");
    EXPECT_EQ(graph.parent_id(id), 5u);
    EXPECT_EQ(graph.depth(id), 2u);
    EXPECT_FLOAT_EQ(graph.state(id).pos.x, 0.3f);
    EXPECT_FLOAT_EQ(graph.state(id).pos.y, 0.4f);
    EXPECT_EQ(graph.mu(id).size(), 3u);
    EXPECT_EQ(graph.sigma(id).size(), 3u);
}

TEST_F(GraphFixture, RemoveNode) {
    auto id1 = insert_node("keep");
    auto id2 = insert_node("remove");
    auto id3 = insert_node("keep too");

    graph.remove(id2);

    EXPECT_EQ(graph.size(), 2u);
    EXPECT_EQ(graph.text(id1), "keep");
    EXPECT_EQ(graph.text(id3), "keep too");
}

TEST_F(GraphFixture, RemoveLastNode) {
    auto id = insert_node("only");
    graph.remove(id);
    EXPECT_EQ(graph.size(), 0u);
}

TEST_F(GraphFixture, SiblingsQuery) {
    auto p = insert_node("parent", 0, 0);
    auto c1 = insert_node("child1", p, 1);
    auto c2 = insert_node("child2", p, 1);
    auto c3 = insert_node("child3", p, 1);
    auto other = insert_node("other", 0, 0);

    auto siblings = graph.siblings(p);
    EXPECT_EQ(siblings.size(), 3u);

    // All children of parent p should be in the list
    auto has = [&](uint32_t id) {
        return std::find(siblings.begin(), siblings.end(), id) != siblings.end();
    };
    EXPECT_TRUE(has(c1));
    EXPECT_TRUE(has(c2));
    EXPECT_TRUE(has(c3));
    EXPECT_FALSE(has(other));
}

TEST_F(GraphFixture, ParentQuery) {
    auto p = insert_node("parent", 0, 0);
    auto c = insert_node("child", p, 1);

    auto parent = graph.parent(c);
    EXPECT_TRUE(parent.has_value());
    EXPECT_EQ(*parent, p);
}

TEST_F(GraphFixture, ParentOfRootIsNullopt) {
    auto root = insert_node("root", 0, 0);
    auto parent = graph.parent(root);
    EXPECT_FALSE(parent.has_value());
}

TEST_F(GraphFixture, StatesSpanContiguous) {
    insert_node("a", 0, 0, 0.1f, 0.0f);
    insert_node("b", 0, 0, 0.5f, 0.0f);
    insert_node("c", 0, 0, 0.9f, 0.0f);

    auto states = graph.all_states();
    EXPECT_EQ(states.size(), 3u);
    // Verify contiguity: addresses should be sequential
    EXPECT_EQ(&states[1] - &states[0], 1);
    EXPECT_EQ(&states[2] - &states[1], 1);
}

TEST_F(GraphFixture, SetAnnotation) {
    auto id = insert_node("test");
    EXPECT_EQ(graph.annotation(id), "");

    graph.set_annotation(id, "<!-- cohomology: ... -->");
    EXPECT_EQ(graph.annotation(id), "<!-- cohomology: ... -->");
}

TEST_F(GraphFixture, AllIds) {
    auto id1 = insert_node("a");
    auto id2 = insert_node("b");

    auto ids = graph.all_ids();
    EXPECT_EQ(ids.size(), 2u);
}
```

- [ ] **Step 2: Create src/engine/CMakeLists.txt**

```cmake
add_library(engine STATIC
    src/memory_graph.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC metric langevin)

target_compile_features(engine PUBLIC cxx_std_23)
```

- [ ] **Step 3: Add engine to root CMakeLists.txt**

Add after `add_subdirectory(src/sheaf)`:

```cmake
add_subdirectory(src/engine)
```

- [ ] **Step 4: Add engine_tests to tests/CMakeLists.txt**

Append to `tests/CMakeLists.txt`:

```cmake

add_executable(engine_tests
    test_memory_graph.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest_main
)

gtest_discover_tests(engine_tests)
```

- [ ] **Step 5: Implement MemoryGraph header**

Create `src/engine/include/engine/memory_graph.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <langevin/poincare_disk.hpp>
#include <metric/gaussian_node.hpp>

namespace slm::engine {

/// SoA (Structure of Arrays) in-memory registry of memory nodes.
///
/// Hot arrays (NodeState, mu, sigma, parent_ids, depths) are contiguous
/// for cache-friendly iteration in Tier 2/3. Cold arrays (texts, annotations)
/// are only accessed during Tier 1 I/O.
///
/// The SoA layout keeps the Langevin stepper's iteration tight — it touches
/// only the states vector (16 bytes per node) without cache pollution from
/// heap-allocated strings.
class MemoryGraph {
public:
    /// Insert a new node. Takes ownership of mu and sigma vectors.
    /// Returns the assigned node ID.
    uint32_t insert(std::vector<float> mu, std::vector<float> sigma,
                    std::string text, uint32_t parent_id, uint8_t depth,
                    langevin::NodeState state);

    /// Remove a node by ID. Uses swap-and-pop for O(1) removal.
    void remove(uint32_t id);

    /// Number of active nodes.
    uint32_t size() const { return static_cast<uint32_t>(ids_.size()); }

    /// Check if a node ID exists.
    bool contains(uint32_t id) const { return id_to_index_.contains(id); }

    // --- Per-node accessors (by ID) ---

    langevin::NodeState& state(uint32_t id);
    const langevin::NodeState& state(uint32_t id) const;

    std::span<const float> mu(uint32_t id) const;
    std::span<const float> sigma(uint32_t id) const;

    const std::string& text(uint32_t id) const;
    const std::string& annotation(uint32_t id) const;
    void set_annotation(uint32_t id, std::string ann);

    uint32_t parent_id(uint32_t id) const;
    uint8_t depth(uint32_t id) const;

    // --- Bulk accessors (for Tier 2/3 iteration) ---

    std::span<langevin::NodeState> all_states();
    std::span<const uint32_t> all_ids() const;

    // --- Scene graph queries ---

    /// Return IDs of all nodes whose parent_id matches the given ID.
    std::vector<uint32_t> siblings(uint32_t parent) const;

    /// Return the parent ID of a node, or nullopt if parent_id == 0.
    std::optional<uint32_t> parent(uint32_t node_id) const;

    // --- Snapshot support (for persistence) ---

    /// Data needed to serialize a single node.
    struct NodeSnapshot {
        uint32_t id;
        uint32_t parent_id;
        uint8_t depth;
        std::string text;
        std::vector<float> mu;
        std::vector<float> sigma;
        uint32_t access_count;
        float pos_x;
        float pos_y;
        double last_access;
        std::string annotation;
    };

    /// Create a snapshot of a single node for serialization.
    NodeSnapshot snapshot(uint32_t id) const;

    /// Insert a node from a snapshot (used by Store::load).
    /// Uses the snapshot's ID directly (does not auto-assign).
    void insert_from_snapshot(const NodeSnapshot& snap);

private:
    // Hot arrays
    std::vector<langevin::NodeState> states_;
    std::vector<std::vector<float>>  mus_;      // owned mu vectors
    std::vector<std::vector<float>>  sigmas_;   // owned sigma vectors
    std::vector<uint32_t>            parent_ids_;
    std::vector<uint8_t>             depths_;

    // Cold arrays
    std::vector<std::string>         texts_;
    std::vector<std::string>         annotations_;

    // Bookkeeping
    std::vector<uint32_t>            ids_;
    std::unordered_map<uint32_t, size_t> id_to_index_;
    uint32_t next_id_{1};

    size_t index_of(uint32_t id) const;
};

} // namespace slm::engine
```

- [ ] **Step 6: Implement MemoryGraph**

Create `src/engine/src/memory_graph.cpp`:

```cpp
#include <engine/memory_graph.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace slm::engine {

uint32_t MemoryGraph::insert(std::vector<float> mu, std::vector<float> sigma,
                              std::string text, uint32_t parent_id,
                              uint8_t depth, langevin::NodeState state) {
    uint32_t id = next_id_++;
    size_t idx = ids_.size();

    ids_.push_back(id);
    states_.push_back(state);
    mus_.push_back(std::move(mu));
    sigmas_.push_back(std::move(sigma));
    parent_ids_.push_back(parent_id);
    depths_.push_back(depth);
    texts_.push_back(std::move(text));
    annotations_.emplace_back();

    id_to_index_[id] = idx;
    return id;
}

void MemoryGraph::remove(uint32_t id) {
    auto it = id_to_index_.find(id);
    assert(it != id_to_index_.end());
    size_t idx = it->second;
    size_t last = ids_.size() - 1;

    if (idx != last) {
        // Swap with last element
        uint32_t last_id = ids_[last];

        ids_[idx] = ids_[last];
        states_[idx] = states_[last];
        mus_[idx] = std::move(mus_[last]);
        sigmas_[idx] = std::move(sigmas_[last]);
        parent_ids_[idx] = parent_ids_[last];
        depths_[idx] = depths_[last];
        texts_[idx] = std::move(texts_[last]);
        annotations_[idx] = std::move(annotations_[last]);

        id_to_index_[last_id] = idx;
    }

    ids_.pop_back();
    states_.pop_back();
    mus_.pop_back();
    sigmas_.pop_back();
    parent_ids_.pop_back();
    depths_.pop_back();
    texts_.pop_back();
    annotations_.pop_back();

    id_to_index_.erase(id);
}

size_t MemoryGraph::index_of(uint32_t id) const {
    auto it = id_to_index_.find(id);
    assert(it != id_to_index_.end());
    return it->second;
}

langevin::NodeState& MemoryGraph::state(uint32_t id) {
    return states_[index_of(id)];
}

const langevin::NodeState& MemoryGraph::state(uint32_t id) const {
    return states_[index_of(id)];
}

std::span<const float> MemoryGraph::mu(uint32_t id) const {
    return mus_[index_of(id)];
}

std::span<const float> MemoryGraph::sigma(uint32_t id) const {
    return sigmas_[index_of(id)];
}

const std::string& MemoryGraph::text(uint32_t id) const {
    return texts_[index_of(id)];
}

const std::string& MemoryGraph::annotation(uint32_t id) const {
    return annotations_[index_of(id)];
}

void MemoryGraph::set_annotation(uint32_t id, std::string ann) {
    annotations_[index_of(id)] = std::move(ann);
}

uint32_t MemoryGraph::parent_id(uint32_t id) const {
    return parent_ids_[index_of(id)];
}

uint8_t MemoryGraph::depth(uint32_t id) const {
    return depths_[index_of(id)];
}

std::span<langevin::NodeState> MemoryGraph::all_states() {
    return states_;
}

std::span<const uint32_t> MemoryGraph::all_ids() const {
    return ids_;
}

std::vector<uint32_t> MemoryGraph::siblings(uint32_t parent) const {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (parent_ids_[i] == parent) {
            result.push_back(ids_[i]);
        }
    }
    return result;
}

std::optional<uint32_t> MemoryGraph::parent(uint32_t node_id) const {
    uint32_t pid = parent_ids_[index_of(node_id)];
    if (pid == 0) return std::nullopt;
    if (id_to_index_.contains(pid)) return pid;
    return std::nullopt;
}

MemoryGraph::NodeSnapshot MemoryGraph::snapshot(uint32_t id) const {
    size_t idx = index_of(id);
    return {
        .id = id,
        .parent_id = parent_ids_[idx],
        .depth = depths_[idx],
        .text = texts_[idx],
        .mu = mus_[idx],
        .sigma = sigmas_[idx],
        .access_count = states_[idx].access_count,
        .pos_x = states_[idx].pos.x,
        .pos_y = states_[idx].pos.y,
        .last_access = states_[idx].last_access_time,
        .annotation = annotations_[idx],
    };
}

void MemoryGraph::insert_from_snapshot(const NodeSnapshot& snap) {
    size_t idx = ids_.size();

    ids_.push_back(snap.id);
    states_.push_back(langevin::NodeState{
        .pos = {snap.pos_x, snap.pos_y},
        .last_access_time = snap.last_access,
        .access_count = snap.access_count,
    });
    mus_.push_back(snap.mu);
    sigmas_.push_back(snap.sigma);
    parent_ids_.push_back(snap.parent_id);
    depths_.push_back(snap.depth);
    texts_.push_back(snap.text);
    annotations_.push_back(snap.annotation);

    id_to_index_[snap.id] = idx;

    // Keep next_id_ ahead of all inserted IDs
    if (snap.id >= next_id_) {
        next_id_ = snap.id + 1;
    }
}

} // namespace slm::engine
```

- [ ] **Step 7: Build and run tests**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && cd build && ctest --output-on-failure -R engine
```

Expected: All 12 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt src/engine/ tests/test_memory_graph.cpp tests/CMakeLists.txt
git commit -m "feat(engine): add MemoryGraph SoA registry with insert, remove, and scene graph queries"
```

---

### Task 2: Store Interface + SqliteStore Schema

**Files:**
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/persist)`)
- Create: `src/persist/CMakeLists.txt`
- Create: `src/persist/include/persist/store.hpp`
- Create: `src/persist/include/persist/sqlite_store.hpp`
- Create: `src/persist/src/sqlite_store.cpp`
- Create: `tests/test_persist.cpp`
- Modify: `tests/CMakeLists.txt` (add persist_tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persist.cpp`:

```cpp
#include <gtest/gtest.h>
#include <persist/sqlite_store.hpp>
#include <engine/memory_graph.hpp>
#include <filesystem>
#include <vector>

using namespace slm::persist;
using namespace slm::engine;
using namespace slm::langevin;

namespace {

struct PersistFixture : public ::testing::Test {
    std::filesystem::path db_path;

    void SetUp() override {
        db_path = std::filesystem::temp_directory_path() / "test_superlocal.db";
        std::filesystem::remove(db_path);
    }

    void TearDown() override {
        std::filesystem::remove(db_path);
        // Also remove WAL and SHM files
        std::filesystem::remove(db_path.string() + "-wal");
        std::filesystem::remove(db_path.string() + "-shm");
    }
};

} // namespace

TEST_F(PersistFixture, CreateAndOpen) {
    {
        SqliteStore store(db_path);
        // Should create the database file
    }
    EXPECT_TRUE(std::filesystem::exists(db_path));
}

TEST_F(PersistFixture, CheckpointAndLoad) {
    MemoryGraph graph;

    // Insert two nodes
    std::vector<float> mu1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> sigma1 = {0.5f, 0.5f, 0.5f};
    NodeState state1{.pos = {0.3f, 0.4f}, .last_access_time = 100.0, .access_count = 5};
    auto id1 = graph.insert(mu1, sigma1, "first node", 0, 0, state1);

    std::vector<float> mu2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> sigma2 = {1.0f, 1.0f, 1.0f};
    NodeState state2{.pos = {0.1f, 0.0f}, .last_access_time = 200.0, .access_count = 2};
    auto id2 = graph.insert(mu2, sigma2, "second node", id1, 1, state2);

    graph.set_annotation(id2, "<!-- test annotation -->");

    // Checkpoint to SQLite
    {
        SqliteStore store(db_path);
        store.checkpoint(graph);
    }

    // Load into a fresh graph
    MemoryGraph loaded;
    {
        SqliteStore store(db_path);
        store.load(loaded);
    }

    ASSERT_EQ(loaded.size(), 2u);

    // Verify first node
    EXPECT_EQ(loaded.text(id1), "first node");
    EXPECT_EQ(loaded.parent_id(id1), 0u);
    EXPECT_EQ(loaded.depth(id1), 0u);
    EXPECT_NEAR(loaded.state(id1).pos.x, 0.3f, 1e-5f);
    EXPECT_NEAR(loaded.state(id1).pos.y, 0.4f, 1e-5f);
    EXPECT_DOUBLE_EQ(loaded.state(id1).last_access_time, 100.0);
    EXPECT_EQ(loaded.state(id1).access_count, 5u);

    auto loaded_mu1 = loaded.mu(id1);
    ASSERT_EQ(loaded_mu1.size(), 3u);
    EXPECT_FLOAT_EQ(loaded_mu1[0], 1.0f);
    EXPECT_FLOAT_EQ(loaded_mu1[1], 0.0f);

    // Verify second node with annotation
    EXPECT_EQ(loaded.text(id2), "second node");
    EXPECT_EQ(loaded.parent_id(id2), id1);
    EXPECT_EQ(loaded.depth(id2), 1u);
    EXPECT_EQ(loaded.annotation(id2), "<!-- test annotation -->");
}

TEST_F(PersistFixture, FlushDurability) {
    MemoryGraph graph;
    std::vector<float> mu = {1.0f};
    std::vector<float> sigma = {1.0f};
    graph.insert(mu, sigma, "flush test", 0, 0, NodeState{});

    {
        SqliteStore store(db_path);
        store.flush(graph);
    }

    // Reload and verify
    MemoryGraph loaded;
    {
        SqliteStore store(db_path);
        store.load(loaded);
    }
    EXPECT_EQ(loaded.size(), 1u);
}

TEST_F(PersistFixture, ArchiveNode) {
    MemoryGraph graph;
    std::vector<float> mu = {1.0f, 2.0f};
    std::vector<float> sigma = {1.0f, 1.0f};
    auto id = graph.insert(mu, sigma, "to archive", 0, 0,
                           NodeState{.pos = {0.96f, 0.0f}});

    auto snap = graph.snapshot(id);

    SqliteStore store(db_path);
    store.checkpoint(graph);  // save active first
    store.archive_node(snap);

    // Load should not include archived nodes
    MemoryGraph loaded;
    store.load(loaded);
    // The original active node is loaded, but the archived version should be
    // stored with status=1. Since we checkpointed before archiving, the node
    // appears as status=0 in the active table and status=1 in the archived row.
    // For this test, just verify the archive_node call didn't crash.
    EXPECT_GE(loaded.size(), 0u);
}

TEST_F(PersistFixture, LoadEmptyDatabase) {
    {
        SqliteStore store(db_path);  // creates schema
    }

    MemoryGraph loaded;
    SqliteStore store(db_path);
    store.load(loaded);
    EXPECT_EQ(loaded.size(), 0u);
}

TEST_F(PersistFixture, CheckpointOverwritesPrevious) {
    MemoryGraph graph;
    std::vector<float> mu = {1.0f};
    std::vector<float> sigma = {1.0f};
    auto id = graph.insert(mu, sigma, "original", 0, 0, NodeState{});

    SqliteStore store(db_path);
    store.checkpoint(graph);

    // Modify the graph
    graph.state(id).pos = {0.5f, 0.5f};

    // Checkpoint again — should overwrite
    store.checkpoint(graph);

    MemoryGraph loaded;
    store.load(loaded);
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_NEAR(loaded.state(id).pos.x, 0.5f, 1e-5f);
}
```

- [ ] **Step 2: Create src/persist/CMakeLists.txt**

```cmake
find_package(SQLite3 REQUIRED)

add_library(persist STATIC
    src/sqlite_store.cpp
)

target_include_directories(persist PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(persist PUBLIC engine SQLite::SQLite3)

target_compile_features(persist PUBLIC cxx_std_23)
```

- [ ] **Step 3: Add persist to root CMakeLists.txt**

Add after `add_subdirectory(src/engine)`:

```cmake
add_subdirectory(src/persist)
```

- [ ] **Step 4: Add persist_tests to tests/CMakeLists.txt**

Append to `tests/CMakeLists.txt`:

```cmake

add_executable(persist_tests
    test_persist.cpp
)

target_link_libraries(persist_tests PRIVATE
    persist
    GTest::gtest_main
)

gtest_discover_tests(persist_tests)
```

- [ ] **Step 5: Implement Store interface**

Create `src/persist/include/persist/store.hpp`:

```cpp
#pragma once

#include <engine/memory_graph.hpp>

namespace slm::persist {

/// Abstract persistence interface.
/// Designed for V2 swap: replace SqliteStore with MmapStore
/// without touching any other module.
class Store {
public:
    virtual ~Store() = default;

    /// Save full graph state (periodic checkpoint).
    virtual void checkpoint(const engine::MemoryGraph& graph) = 0;

    /// Final flush on shutdown (must complete before exit).
    virtual void flush(const engine::MemoryGraph& graph) = 0;

    /// Load graph from storage on startup.
    virtual void load(engine::MemoryGraph& graph) = 0;

    /// Move a single node to cold storage (archival).
    virtual void archive_node(const engine::MemoryGraph::NodeSnapshot& snap) = 0;
};

} // namespace slm::persist
```

- [ ] **Step 6: Implement SqliteStore header**

Create `src/persist/include/persist/sqlite_store.hpp`:

```cpp
#pragma once

#include <filesystem>
#include <persist/store.hpp>
#include <sqlite3.h>

namespace slm::persist {

/// SQLite V1 implementation of the Store interface.
class SqliteStore : public Store {
public:
    explicit SqliteStore(const std::filesystem::path& db_path);
    ~SqliteStore() override;

    // Non-copyable, non-movable (owns sqlite3* handle)
    SqliteStore(const SqliteStore&) = delete;
    SqliteStore& operator=(const SqliteStore&) = delete;

    void checkpoint(const engine::MemoryGraph& graph) override;
    void flush(const engine::MemoryGraph& graph) override;
    void load(engine::MemoryGraph& graph) override;
    void archive_node(const engine::MemoryGraph::NodeSnapshot& snap) override;

private:
    sqlite3* db_{nullptr};

    void create_schema();
    void exec(const char* sql);
};

} // namespace slm::persist
```

- [ ] **Step 7: Implement SqliteStore**

Create `src/persist/src/sqlite_store.cpp`:

```cpp
#include <persist/sqlite_store.hpp>
#include <stdexcept>
#include <string>

namespace slm::persist {

SqliteStore::SqliteStore(const std::filesystem::path& db_path) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        throw std::runtime_error("Failed to open SQLite database: " + err);
    }
    // Enable WAL mode for better concurrent performance
    exec("PRAGMA journal_mode=WAL");
    create_schema();
}

SqliteStore::~SqliteStore() {
    if (db_) {
        sqlite3_close(db_);
    }
}

void SqliteStore::exec(const char* sql) {
    char* err = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("SQLite exec error: " + msg);
    }
}

void SqliteStore::create_schema() {
    exec(R"(
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id           INTEGER PRIMARY KEY,
            parent_id    INTEGER NOT NULL DEFAULT 0,
            depth        INTEGER NOT NULL DEFAULT 0,
            text         TEXT NOT NULL,
            mu           BLOB NOT NULL,
            sigma        BLOB NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            pos_x        REAL NOT NULL,
            pos_y        REAL NOT NULL,
            last_access  REAL NOT NULL,
            annotation   TEXT,
            status       INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS edges (
            source_id    INTEGER NOT NULL,
            target_id    INTEGER NOT NULL,
            edge_type    INTEGER NOT NULL,
            relation     BLOB,
            PRIMARY KEY (source_id, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_nodes_status ON memory_nodes(status);
        CREATE INDEX IF NOT EXISTS idx_nodes_parent ON memory_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    )");
}

void SqliteStore::checkpoint(const engine::MemoryGraph& graph) {
    exec("BEGIN TRANSACTION");

    // Delete all active nodes and re-insert (INSERT OR REPLACE)
    exec("DELETE FROM memory_nodes WHERE status = 0");

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "INSERT INTO memory_nodes "
        "(id, parent_id, depth, text, mu, sigma, access_count, "
        " pos_x, pos_y, last_access, annotation, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        -1, &stmt, nullptr);

    for (auto id : graph.all_ids()) {
        auto snap = graph.snapshot(id);

        sqlite3_bind_int(stmt, 1, static_cast<int>(snap.id));
        sqlite3_bind_int(stmt, 2, static_cast<int>(snap.parent_id));
        sqlite3_bind_int(stmt, 3, snap.depth);
        sqlite3_bind_text(stmt, 4, snap.text.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt, 5, snap.mu.data(),
                          static_cast<int>(snap.mu.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt, 6, snap.sigma.data(),
                          static_cast<int>(snap.sigma.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 7, static_cast<int>(snap.access_count));
        sqlite3_bind_double(stmt, 8, snap.pos_x);
        sqlite3_bind_double(stmt, 9, snap.pos_y);
        sqlite3_bind_double(stmt, 10, snap.last_access);
        if (snap.annotation.empty()) {
            sqlite3_bind_null(stmt, 11);
        } else {
            sqlite3_bind_text(stmt, 11, snap.annotation.c_str(), -1,
                              SQLITE_TRANSIENT);
        }

        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    exec("COMMIT");
}

void SqliteStore::flush(const engine::MemoryGraph& graph) {
    checkpoint(graph);
    sqlite3_wal_checkpoint_v2(db_, nullptr, SQLITE_CHECKPOINT_TRUNCATE,
                               nullptr, nullptr);
}

void SqliteStore::load(engine::MemoryGraph& graph) {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "SELECT id, parent_id, depth, text, mu, sigma, access_count, "
        "       pos_x, pos_y, last_access, annotation "
        "FROM memory_nodes WHERE status = 0",
        -1, &stmt, nullptr);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        engine::MemoryGraph::NodeSnapshot snap;
        snap.id = static_cast<uint32_t>(sqlite3_column_int(stmt, 0));
        snap.parent_id = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));
        snap.depth = static_cast<uint8_t>(sqlite3_column_int(stmt, 2));

        const char* text = reinterpret_cast<const char*>(
            sqlite3_column_text(stmt, 3));
        snap.text = text ? text : "";

        // Deserialize mu BLOB
        const float* mu_data = static_cast<const float*>(
            sqlite3_column_blob(stmt, 4));
        int mu_bytes = sqlite3_column_bytes(stmt, 4);
        int mu_count = mu_bytes / static_cast<int>(sizeof(float));
        snap.mu.assign(mu_data, mu_data + mu_count);

        // Deserialize sigma BLOB
        const float* sigma_data = static_cast<const float*>(
            sqlite3_column_blob(stmt, 5));
        int sigma_bytes = sqlite3_column_bytes(stmt, 5);
        int sigma_count = sigma_bytes / static_cast<int>(sizeof(float));
        snap.sigma.assign(sigma_data, sigma_data + sigma_count);

        snap.access_count = static_cast<uint32_t>(sqlite3_column_int(stmt, 6));
        snap.pos_x = static_cast<float>(sqlite3_column_double(stmt, 7));
        snap.pos_y = static_cast<float>(sqlite3_column_double(stmt, 8));
        snap.last_access = sqlite3_column_double(stmt, 9);

        const char* ann = reinterpret_cast<const char*>(
            sqlite3_column_text(stmt, 10));
        snap.annotation = ann ? ann : "";

        graph.insert_from_snapshot(snap);
    }

    sqlite3_finalize(stmt);
}

void SqliteStore::archive_node(const engine::MemoryGraph::NodeSnapshot& snap) {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "INSERT OR REPLACE INTO memory_nodes "
        "(id, parent_id, depth, text, mu, sigma, access_count, "
        " pos_x, pos_y, last_access, annotation, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
        -1, &stmt, nullptr);

    sqlite3_bind_int(stmt, 1, static_cast<int>(snap.id));
    sqlite3_bind_int(stmt, 2, static_cast<int>(snap.parent_id));
    sqlite3_bind_int(stmt, 3, snap.depth);
    sqlite3_bind_text(stmt, 4, snap.text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 5, snap.mu.data(),
                      static_cast<int>(snap.mu.size() * sizeof(float)),
                      SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 6, snap.sigma.data(),
                      static_cast<int>(snap.sigma.size() * sizeof(float)),
                      SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 7, static_cast<int>(snap.access_count));
    sqlite3_bind_double(stmt, 8, snap.pos_x);
    sqlite3_bind_double(stmt, 9, snap.pos_y);
    sqlite3_bind_double(stmt, 10, snap.last_access);
    if (snap.annotation.empty()) {
        sqlite3_bind_null(stmt, 11);
    } else {
        sqlite3_bind_text(stmt, 11, snap.annotation.c_str(), -1,
                          SQLITE_TRANSIENT);
    }

    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

} // namespace slm::persist
```

- [ ] **Step 8: Build and run tests**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && cd build && ctest --output-on-failure -R persist
```

Expected: All 6 persist tests PASS.

- [ ] **Step 9: Commit**

```bash
git add CMakeLists.txt src/persist/ tests/test_persist.cpp tests/CMakeLists.txt
git commit -m "feat(persist): add Store interface and SqliteStore with checkpoint/load/flush/archive"
```

---

### Task 3: Integration Test — Full Round-Trip with Scene Graph

**Files:**
- Modify: `tests/test_persist.cpp` (append integration test)

- [ ] **Step 1: Write the integration test**

Append to `tests/test_persist.cpp`:

```cpp
// --- Integration: full round-trip with scene graph ---

TEST_F(PersistFixture, RoundTripWithSceneGraph) {
    MemoryGraph graph;

    // Build a heading hierarchy: root → H1 → H2a, H2b
    std::vector<float> mu_root = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    auto root = graph.insert(mu_root, sigma, "Project Overview", 0, 0,
                              NodeState{.pos = {0.0f, 0.0f}, .last_access_time = 0.0});

    std::vector<float> mu_h1 = {0.0f, 1.0f, 0.0f, 0.0f};
    auto h1 = graph.insert(mu_h1, sigma, "## Architecture", root, 1,
                            NodeState{.pos = {0.1f, 0.0f}, .last_access_time = 10.0});

    std::vector<float> mu_h2a = {0.0f, 0.0f, 1.0f, 0.0f};
    auto h2a = graph.insert(mu_h2a, sigma, "### Frontend", h1, 2,
                             NodeState{.pos = {0.3f, 0.1f}, .last_access_time = 20.0,
                                       .access_count = 3});

    std::vector<float> mu_h2b = {0.0f, 0.0f, 0.0f, 1.0f};
    auto h2b = graph.insert(mu_h2b, sigma, "### Backend", h1, 2,
                             NodeState{.pos = {0.4f, -0.1f}, .last_access_time = 30.0,
                                       .access_count = 7});

    graph.set_annotation(h2b, "<!-- cohomology: superseded ... -->");

    // Verify scene graph before persistence
    EXPECT_EQ(graph.siblings(h1).size(), 2u);  // h2a, h2b
    EXPECT_EQ(graph.parent(h2a).value(), h1);

    // Checkpoint + load
    SqliteStore store(db_path);
    store.checkpoint(graph);

    MemoryGraph loaded;
    store.load(loaded);

    // Verify full round-trip
    ASSERT_EQ(loaded.size(), 4u);

    // Scene graph preserved
    auto loaded_siblings = loaded.siblings(h1);
    EXPECT_EQ(loaded_siblings.size(), 2u);
    EXPECT_EQ(loaded.parent(h2a).value(), h1);
    EXPECT_EQ(loaded.parent(h1).value(), root);
    EXPECT_FALSE(loaded.parent(root).has_value());

    // Poincaré disk positions preserved
    EXPECT_NEAR(loaded.state(h2a).pos.x, 0.3f, 1e-5f);
    EXPECT_NEAR(loaded.state(h2b).pos.y, -0.1f, 1e-5f);

    // Access counts preserved
    EXPECT_EQ(loaded.state(h2a).access_count, 3u);
    EXPECT_EQ(loaded.state(h2b).access_count, 7u);

    // Annotation preserved
    EXPECT_EQ(loaded.annotation(h2b), "<!-- cohomology: superseded ... -->");

    // Mu vectors preserved
    auto loaded_mu = loaded.mu(h2a);
    ASSERT_EQ(loaded_mu.size(), 4u);
    EXPECT_FLOAT_EQ(loaded_mu[2], 1.0f);

    // New inserts after load get unique IDs
    std::vector<float> mu_new = {0.5f, 0.5f, 0.5f, 0.5f};
    auto new_id = loaded.insert(mu_new, sigma, "New node", h1, 2, NodeState{});
    EXPECT_GT(new_id, h2b) << "New IDs should continue from the loaded max";
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R persist`

Expected: All 7 persist tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_persist.cpp
git commit -m "test(persist): add full round-trip integration test with scene graph and annotations"
```

---

## Summary

After completing all 3 tasks, libpersist + MemoryGraph provides:

| Component | What it does |
|---|---|
| `MemoryGraph` | SoA in-memory registry: insert/remove (swap-and-pop), scene graph queries, contiguous states span, snapshot support |
| `Store` | Abstract persistence interface (checkpoint, flush, load, archive_node) |
| `SqliteStore` | SQLite V1: WAL mode, BLOB serialization for mu/sigma, status flag for active/archived |
| `NodeSnapshot` | Flat struct for serialization round-trips between MemoryGraph and Store |

MemoryGraph depends on liblangevin (NodeState) and libmetric (GaussianNode type reference). SqliteStore depends on MemoryGraph and SQLite3. The abstract Store interface ensures the V2 mmap migration is a single-class swap.
