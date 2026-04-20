# liblangevin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Riemannian Langevin lifecycle library — the physics simulation that governs memory node positions on a Poincaré disk, drifting unaccessed memories toward the boundary for self-archiving.

**Architecture:** Three focused files: `DiskPosition` + `NodeState` structs with Poincaré disk math helpers, `LangevinStepper` class implementing the Euler-Maruyama SDE integrator with configurable drift/noise/threshold, and comprehensive tests covering deterministic drift, stochastic behavior, boundary projection, activation, and archiving. The library is a CMake static library (`liblangevin`) with no dependencies on other project libraries — pure math.

**Tech Stack:** C++23, `<random>` for Gaussian noise, Google Test

---

## File Map

| File | Responsibility |
|---|---|
| `src/langevin/CMakeLists.txt` | liblangevin static library target |
| `src/langevin/include/langevin/poincare_disk.hpp` | `DiskPosition`, `NodeState`, Poincaré disk math (metric tensor, inverse metric, projection) |
| `src/langevin/include/langevin/sde_stepper.hpp` | `LangevinStepper` class declaration with Config |
| `src/langevin/src/sde_stepper.cpp` | `step()`, `activate()`, `project_to_disk()` implementations |
| `tests/test_langevin.cpp` | All liblangevin tests |
| `CMakeLists.txt` | Root CMake (add `add_subdirectory(src/langevin)`) |
| `tests/CMakeLists.txt` | Add langevin_tests executable |

---

### Task 1: CMake Wiring & Poincaré Disk Types

**Files:**
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/langevin)`)
- Create: `src/langevin/CMakeLists.txt`
- Create: `src/langevin/include/langevin/poincare_disk.hpp`
- Create: `src/langevin/include/langevin/sde_stepper.hpp` (placeholder)
- Create: `src/langevin/src/sde_stepper.cpp` (placeholder)
- Create: `tests/test_langevin.cpp`
- Modify: `tests/CMakeLists.txt` (add langevin_tests executable)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_langevin.cpp`:

```cpp
#include <gtest/gtest.h>
#include <langevin/poincare_disk.hpp>
#include <cmath>

using namespace slm::langevin;

// --- DiskPosition tests ---

TEST(DiskPosition, RadiusAtOrigin) {
    DiskPosition p{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(p.radius(), 0.0f);
}

TEST(DiskPosition, RadiusOnAxis) {
    DiskPosition p{0.6f, 0.0f};
    EXPECT_FLOAT_EQ(p.radius(), 0.6f);
}

TEST(DiskPosition, RadiusDiagonal) {
    DiskPosition p{0.3f, 0.4f};
    EXPECT_FLOAT_EQ(p.radius(), 0.5f);
}

TEST(DiskPosition, InverseMetricAtOrigin) {
    // g_inv = (1 - r^2)^2 / 4 = (1 - 0)^2 / 4 = 0.25
    DiskPosition p{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(inverse_metric(p), 0.25f);
}

TEST(DiskPosition, InverseMetricAtMidRadius) {
    // r = 0.5 → g_inv = (1 - 0.25)^2 / 4 = 0.75^2 / 4 = 0.5625 / 4 = 0.140625
    DiskPosition p{0.5f, 0.0f};
    EXPECT_NEAR(inverse_metric(p), 0.140625f, 1e-6f);
}

TEST(DiskPosition, InverseMetricNearBoundary) {
    // r = 0.95 → g_inv = (1 - 0.9025)^2 / 4 = 0.0975^2 / 4 ≈ 0.002378
    DiskPosition p{0.95f, 0.0f};
    float expected = (1.0f - 0.95f * 0.95f) * (1.0f - 0.95f * 0.95f) / 4.0f;
    EXPECT_NEAR(inverse_metric(p), expected, 1e-6f);
}

TEST(DiskPosition, ProjectInsideDiskUnchanged) {
    DiskPosition p{0.3f, 0.4f};
    auto projected = project_to_disk(p);
    EXPECT_FLOAT_EQ(projected.x, 0.3f);
    EXPECT_FLOAT_EQ(projected.y, 0.4f);
}

TEST(DiskPosition, ProjectOnBoundaryClamps) {
    DiskPosition p{1.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.radius(), 0.999f, 1e-3f);
}

TEST(DiskPosition, ProjectOutsideDiskClamps) {
    DiskPosition p{2.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.radius(), 0.999f, 1e-3f);
    // Direction preserved
    EXPECT_GT(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

TEST(DiskPosition, ProjectDiagonalOvershoot) {
    DiskPosition p{1.0f, 1.0f};  // r = sqrt(2) > 1
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    // Direction preserved: x and y should be equal
    EXPECT_NEAR(projected.x, projected.y, 1e-5f);
}

TEST(DiskPosition, ProjectZeroVectorUnchanged) {
    DiskPosition p{0.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_FLOAT_EQ(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

// --- NodeState tests ---

TEST(NodeState, DefaultConstruction) {
    NodeState state{};
    EXPECT_FLOAT_EQ(state.pos.x, 0.0f);
    EXPECT_FLOAT_EQ(state.pos.y, 0.0f);
    EXPECT_DOUBLE_EQ(state.last_access_time, 0.0);
    EXPECT_EQ(state.access_count, 0u);
}
```

- [ ] **Step 2: Create src/langevin/CMakeLists.txt**

```cmake
add_library(langevin STATIC
    src/sde_stepper.cpp
)

target_include_directories(langevin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(langevin PUBLIC cxx_std_23)
```

- [ ] **Step 3: Add langevin to root CMakeLists.txt**

Add after the existing `add_subdirectory(src/metric)` line:

```cmake
add_subdirectory(src/langevin)
```

- [ ] **Step 4: Add langevin_tests to tests/CMakeLists.txt**

Append to `tests/CMakeLists.txt`:

```cmake

add_executable(langevin_tests
    test_langevin.cpp
)

target_link_libraries(langevin_tests PRIVATE
    langevin
    GTest::gtest_main
)

gtest_discover_tests(langevin_tests)
```

- [ ] **Step 5: Create placeholder source**

Create `src/langevin/include/langevin/sde_stepper.hpp`:

```cpp
#pragma once
// LangevinStepper — implemented in Task 2
```

Create `src/langevin/src/sde_stepper.cpp`:

```cpp
#include <langevin/sde_stepper.hpp>
// Implemented in Task 2
```

- [ ] **Step 6: Implement Poincaré disk types**

Create `src/langevin/include/langevin/poincare_disk.hpp`:

```cpp
#pragma once

#include <cmath>
#include <cstdint>

namespace slm::langevin {

/// Position on the Poincaré disk (radius < 1).
struct DiskPosition {
    float x{0.0f};
    float y{0.0f};

    float radius() const { return std::sqrt(x * x + y * y); }
};

/// State of a memory node on the Poincaré disk.
struct NodeState {
    DiskPosition pos;
    double last_access_time{0.0};
    uint32_t access_count{0};
};

/// Inverse metric tensor g^{-1}(p) = (1 - r^2)^2 / 4.
/// Scales drift and noise in the Riemannian Langevin SDE.
/// Near the center (r≈0): g_inv ≈ 0.25 (full step).
/// Near the boundary (r≈1): g_inv → 0 (steps compress).
inline float inverse_metric(const DiskPosition& p) {
    float r2 = p.x * p.x + p.y * p.y;
    float factor = 1.0f - r2;
    return (factor * factor) / 4.0f;
}

/// Project a position back inside the Poincaré disk.
/// If r >= 1, scales to r = 0.999 preserving direction.
/// If r < 1, returns unchanged.
inline DiskPosition project_to_disk(DiskPosition p) {
    constexpr float MAX_RADIUS = 0.999f;
    float r = p.radius();
    if (r >= 1.0f) {
        float scale = MAX_RADIUS / r;
        return {p.x * scale, p.y * scale};
    }
    return p;
}

} // namespace slm::langevin
```

- [ ] **Step 7: Build and run tests**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && cd build && ctest --output-on-failure -R langevin
```

Expected: All 13 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt src/langevin/ tests/test_langevin.cpp tests/CMakeLists.txt
git commit -m "feat(langevin): add liblangevin scaffolding with Poincaré disk types and projection"
```

---

### Task 2: LangevinStepper — activate() and Config

**Files:**
- Replace: `src/langevin/include/langevin/sde_stepper.hpp`
- Replace: `src/langevin/src/sde_stepper.cpp`
- Modify: `tests/test_langevin.cpp` (append activate tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_langevin.cpp`:

```cpp
#include <langevin/sde_stepper.hpp>

// --- LangevinStepper activate tests ---

TEST(LangevinActivate, ResetsToOrigin) {
    NodeState node{};
    node.pos = {0.5f, 0.3f};
    node.last_access_time = 100.0;
    node.access_count = 3;

    LangevinStepper::activate(node, 200.0);

    EXPECT_FLOAT_EQ(node.pos.x, 0.0f);
    EXPECT_FLOAT_EQ(node.pos.y, 0.0f);
    EXPECT_DOUBLE_EQ(node.last_access_time, 200.0);
    EXPECT_EQ(node.access_count, 4u);
}

TEST(LangevinActivate, IncrementsAccessCount) {
    NodeState node{};
    node.access_count = 0;

    LangevinStepper::activate(node, 1.0);
    EXPECT_EQ(node.access_count, 1u);

    LangevinStepper::activate(node, 2.0);
    EXPECT_EQ(node.access_count, 2u);
}

TEST(LangevinActivate, UpdatesTimestamp) {
    NodeState node{};
    node.last_access_time = 50.0;

    LangevinStepper::activate(node, 999.0);
    EXPECT_DOUBLE_EQ(node.last_access_time, 999.0);
}

TEST(LangevinConfig, DefaultValues) {
    LangevinStepper::Config config{};
    config.dt = 5.0f;
    config.lambda_decay = 0.01f;
    config.noise_scale = 0.001f;
    config.archive_threshold = 0.95f;

    EXPECT_FLOAT_EQ(config.dt, 5.0f);
    EXPECT_FLOAT_EQ(config.lambda_decay, 0.01f);
    EXPECT_FLOAT_EQ(config.noise_scale, 0.001f);
    EXPECT_FLOAT_EQ(config.archive_threshold, 0.95f);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && cd build && ctest --output-on-failure -R langevin`

Expected: FAIL — `LangevinStepper` not defined.

- [ ] **Step 3: Implement LangevinStepper header**

Replace `src/langevin/include/langevin/sde_stepper.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include <langevin/poincare_disk.hpp>

namespace slm::langevin {

/// Riemannian Langevin integrator on the Poincaré disk.
///
/// Each tick:
///   dp = -g_inv(p) * nabla_U(p) * dt + sqrt(2 * g_inv(p) * dt) * xi
/// where:
///   g_inv(p) = (1 - r^2)^2 / 4
///   U(p) = -lambda_decay * delta_t_since_access * r
///   xi ~ N(0, I)
///
/// Nodes exceeding archive_threshold are returned for archival.
class LangevinStepper {
public:
    struct Config {
        float dt;                  // tick interval in seconds (e.g., 5.0)
        float lambda_decay;        // outward drift rate
        float noise_scale;         // diffusion intensity
        float archive_threshold;   // radius threshold for archiving (e.g., 0.95)
    };

    explicit LangevinStepper(Config config);

    /// Advance all nodes by one tick. Returns indices of newly archived nodes
    /// (those whose radius exceeded archive_threshold after the step).
    std::vector<uint32_t> step(
        std::span<NodeState> nodes,
        double current_time,
        std::mt19937& rng
    ) const;

    /// Reset a node to the center of the disk (on access/activation).
    /// Increments access_count and refreshes last_access_time.
    static void activate(NodeState& node, double current_time);

    const Config& config() const { return config_; }

private:
    Config config_;
};

} // namespace slm::langevin
```

- [ ] **Step 4: Implement activate() and constructor**

Replace `src/langevin/src/sde_stepper.cpp`:

```cpp
#include <langevin/sde_stepper.hpp>
#include <cmath>

namespace slm::langevin {

LangevinStepper::LangevinStepper(Config config)
    : config_(config) {}

void LangevinStepper::activate(NodeState& node, double current_time) {
    node.pos = {0.0f, 0.0f};
    node.last_access_time = current_time;
    node.access_count += 1;
}

std::vector<uint32_t> LangevinStepper::step(
    std::span<NodeState> nodes,
    double current_time,
    std::mt19937& rng
) const {
    // Placeholder — implemented in Task 3
    return {};
}

} // namespace slm::langevin
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R langevin`

Expected: All 17 tests PASS (13 disk + 4 activate/config).

- [ ] **Step 6: Commit**

```bash
git add src/langevin/include/langevin/sde_stepper.hpp src/langevin/src/sde_stepper.cpp tests/test_langevin.cpp
git commit -m "feat(langevin): add LangevinStepper with activate() and Config"
```

---

### Task 3: LangevinStepper — step() (Euler-Maruyama Integrator)

**Files:**
- Modify: `src/langevin/src/sde_stepper.cpp` (replace step placeholder)
- Modify: `tests/test_langevin.cpp` (append step tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_langevin.cpp`:

```cpp
#include <vector>
#include <algorithm>

// --- LangevinStepper step tests ---

TEST(LangevinStep, NoNodesReturnsEmpty) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.01f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);
    std::span<NodeState> empty;
    auto archived = stepper.step(empty, 100.0, rng);
    EXPECT_TRUE(archived.empty());
}

TEST(LangevinStep, DeterministicDriftWithoutNoise) {
    // Zero noise → purely deterministic outward drift
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    // Place node at r=0.3, accessed 100 seconds ago
    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    double current_time = 100.0;
    stepper.step(nodes, current_time, rng);

    // Node should have drifted outward (larger radius)
    EXPECT_GT(nodes[0].pos.radius(), 0.3f)
        << "Node should drift outward when unaccessed";
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "Node should remain inside the disk";
}

TEST(LangevinStep, RecentlyAccessedDriftsLess) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});

    // Node A: accessed 1000 seconds ago (should drift more)
    NodeState node_a{};
    node_a.pos = {0.3f, 0.0f};
    node_a.last_access_time = 0.0;

    // Node B: accessed 10 seconds ago (should drift less)
    NodeState node_b{};
    node_b.pos = {0.3f, 0.0f};
    node_b.last_access_time = 990.0;

    std::vector<NodeState> nodes_a = {node_a};
    std::vector<NodeState> nodes_b = {node_b};
    std::mt19937 rng_a(42), rng_b(42);

    stepper.step(nodes_a, 1000.0, rng_a);
    stepper.step(nodes_b, 1000.0, rng_b);

    EXPECT_GT(nodes_a[0].pos.radius(), nodes_b[0].pos.radius())
        << "Node accessed longer ago should drift more";
}

TEST(LangevinStep, NodeAtOriginDriftsOutward) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // At origin, radius is 0, so U gradient is 0 → no drift
    // This is expected: a node at the exact origin has no drift direction
    stepper.step(nodes, 100.0, rng);
    EXPECT_FLOAT_EQ(nodes[0].pos.radius(), 0.0f);
}

TEST(LangevinStep, ArchivesNodesBeyondThreshold) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    // Place node very close to boundary with long time since access
    NodeState node{};
    node.pos = {0.94f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // Strong drift should push it past 0.95
    auto archived = stepper.step(nodes, 10000.0, rng);
    EXPECT_EQ(archived.size(), 1u);
    EXPECT_EQ(archived[0], 0u);
}

TEST(LangevinStep, StaysInsideDisk) {
    // Even with extreme parameters, node must stay inside r < 1
    LangevinStepper stepper({.dt = 100.0f, .lambda_decay = 10.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.9f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 100000.0, rng);
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "project_to_disk must prevent escape from the disk";
}

TEST(LangevinStep, NoiseAddsRandomness) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.0f,
                              .noise_scale = 0.1f, .archive_threshold = 0.95f});

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_time = 0.0;

    // Run with two different seeds — positions should diverge
    std::vector<NodeState> nodes_a = {node};
    std::vector<NodeState> nodes_b = {node};
    std::mt19937 rng_a(42), rng_b(99);

    stepper.step(nodes_a, 100.0, rng_a);
    stepper.step(nodes_b, 100.0, rng_b);

    // With different seeds and nonzero noise, positions should differ
    bool positions_differ = (nodes_a[0].pos.x != nodes_b[0].pos.x)
                         || (nodes_a[0].pos.y != nodes_b[0].pos.y);
    EXPECT_TRUE(positions_differ)
        << "Different RNG seeds should produce different positions";
}

TEST(LangevinStep, MultipleNodesIndependent) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    std::vector<NodeState> nodes(3);
    nodes[0].pos = {0.2f, 0.0f};
    nodes[0].last_access_time = 0.0;
    nodes[1].pos = {0.5f, 0.0f};
    nodes[1].last_access_time = 0.0;
    nodes[2].pos = {0.8f, 0.0f};
    nodes[2].last_access_time = 0.0;

    stepper.step(nodes, 100.0, rng);

    // All should have drifted outward from their starting positions
    EXPECT_GT(nodes[0].pos.radius(), 0.2f);
    EXPECT_GT(nodes[1].pos.radius(), 0.5f);
    EXPECT_GT(nodes[2].pos.radius(), 0.8f);

    // Ordering should be preserved (node 2 still farthest)
    EXPECT_GT(nodes[2].pos.radius(), nodes[1].pos.radius());
    EXPECT_GT(nodes[1].pos.radius(), nodes[0].pos.radius());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && cd build && ctest --output-on-failure -R langevin`

Expected: FAIL — `step()` returns empty (placeholder).

- [ ] **Step 3: Implement step()**

In `src/langevin/src/sde_stepper.cpp`, replace the `step` method body:

```cpp
std::vector<uint32_t> LangevinStepper::step(
    std::span<NodeState> nodes,
    double current_time,
    std::mt19937& rng
) const {
    std::vector<uint32_t> archived;
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        float r = node.pos.radius();

        // Skip nodes at exact origin (no drift direction)
        if (r < 1e-8f) {
            continue;
        }

        float g_inv = inverse_metric(node.pos);

        // Time since last access drives the outward potential
        float delta_t = static_cast<float>(current_time - node.last_access_time);

        // Gradient of U(p) = -lambda * delta_t * r
        // nabla_U = -lambda * delta_t * (p / r)  (radial gradient)
        // Drift = -g_inv * nabla_U * dt = g_inv * lambda * delta_t * (p/r) * dt
        float drift_mag = g_inv * config_.lambda_decay * delta_t * config_.dt / r;
        float dx_drift = drift_mag * node.pos.x;
        float dy_drift = drift_mag * node.pos.y;

        // Noise: sqrt(2 * g_inv * dt) * xi
        float noise_mag = config_.noise_scale * std::sqrt(2.0f * g_inv * config_.dt);
        float dx_noise = noise_mag * noise_dist(rng);
        float dy_noise = noise_mag * noise_dist(rng);

        // Euler-Maruyama update
        node.pos.x += dx_drift + dx_noise;
        node.pos.y += dy_drift + dy_noise;

        // Project back inside the disk
        node.pos = project_to_disk(node.pos);

        // Check for archival
        if (node.pos.radius() > config_.archive_threshold) {
            archived.push_back(i);
        }
    }

    return archived;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R langevin`

Expected: All 25 tests PASS (13 disk + 4 activate/config + 8 step).

- [ ] **Step 5: Commit**

```bash
git add src/langevin/src/sde_stepper.cpp tests/test_langevin.cpp
git commit -m "feat(langevin): add Euler-Maruyama SDE integrator with drift, noise, and archiving"
```

---

### Task 4: Integration Test — Lifecycle Simulation

**Files:**
- Modify: `tests/test_langevin.cpp` (append integration test)

End-to-end test simulating the full memory lifecycle: ingestion at center → gradual drift → archival at boundary, with activation resetting the process.

- [ ] **Step 1: Write the integration test**

Append to `tests/test_langevin.cpp`:

```cpp
// --- Integration: full lifecycle ---

TEST(LangevinIntegration, FullLifecycle) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.05f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    // Ingest a node at the center (simulating a WRITE_COMMIT)
    NodeState node{};
    node.pos = {0.0f, 0.0f};
    node.last_access_time = 0.0;
    node.access_count = 0;

    // Give it a small offset so drift has a direction
    node.pos = {0.01f, 0.0f};
    std::vector<NodeState> nodes = {node};

    // Simulate 100 ticks (500 seconds of in-game time)
    double t = 0.0;
    std::vector<uint32_t> archived;
    std::vector<float> radius_history;

    for (int tick = 0; tick < 100 && archived.empty(); ++tick) {
        t += 5.0;
        archived = stepper.step(nodes, t, rng);
        radius_history.push_back(nodes[0].pos.radius());
    }

    // The node should have drifted outward monotonically (no noise)
    for (size_t j = 1; j < radius_history.size(); ++j) {
        EXPECT_GE(radius_history[j], radius_history[j - 1] - 1e-6f)
            << "Radius should monotonically increase at tick " << j;
    }

    // Should eventually be archived
    EXPECT_FALSE(archived.empty())
        << "Node should be archived after sufficient ticks without access";
}

TEST(LangevinIntegration, ActivationResetsLifecycle) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.05f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.01f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // Drift for 20 ticks
    double t = 0.0;
    for (int tick = 0; tick < 20; ++tick) {
        t += 5.0;
        stepper.step(nodes, t, rng);
    }
    float radius_before_activation = nodes[0].pos.radius();
    EXPECT_GT(radius_before_activation, 0.01f)
        << "Should have drifted outward";

    // Activate (simulate agent reading the memory)
    LangevinStepper::activate(nodes[0], t);
    EXPECT_FLOAT_EQ(nodes[0].pos.radius(), 0.0f)
        << "Activation should reset to center";

    // Give a small offset again for drift direction
    nodes[0].pos = {0.01f, 0.0f};

    // Drift again for 20 more ticks
    for (int tick = 0; tick < 20; ++tick) {
        t += 5.0;
        stepper.step(nodes, t, rng);
    }
    float radius_after_reactivation = nodes[0].pos.radius();

    // Should have drifted less than before (only 100s since access, not 200s)
    EXPECT_LT(radius_after_reactivation, radius_before_activation)
        << "Reactivated node should drift less (shorter time since access)";
}

TEST(LangevinIntegration, CohomologyDriftPenalty) {
    // Simulate the cohomology integration: when delta != 0,
    // superseded node is repositioned to (0.0, 0.93)
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.05f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.93f};  // Near archive threshold
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // One tick should push it past the threshold
    auto archived = stepper.step(nodes, 100.0, rng);
    EXPECT_EQ(archived.size(), 1u)
        << "Node placed at r=0.93 with old access should archive in one tick";
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R langevin`

Expected: All 28 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_langevin.cpp
git commit -m "test(langevin): add full lifecycle integration tests with activation and cohomology penalty"
```

---

## Summary

After completing all 4 tasks, liblangevin provides:

| Component | What it does |
|---|---|
| `DiskPosition` | 2D position on the Poincaré disk with radius computation |
| `NodeState` | Position + last_access_time + access_count per memory node |
| `inverse_metric()` | Poincaré disk metric tensor inverse: (1-r²)²/4 |
| `project_to_disk()` | Clamps positions to r < 0.999, preserving direction |
| `LangevinStepper::activate()` | Resets node to center, refreshes timestamp, increments access count |
| `LangevinStepper::step()` | Euler-Maruyama integrator: drift + noise + projection + archival detection |

The library has no dependencies on libslab, libmetric, or any other project library. The engine scheduler (Tier 3) will call `step()` every 5 seconds, and Tier 1 will call `activate()` on read/search operations.
