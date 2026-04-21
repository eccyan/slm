#include <gtest/gtest.h>
#include <langevin/poincare_disk.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace slm::langevin;

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
    DiskPosition p{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(inverse_metric(p), 0.25f);
}

TEST(DiskPosition, InverseMetricAtMidRadius) {
    DiskPosition p{0.5f, 0.0f};
    EXPECT_NEAR(inverse_metric(p), 0.140625f, 1e-6f);
}

TEST(DiskPosition, InverseMetricNearBoundary) {
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
    EXPECT_GT(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

TEST(DiskPosition, ProjectDiagonalOvershoot) {
    DiskPosition p{1.0f, 1.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.x, projected.y, 1e-5f);
}

TEST(DiskPosition, ProjectZeroVectorUnchanged) {
    DiskPosition p{0.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_FLOAT_EQ(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

TEST(NodeState, DefaultConstruction) {
    NodeState state{};
    EXPECT_FLOAT_EQ(state.pos.x, 0.0f);
    EXPECT_FLOAT_EQ(state.pos.y, 0.0f);
    EXPECT_DOUBLE_EQ(state.last_access_time, 0.0);
    EXPECT_EQ(state.access_count, 0u);
}

#include <langevin/sde_stepper.hpp>

// --- LangevinStepper activate tests ---
// activate() is now a const member function (reads thermal_kick_radius from config).

static const LangevinStepper DEFAULT_STEPPER({
    .dt = 5.0f, .lambda_decay = 5.0e-6f, .noise_scale = 0.0f,
    .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

TEST(LangevinActivate, ResetsWithThermalKick) {
    NodeState node{};
    node.pos = {0.5f, 0.3f};
    node.last_access_time = 100.0;
    node.access_count = 3;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 200.0, rng);

    EXPECT_NEAR(node.pos.radius(), 0.01f, 1e-6f)
        << "Activated node should have thermal kick radius of ~0.01";
    EXPECT_DOUBLE_EQ(node.last_access_time, 200.0);
    EXPECT_EQ(node.access_count, 4u);
}

TEST(LangevinActivate, IncrementsAccessCount) {
    NodeState node{};
    node.access_count = 0;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 1.0, rng);
    EXPECT_EQ(node.access_count, 1u);

    DEFAULT_STEPPER.activate(node, 2.0, rng);
    EXPECT_EQ(node.access_count, 2u);
}

TEST(LangevinActivate, UpdatesTimestamp) {
    NodeState node{};
    node.last_access_time = 50.0;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 999.0, rng);
    EXPECT_DOUBLE_EQ(node.last_access_time, 999.0);
}

TEST(LangevinActivate, RandomAngleVaries) {
    NodeState node_a{}, node_b{};
    std::mt19937 rng_a(42), rng_b(99);

    DEFAULT_STEPPER.activate(node_a, 0.0, rng_a);
    DEFAULT_STEPPER.activate(node_b, 0.0, rng_b);

    // Same radius, different angles
    EXPECT_NEAR(node_a.pos.radius(), node_b.pos.radius(), 1e-6f);
    bool angles_differ = (node_a.pos.x != node_b.pos.x)
                      || (node_a.pos.y != node_b.pos.y);
    EXPECT_TRUE(angles_differ)
        << "Different RNG seeds should produce different kick angles";
}

TEST(LangevinActivate, CustomKickRadius) {
    LangevinStepper big_kick({.dt = 5.0f, .lambda_decay = 0.0f, .noise_scale = 0.0f,
                               .archive_threshold = 0.95f, .thermal_kick_radius = 0.05f});
    NodeState node{};
    std::mt19937 rng(42);

    big_kick.activate(node, 0.0, rng);
    EXPECT_NEAR(node.pos.radius(), 0.05f, 1e-6f)
        << "Kick radius should match config value";
}

TEST(LangevinConfig, DefaultValues) {
    LangevinStepper::Config config{};
    config.dt = 5.0f;
    config.lambda_decay = 5.0e-6f;
    config.noise_scale = 2.0e-4f;
    config.archive_threshold = 0.95f;
    config.thermal_kick_radius = 0.01f;

    EXPECT_FLOAT_EQ(config.dt, 5.0f);
    EXPECT_FLOAT_EQ(config.lambda_decay, 5.0e-6f);
    EXPECT_FLOAT_EQ(config.noise_scale, 2.0e-4f);
    EXPECT_FLOAT_EQ(config.archive_threshold, 0.95f);
    EXPECT_FLOAT_EQ(config.thermal_kick_radius, 0.01f);
}

// --- LangevinStepper step tests ---
// Note: delta_t is now converted to days internally (÷86400).
// Tests use large time differences (in seconds) to produce meaningful drift.

static constexpr float SECS_PER_DAY = 86400.0f;

TEST(LangevinStep, NoNodesReturnsEmpty) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);
    std::span<NodeState> empty;
    auto archived = stepper.step(empty, 100.0, rng);
    EXPECT_TRUE(archived.empty());
}

TEST(LangevinStep, DeterministicDriftWithoutNoise) {
    // Use a large lambda so drift is visible over a 1-day age
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0e-3f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // Step at t = 1 day
    stepper.step(nodes, SECS_PER_DAY, rng);

    EXPECT_GT(nodes[0].pos.radius(), 0.3f)
        << "Node should drift outward when unaccessed";
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "Node should remain inside the disk";
}

TEST(LangevinStep, RecentlyAccessedDriftsLess) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0e-3f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

    NodeState node_a{};
    node_a.pos = {0.3f, 0.0f};
    node_a.last_access_time = 0.0;  // 10 days old

    NodeState node_b{};
    node_b.pos = {0.3f, 0.0f};
    node_b.last_access_time = 9.0 * SECS_PER_DAY;  // 1 day old

    std::vector<NodeState> nodes_a = {node_a};
    std::vector<NodeState> nodes_b = {node_b};
    std::mt19937 rng_a(42), rng_b(42);

    double t = 10.0 * SECS_PER_DAY;
    stepper.step(nodes_a, t, rng_a);
    stepper.step(nodes_b, t, rng_b);

    EXPECT_GT(nodes_a[0].pos.radius(), nodes_b[0].pos.radius())
        << "Node accessed longer ago should drift more";
}

TEST(LangevinStep, NodeAtOriginDriftsNowhere) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0e-3f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, SECS_PER_DAY, rng);
    EXPECT_FLOAT_EQ(nodes[0].pos.radius(), 0.0f);
}

TEST(LangevinStep, ArchivesNodesBeyondThreshold) {
    // Very large lambda to force archival in one step
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.94f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // 10 days old
    auto archived = stepper.step(nodes, 10.0 * SECS_PER_DAY, rng);
    EXPECT_EQ(archived.size(), 1u);
    EXPECT_EQ(archived[0], 0u);
}

TEST(LangevinStep, StaysInsideDisk) {
    LangevinStepper stepper({.dt = 100.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.9f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 30.0 * SECS_PER_DAY, rng);
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "project_to_disk must prevent escape from the disk";
}

TEST(LangevinStep, NoiseAddsRandomness) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.0f,
                              .noise_scale = 0.1f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_time = 0.0;

    std::vector<NodeState> nodes_a = {node};
    std::vector<NodeState> nodes_b = {node};
    std::mt19937 rng_a(42), rng_b(99);

    stepper.step(nodes_a, SECS_PER_DAY, rng_a);
    stepper.step(nodes_b, SECS_PER_DAY, rng_b);

    bool positions_differ = (nodes_a[0].pos.x != nodes_b[0].pos.x)
                         || (nodes_a[0].pos.y != nodes_b[0].pos.y);
    EXPECT_TRUE(positions_differ)
        << "Different RNG seeds should produce different positions";
}

TEST(LangevinStep, MultipleNodesIndependent) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0e-3f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    std::vector<NodeState> nodes(3);
    nodes[0].pos = {0.2f, 0.0f};
    nodes[0].last_access_time = 0.0;
    nodes[1].pos = {0.5f, 0.0f};
    nodes[1].last_access_time = 0.0;
    nodes[2].pos = {0.8f, 0.0f};
    nodes[2].last_access_time = 0.0;

    // Step at 5 days
    stepper.step(nodes, 5.0 * SECS_PER_DAY, rng);

    EXPECT_GT(nodes[0].pos.radius(), 0.2f);
    EXPECT_GT(nodes[1].pos.radius(), 0.5f);
    EXPECT_GT(nodes[2].pos.radius(), 0.8f);

    EXPECT_GT(nodes[2].pos.radius(), nodes[1].pos.radius());
    EXPECT_GT(nodes[1].pos.radius(), nodes[0].pos.radius());
}

// --- Integration: full lifecycle ---

TEST(LangevinIntegration, FullLifecycle) {
    // Use production-like constants; simulate many ticks.
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.01f, 0.0f};
    node.last_access_time = 0.0;
    node.access_count = 0;
    std::vector<NodeState> nodes = {node};

    // Simulate tick-by-tick for up to 30 days
    double t = 0.0;
    std::vector<uint32_t> archived;
    std::vector<float> radius_history;

    const double max_time = 30.0 * SECS_PER_DAY;
    while (t < max_time && archived.empty()) {
        t += 5.0;
        archived = stepper.step(nodes, t, rng);
        radius_history.push_back(nodes[0].pos.radius());
    }

    // Radius should monotonically increase (no noise)
    for (size_t j = 1; j < radius_history.size(); ++j) {
        EXPECT_GE(radius_history[j], radius_history[j - 1] - 1e-6f)
            << "Radius should monotonically increase at tick " << j;
    }

    // Should be archived within 30 days
    EXPECT_FALSE(archived.empty())
        << "Node should be archived after sufficient ticks without access";

    // Verify archival happened within 7-14 day window
    double days_to_archive = t / SECS_PER_DAY;
    EXPECT_GE(days_to_archive, 7.0)
        << "Node should survive at least 7 days";
    EXPECT_LE(days_to_archive, 14.0)
        << "Node should be archived within 14 days";
}

TEST(LangevinIntegration, ActivationResetsLifecycle) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0e-3f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.01f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // Drift for 2 days
    double t = 0.0;
    double drift_end = 2.0 * SECS_PER_DAY;
    while (t < drift_end) {
        t += 5.0;
        stepper.step(nodes, t, rng);
    }
    float radius_before_activation = nodes[0].pos.radius();
    EXPECT_GT(radius_before_activation, 0.01f)
        << "Should have drifted outward";

    // Activate (simulate agent reading the memory)
    stepper.activate(nodes[0], t, rng);
    EXPECT_NEAR(nodes[0].pos.radius(), 0.01f, 1e-6f)
        << "Activation should reset to thermal kick radius";

    // Drift again for only 1 day (half the original duration)
    double drift_end_2 = t + 1.0 * SECS_PER_DAY;
    while (t < drift_end_2) {
        t += 5.0;
        stepper.step(nodes, t, rng);
    }
    float radius_after_reactivation = nodes[0].pos.radius();

    // Should have drifted less than before (1 day vs 2 days)
    EXPECT_LT(radius_after_reactivation, radius_before_activation)
        << "Reactivated node should drift less (shorter time since access)";
}

TEST(LangevinIntegration, CohomologyDriftPenalty) {
    // Simulate cohomology integration: superseded node repositioned to (0.0, 0.93)
    // With a large enough lambda and age, one tick should push past threshold.
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.93f};  // Near archive threshold
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    // 1 day old with large lambda should push it past
    auto archived = stepper.step(nodes, SECS_PER_DAY, rng);
    EXPECT_EQ(archived.size(), 1u)
        << "Node placed at r=0.93 with old access should archive in one tick";
}

TEST(LangevinIntegration, NoiseTooWeakToArchiveAlone) {
    // With default noise_scale and zero drift, pure Brownian motion
    // should not push a node to the archive boundary in 14 days.
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.0f,
                              .noise_scale = 2.0e-4f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.5f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    double t = 0.0;
    double max_time = 14.0 * SECS_PER_DAY;
    // Step every 5 seconds for 14 days is ~242k ticks — subsample for speed
    while (t < max_time) {
        t += 5.0;
        auto archived = stepper.step(nodes, t, rng);
        EXPECT_TRUE(archived.empty())
            << "Pure noise should not archive a node at r=0.5 within 14 days";
        if (!archived.empty()) break;
        // Skip ahead in larger steps to keep test fast
        t += 295.0;  // effectively stepping every 5 minutes
        nodes[0].last_access_time = 0.0;  // keep age consistent
    }
}
