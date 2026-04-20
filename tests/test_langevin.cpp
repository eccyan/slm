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
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 100.0, rng);

    EXPECT_GT(nodes[0].pos.radius(), 0.3f)
        << "Node should drift outward when unaccessed";
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "Node should remain inside the disk";
}

TEST(LangevinStep, RecentlyAccessedDriftsLess) {
    LangevinStepper stepper({.dt = 0.1f, .lambda_decay = 0.01f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});

    NodeState node_a{};
    node_a.pos = {0.3f, 0.0f};
    node_a.last_access_time = 0.0;

    NodeState node_b{};
    node_b.pos = {0.3f, 0.0f};
    node_b.last_access_time = 90.0;

    std::vector<NodeState> nodes_a = {node_a};
    std::vector<NodeState> nodes_b = {node_b};
    std::mt19937 rng_a(42), rng_b(42);

    stepper.step(nodes_a, 100.0, rng_a);
    stepper.step(nodes_b, 100.0, rng_b);

    EXPECT_GT(nodes_a[0].pos.radius(), nodes_b[0].pos.radius())
        << "Node accessed longer ago should drift more";
}

TEST(LangevinStep, NodeAtOriginDriftsNowhere) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 0.1f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 100.0, rng);
    EXPECT_FLOAT_EQ(nodes[0].pos.radius(), 0.0f);
}

TEST(LangevinStep, ArchivesNodesBeyondThreshold) {
    LangevinStepper stepper({.dt = 5.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.94f, 0.0f};
    node.last_access_time = 0.0;
    std::vector<NodeState> nodes = {node};

    auto archived = stepper.step(nodes, 10000.0, rng);
    EXPECT_EQ(archived.size(), 1u);
    EXPECT_EQ(archived[0], 0u);
}

TEST(LangevinStep, StaysInsideDisk) {
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

    std::vector<NodeState> nodes_a = {node};
    std::vector<NodeState> nodes_b = {node};
    std::mt19937 rng_a(42), rng_b(99);

    stepper.step(nodes_a, 100.0, rng_a);
    stepper.step(nodes_b, 100.0, rng_b);

    bool positions_differ = (nodes_a[0].pos.x != nodes_b[0].pos.x)
                         || (nodes_a[0].pos.y != nodes_b[0].pos.y);
    EXPECT_TRUE(positions_differ)
        << "Different RNG seeds should produce different positions";
}

TEST(LangevinStep, MultipleNodesIndependent) {
    LangevinStepper stepper({.dt = 0.1f, .lambda_decay = 0.01f,
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

    EXPECT_GT(nodes[0].pos.radius(), 0.2f);
    EXPECT_GT(nodes[1].pos.radius(), 0.5f);
    EXPECT_GT(nodes[2].pos.radius(), 0.8f);

    EXPECT_GT(nodes[2].pos.radius(), nodes[1].pos.radius());
    EXPECT_GT(nodes[1].pos.radius(), nodes[0].pos.radius());
}
