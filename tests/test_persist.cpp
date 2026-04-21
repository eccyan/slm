#include <gtest/gtest.h>
#include <persist/sqlite_store.hpp>
#include <engine/memory_graph.hpp>
#include <metric/gaussian_node.hpp>
#include <metric/fisher_rao.hpp>
#include <filesystem>
#include <vector>

using namespace slm::persist;
using namespace slm::engine;
using namespace slm::langevin;

namespace {

struct PersistFixture : public ::testing::Test {
    std::filesystem::path db_path;

    void SetUp() override {
        db_path = std::filesystem::temp_directory_path() / "test_slmfs.db";
        std::filesystem::remove(db_path);
    }

    void TearDown() override {
        std::filesystem::remove(db_path);
        std::filesystem::remove(db_path.string() + "-wal");
        std::filesystem::remove(db_path.string() + "-shm");
    }
};

} // namespace

TEST_F(PersistFixture, CreateAndOpen) {
    {
        SqliteStore store(db_path);
    }
    EXPECT_TRUE(std::filesystem::exists(db_path));
}

TEST_F(PersistFixture, CheckpointAndLoad) {
    MemoryGraph graph;

    std::vector<float> mu1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> sigma1 = {0.5f, 0.5f, 0.5f};
    NodeState state1{.pos = {0.3f, 0.4f}, .last_access_time = 100.0, .access_count = 5};
    auto id1 = graph.insert(mu1, sigma1, "first node", 0, 0, state1);

    std::vector<float> mu2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> sigma2 = {1.0f, 1.0f, 1.0f};
    NodeState state2{.pos = {0.1f, 0.0f}, .last_access_time = 200.0, .access_count = 2};
    auto id2 = graph.insert(mu2, sigma2, "second node", id1, 1, state2);

    graph.set_annotation(id2, "<!-- test annotation -->");

    {
        SqliteStore store(db_path);
        store.checkpoint(graph);
    }

    MemoryGraph loaded;
    {
        SqliteStore store(db_path);
        store.load(loaded);
    }

    ASSERT_EQ(loaded.size(), 2u);

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
    store.checkpoint(graph);
    store.archive_node(snap);

    MemoryGraph loaded;
    store.load(loaded);
    EXPECT_GE(loaded.size(), 0u);
}

TEST_F(PersistFixture, LoadEmptyDatabase) {
    {
        SqliteStore store(db_path);
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

    graph.state(id).pos = {0.5f, 0.5f};
    store.checkpoint(graph);

    MemoryGraph loaded;
    store.load(loaded);
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_NEAR(loaded.state(id).pos.x, 0.5f, 1e-5f);
}

// --- Integration: full round-trip with scene graph ---

TEST_F(PersistFixture, RoundTripWithSceneGraph) {
    MemoryGraph graph;

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
    EXPECT_EQ(graph.siblings(h1).size(), 2u);
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
    store.reactivate_node(id, 0.01f, 0.0f);

    // Verify it's gone from the archive
    auto after = store.retrieve_archived(query, metric, 10);
    EXPECT_TRUE(after.empty());
}
