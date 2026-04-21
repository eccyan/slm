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
    LangevinStepper langevin{{.dt = 5.0f, .lambda_decay = 5.0e-6f,
                               .noise_scale = 0.0f, .archive_threshold = 0.95f,
                               .thermal_kick_radius = 0.01f}};
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

    std::thread t([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.request_stop();
    t.join();

    EXPECT_EQ(graph.size(), 1u);

    auto ids = graph.all_ids();
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(graph.text(ids[0]), "Hello from the agent");
    EXPECT_EQ(graph.parent_id(ids[0]), 0u);

    // Node gets a thermal kick (r ≈ 0.01) instead of exact origin
    EXPECT_NEAR(graph.state(ids[0]).pos.radius(), 0.01f, 1e-6f);
}

TEST_F(SchedulerFixture, MultipleWriteCommits) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("First memory");
    submit_write("Second memory");
    submit_write("Third memory");

    std::thread t([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.request_stop();
    t.join();

    EXPECT_EQ(graph.size(), 3u);
}

TEST_F(SchedulerFixture, WriteCommitWithParentId) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("Parent node", 0, 0);

    std::thread t([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.request_stop();
    t.join();

    ASSERT_EQ(graph.size(), 1u);
    auto parent_id = graph.all_ids()[0];

    submit_write("Child node", parent_id, 1);

    std::thread t2([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.request_stop();
    t2.join();

    ASSERT_EQ(graph.size(), 2u);
    auto siblings = graph.siblings(parent_id);
    EXPECT_EQ(siblings.size(), 1u);
}

TEST_F(SchedulerFixture, GracefulShutdownFlushes) {
    Scheduler scheduler(*slab, slab->cmd_queue(), graph, metric, sheaf,
                         langevin, *store, Scheduler::Config{});

    submit_write("Persisted on shutdown");

    std::thread t([&] { scheduler.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.request_stop();
    t.join();

    MemoryGraph loaded;
    store->load(loaded);
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded.text(loaded.all_ids()[0]), "Persisted on shutdown");
}

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
