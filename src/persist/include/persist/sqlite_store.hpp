#pragma once

#include <filesystem>
#include <vector>
#include <persist/store.hpp>
#include <metric/gaussian_node.hpp>
#include <metric/fisher_rao.hpp>
#include <sqlite3.h>

namespace slm::persist {

class SqliteStore : public Store {
public:
    explicit SqliteStore(const std::filesystem::path& db_path);
    ~SqliteStore() override;

    SqliteStore(const SqliteStore&) = delete;
    SqliteStore& operator=(const SqliteStore&) = delete;

    void checkpoint(const engine::MemoryGraph& graph) override;
    void flush(const engine::MemoryGraph& graph) override;
    void load(engine::MemoryGraph& graph) override;
    void archive_node(const engine::MemoryGraph::NodeSnapshot& snap) override;

    /// Retrieve top-k archived nodes ranked by Fisher-Rao distance to query.
    std::vector<engine::MemoryGraph::NodeSnapshot> retrieve_archived(
        const metric::GaussianNode& query,
        const metric::FisherRaoMetric& metric,
        uint32_t k) override;

    /// Remove an archived node (status=1) from the DB by id.
    void reactivate_node(uint32_t node_id, float pos_x, float pos_y) override;

private:
    sqlite3* db_{nullptr};
    void create_schema();
    void exec(const char* sql);
};

} // namespace slm::persist
