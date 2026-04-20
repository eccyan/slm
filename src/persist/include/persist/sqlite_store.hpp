#pragma once

#include <filesystem>
#include <persist/store.hpp>
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

private:
    sqlite3* db_{nullptr};
    void create_schema();
    void exec(const char* sql);
};

} // namespace slm::persist
