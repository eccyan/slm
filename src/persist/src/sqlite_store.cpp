#include <persist/sqlite_store.hpp>
#include <algorithm>
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
    // Detect legacy schema (last_access REAL → last_access_tick INTEGER).
    // If the old column exists, drop the table and recreate with new schema.
    sqlite3_stmt* pragma = nullptr;
    sqlite3_prepare_v2(db_, "PRAGMA table_info(memory_nodes)", -1, &pragma, nullptr);
    bool has_legacy_column = false;
    while (sqlite3_step(pragma) == SQLITE_ROW) {
        const char* col = reinterpret_cast<const char*>(sqlite3_column_text(pragma, 1));
        if (col && std::string(col) == "last_access") {
            has_legacy_column = true;
            break;
        }
    }
    sqlite3_finalize(pragma);

    if (has_legacy_column) {
        exec("DROP TABLE IF EXISTS memory_nodes");
        exec("DROP TABLE IF EXISTS edges");
    }

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
            last_access_tick INTEGER NOT NULL DEFAULT 0,
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
    exec("DELETE FROM memory_nodes WHERE status = 0");

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "INSERT INTO memory_nodes "
        "(id, parent_id, depth, text, mu, sigma, access_count, "
        " pos_x, pos_y, last_access_tick, annotation, status) "
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
        sqlite3_bind_int64(stmt, 10, static_cast<sqlite3_int64>(snap.last_access_tick));
        if (snap.annotation.empty()) {
            sqlite3_bind_null(stmt, 11);
        } else {
            sqlite3_bind_text(stmt, 11, snap.annotation.c_str(), -1, SQLITE_TRANSIENT);
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
        "       pos_x, pos_y, last_access_tick, annotation "
        "FROM memory_nodes WHERE status = 0",
        -1, &stmt, nullptr);

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
        snap.last_access_tick = static_cast<uint64_t>(sqlite3_column_int64(stmt, 9));

        const char* ann = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
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
        " pos_x, pos_y, last_access_tick, annotation, status) "
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
    sqlite3_bind_int64(stmt, 10, static_cast<sqlite3_int64>(snap.last_access_tick));
    if (snap.annotation.empty()) {
        sqlite3_bind_null(stmt, 11);
    } else {
        sqlite3_bind_text(stmt, 11, snap.annotation.c_str(), -1, SQLITE_TRANSIENT);
    }

    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::vector<engine::MemoryGraph::NodeSnapshot> SqliteStore::retrieve_archived(
    const metric::GaussianNode& query,
    const metric::FisherRaoMetric& fr_metric,
    uint32_t k)
{
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "SELECT id, parent_id, depth, text, mu, sigma, access_count, "
        "       pos_x, pos_y, last_access_tick, annotation "
        "FROM memory_nodes WHERE status = 1",
        -1, &stmt, nullptr);

    std::vector<engine::MemoryGraph::NodeSnapshot> all;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        engine::MemoryGraph::NodeSnapshot snap;
        snap.id        = static_cast<uint32_t>(sqlite3_column_int(stmt, 0));
        snap.parent_id = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));
        snap.depth     = static_cast<uint8_t>(sqlite3_column_int(stmt, 2));

        const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        snap.text = text ? text : "";

        const float* mu_data = static_cast<const float*>(sqlite3_column_blob(stmt, 4));
        int mu_bytes = sqlite3_column_bytes(stmt, 4);
        snap.mu.assign(mu_data, mu_data + mu_bytes / static_cast<int>(sizeof(float)));

        const float* sigma_data = static_cast<const float*>(sqlite3_column_blob(stmt, 5));
        int sigma_bytes = sqlite3_column_bytes(stmt, 5);
        snap.sigma.assign(sigma_data, sigma_data + sigma_bytes / static_cast<int>(sizeof(float)));

        snap.access_count = static_cast<uint32_t>(sqlite3_column_int(stmt, 6));
        snap.pos_x        = static_cast<float>(sqlite3_column_double(stmt, 7));
        snap.pos_y        = static_cast<float>(sqlite3_column_double(stmt, 8));
        snap.last_access_tick = static_cast<uint64_t>(sqlite3_column_int64(stmt, 9));

        const char* ann = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        snap.annotation = ann ? ann : "";

        all.push_back(std::move(snap));
    }
    sqlite3_finalize(stmt);

    if (all.empty()) return {};

    // Build GaussianNode views and score with Fisher-Rao distance
    std::vector<metric::GaussianNode> candidates;
    candidates.reserve(all.size());
    for (const auto& snap : all) {
        candidates.push_back(metric::GaussianNode{
            std::span<const float>(snap.mu),
            std::span<const float>(snap.sigma),
            snap.access_count
        });
    }

    // Compute distances and sort indices
    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(all.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        float d = fr_metric.distance(query, candidates[i]);
        scored.emplace_back(d, i);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    std::size_t count = std::min(static_cast<std::size_t>(k), scored.size());
    std::vector<engine::MemoryGraph::NodeSnapshot> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(std::move(all[scored[i].second]));
    }
    return result;
}

void SqliteStore::reactivate_node(uint32_t node_id, float pos_x, float pos_y,
                                   uint64_t tick) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_,
        "UPDATE memory_nodes SET status = 0, pos_x = ?, pos_y = ?, "
        "last_access_tick = ? WHERE id = ? AND status = 1",
        -1, &stmt, nullptr);
    if (rc != SQLITE_OK || !stmt) return;
    sqlite3_bind_double(stmt, 1, static_cast<double>(pos_x));
    sqlite3_bind_double(stmt, 2, static_cast<double>(pos_y));
    sqlite3_bind_int64(stmt, 3, static_cast<sqlite3_int64>(tick));
    sqlite3_bind_int(stmt, 4, static_cast<int>(node_id));
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

uint64_t SqliteStore::max_tick() {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
        "SELECT MAX(last_access_tick) FROM memory_nodes",
        -1, &stmt, nullptr);

    uint64_t result = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        result = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
    }
    sqlite3_finalize(stmt);
    return result;
}

} // namespace slm::persist
