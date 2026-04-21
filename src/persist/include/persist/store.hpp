#pragma once

#include <engine/memory_graph.hpp>
#include <metric/fisher_rao.hpp>

namespace slm::persist {

class Store {
public:
    virtual ~Store() = default;
    virtual void checkpoint(const engine::MemoryGraph& graph) = 0;
    virtual void flush(const engine::MemoryGraph& graph) = 0;
    virtual void load(engine::MemoryGraph& graph) = 0;
    virtual void archive_node(const engine::MemoryGraph::NodeSnapshot& snap) = 0;
    virtual std::vector<engine::MemoryGraph::NodeSnapshot> retrieve_archived(
        const metric::GaussianNode& query,
        const metric::FisherRaoMetric& metric,
        uint32_t k
    ) = 0;
    virtual void reactivate_node(uint32_t id, float pos_x, float pos_y) = 0;
};

} // namespace slm::persist
