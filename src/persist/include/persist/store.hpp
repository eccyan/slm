#pragma once

#include <engine/memory_graph.hpp>

namespace slm::persist {

class Store {
public:
    virtual ~Store() = default;
    virtual void checkpoint(const engine::MemoryGraph& graph) = 0;
    virtual void flush(const engine::MemoryGraph& graph) = 0;
    virtual void load(engine::MemoryGraph& graph) = 0;
    virtual void archive_node(const engine::MemoryGraph::NodeSnapshot& snap) = 0;
};

} // namespace slm::persist
