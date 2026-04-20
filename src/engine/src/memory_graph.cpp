#include <engine/memory_graph.hpp>
#include <algorithm>
#include <cassert>

namespace slm::engine {

uint32_t MemoryGraph::insert(std::vector<float> mu, std::vector<float> sigma,
                              std::string text, uint32_t parent_id,
                              uint8_t depth, langevin::NodeState state) {
    uint32_t id = next_id_++;
    size_t idx = ids_.size();

    ids_.push_back(id);
    states_.push_back(state);
    mus_.push_back(std::move(mu));
    sigmas_.push_back(std::move(sigma));
    parent_ids_.push_back(parent_id);
    depths_.push_back(depth);
    texts_.push_back(std::move(text));
    annotations_.emplace_back();

    id_to_index_[id] = idx;
    return id;
}

void MemoryGraph::remove(uint32_t id) {
    auto it = id_to_index_.find(id);
    assert(it != id_to_index_.end());
    size_t idx = it->second;
    size_t last = ids_.size() - 1;

    if (idx != last) {
        uint32_t last_id = ids_[last];
        ids_[idx] = ids_[last];
        states_[idx] = states_[last];
        mus_[idx] = std::move(mus_[last]);
        sigmas_[idx] = std::move(sigmas_[last]);
        parent_ids_[idx] = parent_ids_[last];
        depths_[idx] = depths_[last];
        texts_[idx] = std::move(texts_[last]);
        annotations_[idx] = std::move(annotations_[last]);
        id_to_index_[last_id] = idx;
    }

    ids_.pop_back();
    states_.pop_back();
    mus_.pop_back();
    sigmas_.pop_back();
    parent_ids_.pop_back();
    depths_.pop_back();
    texts_.pop_back();
    annotations_.pop_back();
    id_to_index_.erase(id);
}

size_t MemoryGraph::index_of(uint32_t id) const {
    auto it = id_to_index_.find(id);
    assert(it != id_to_index_.end());
    return it->second;
}

langevin::NodeState& MemoryGraph::state(uint32_t id) { return states_[index_of(id)]; }
const langevin::NodeState& MemoryGraph::state(uint32_t id) const { return states_[index_of(id)]; }
std::span<const float> MemoryGraph::mu(uint32_t id) const { return mus_[index_of(id)]; }
std::span<const float> MemoryGraph::sigma(uint32_t id) const { return sigmas_[index_of(id)]; }
const std::string& MemoryGraph::text(uint32_t id) const { return texts_[index_of(id)]; }
const std::string& MemoryGraph::annotation(uint32_t id) const { return annotations_[index_of(id)]; }
void MemoryGraph::set_annotation(uint32_t id, std::string ann) { annotations_[index_of(id)] = std::move(ann); }
uint32_t MemoryGraph::parent_id(uint32_t id) const { return parent_ids_[index_of(id)]; }
uint8_t MemoryGraph::depth(uint32_t id) const { return depths_[index_of(id)]; }

std::span<langevin::NodeState> MemoryGraph::all_states() { return states_; }
std::span<const uint32_t> MemoryGraph::all_ids() const { return ids_; }

std::vector<uint32_t> MemoryGraph::siblings(uint32_t parent) const {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (parent_ids_[i] == parent) {
            result.push_back(ids_[i]);
        }
    }
    return result;
}

std::optional<uint32_t> MemoryGraph::parent(uint32_t node_id) const {
    uint32_t pid = parent_ids_[index_of(node_id)];
    if (pid == 0) return std::nullopt;
    if (id_to_index_.contains(pid)) return pid;
    return std::nullopt;
}

MemoryGraph::NodeSnapshot MemoryGraph::snapshot(uint32_t id) const {
    size_t idx = index_of(id);
    return {
        .id = id,
        .parent_id = parent_ids_[idx],
        .depth = depths_[idx],
        .text = texts_[idx],
        .mu = mus_[idx],
        .sigma = sigmas_[idx],
        .access_count = states_[idx].access_count,
        .pos_x = states_[idx].pos.x,
        .pos_y = states_[idx].pos.y,
        .last_access = states_[idx].last_access_time,
        .annotation = annotations_[idx],
    };
}

void MemoryGraph::insert_from_snapshot(const NodeSnapshot& snap) {
    size_t idx = ids_.size();
    ids_.push_back(snap.id);
    states_.push_back(langevin::NodeState{
        .pos = {snap.pos_x, snap.pos_y},
        .last_access_time = snap.last_access,
        .access_count = snap.access_count,
    });
    mus_.push_back(snap.mu);
    sigmas_.push_back(snap.sigma);
    parent_ids_.push_back(snap.parent_id);
    depths_.push_back(snap.depth);
    texts_.push_back(snap.text);
    annotations_.push_back(snap.annotation);
    id_to_index_[snap.id] = idx;
    if (snap.id >= next_id_) {
        next_id_ = snap.id + 1;
    }
}

} // namespace slm::engine
