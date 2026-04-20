#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <langevin/poincare_disk.hpp>
#include <metric/gaussian_node.hpp>

namespace slm::engine {

class MemoryGraph {
public:
    uint32_t insert(std::vector<float> mu, std::vector<float> sigma,
                    std::string text, uint32_t parent_id, uint8_t depth,
                    langevin::NodeState state);

    void remove(uint32_t id);

    uint32_t size() const { return static_cast<uint32_t>(ids_.size()); }
    bool contains(uint32_t id) const { return id_to_index_.contains(id); }

    langevin::NodeState& state(uint32_t id);
    const langevin::NodeState& state(uint32_t id) const;
    std::span<const float> mu(uint32_t id) const;
    std::span<const float> sigma(uint32_t id) const;
    const std::string& text(uint32_t id) const;
    const std::string& annotation(uint32_t id) const;
    void set_annotation(uint32_t id, std::string ann);
    uint32_t parent_id(uint32_t id) const;
    uint8_t depth(uint32_t id) const;

    std::span<langevin::NodeState> all_states();
    std::span<const uint32_t> all_ids() const;

    std::vector<uint32_t> siblings(uint32_t parent) const;
    std::optional<uint32_t> parent(uint32_t node_id) const;

    struct NodeSnapshot {
        uint32_t id;
        uint32_t parent_id;
        uint8_t depth;
        std::string text;
        std::vector<float> mu;
        std::vector<float> sigma;
        uint32_t access_count;
        float pos_x;
        float pos_y;
        double last_access;
        std::string annotation;
    };

    NodeSnapshot snapshot(uint32_t id) const;
    void insert_from_snapshot(const NodeSnapshot& snap);

private:
    std::vector<langevin::NodeState> states_;
    std::vector<std::vector<float>>  mus_;
    std::vector<std::vector<float>>  sigmas_;
    std::vector<uint32_t>            parent_ids_;
    std::vector<uint8_t>             depths_;
    std::vector<std::string>         texts_;
    std::vector<std::string>         annotations_;
    std::vector<uint32_t>            ids_;
    std::unordered_map<uint32_t, size_t> id_to_index_;
    uint32_t next_id_{1};

    size_t index_of(uint32_t id) const;
};

} // namespace slm::engine
