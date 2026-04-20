#include <sheaf/coboundary.hpp>
#include <cmath>
#include <cassert>

namespace slm::sheaf {

CoboundaryResult CoboundaryOperator::compute_local(
    const Neighborhood& hood,
    float threshold
) const {
    CoboundaryResult result;

    if (hood.edges.empty()) {
        return result;
    }

    float total_weighted_sq = 0.0f;

    for (const auto& edge : hood.edges) {
        assert(edge.neighbor_index < hood.neighbor_mus.size());
        const auto& mu_neighbor = hood.neighbor_mus[edge.neighbor_index];
        const auto& relation = edge.relation;

        uint32_t dim = static_cast<uint32_t>(hood.new_node_mu.size());
        assert(mu_neighbor.size() == dim);
        assert(relation.size() == dim);

        float edge_sq_norm = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float delta = hood.new_node_mu[d] - mu_neighbor[d] - relation[d];
            edge_sq_norm += delta * delta;
        }

        float edge_norm = std::sqrt(edge_sq_norm);

        float weight = (edge.type == EdgeType::Structural) ? 2.0f : 1.0f;
        total_weighted_sq += weight * edge_sq_norm;

        if (edge_norm > threshold) {
            result.conflicting.push_back(edge.neighbor_index);
        }
    }

    result.norm = std::sqrt(total_weighted_sq);
    return result;
}

} // namespace slm::sheaf
