#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <metric/gaussian_node.hpp>

namespace slm::metric {

/// Fisher-Rao geodesic distance metric on diagonal Gaussian distributions.
class FisherRaoMetric {
public:
    /// Compute the Fisher-Rao geodesic distance between two Gaussian nodes.
    float distance(const GaussianNode& p, const GaussianNode& q) const;

    /// Find the top-k nearest candidates to a query, ranked by ascending distance.
    std::vector<uint32_t> top_k(
        const GaussianNode& query,
        std::span<const GaussianNode> candidates,
        uint32_t k
    ) const;
};

} // namespace slm::metric
