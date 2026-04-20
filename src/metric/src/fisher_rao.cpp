#include <metric/fisher_rao.hpp>
#include <metric/simd_ops.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace slm::metric {

float FisherRaoMetric::distance(const GaussianNode& p, const GaussianNode& q) const {
    assert(p.mu.size() == q.mu.size());
    assert(p.sigma.size() == q.sigma.size());
    assert(p.mu.size() == p.sigma.size());

    uint32_t dim = static_cast<uint32_t>(p.mu.size());

    float var_div = simd_variance_divergence(
        p.sigma.data(), q.sigma.data(), dim
    );

    float weighted_diff = simd_weighted_sq_diff(
        p.mu.data(), q.mu.data(),
        p.sigma.data(), q.sigma.data(), dim
    );

    return std::sqrt(var_div + weighted_diff);
}

std::vector<uint32_t> FisherRaoMetric::top_k(
    const GaussianNode& query,
    std::span<const GaussianNode> candidates,
    uint32_t k
) const {
    if (k == 0 || candidates.empty()) {
        return {};
    }

    struct IndexDist {
        uint32_t index;
        float distance;
    };

    std::vector<IndexDist> scored;
    scored.reserve(candidates.size());
    for (uint32_t i = 0; i < candidates.size(); ++i) {
        scored.push_back({i, distance(query, candidates[i])});
    }

    uint32_t actual_k = std::min(k, static_cast<uint32_t>(scored.size()));
    std::partial_sort(
        scored.begin(),
        scored.begin() + actual_k,
        scored.end(),
        [](const IndexDist& a, const IndexDist& b) {
            return a.distance < b.distance;
        }
    );

    std::vector<uint32_t> result;
    result.reserve(actual_k);
    for (uint32_t i = 0; i < actual_k; ++i) {
        result.push_back(scored[i].index);
    }
    return result;
}

} // namespace slm::metric
