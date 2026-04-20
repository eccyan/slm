#pragma once

#include <cstdint>

namespace slm::metric {

/// Compute sum of (mu_p[i] - mu_q[i])^2 / (sigma_p[i] * sigma_q[i])
float simd_weighted_sq_diff(
    const float* mu_p, const float* mu_q,
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);

/// Compute sum of (2 * ln(sigma_q[i] / sigma_p[i]))^2
float simd_variance_divergence(
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);

} // namespace slm::metric
