#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace slm::metric {

/// Maximum sigma (high uncertainty, fresh node with 0 accesses).
inline constexpr float SIGMA_MAX = 10.0f;

/// Minimum sigma (low uncertainty, well-accessed node with 10+ accesses).
inline constexpr float SIGMA_MIN = 0.1f;

/// Number of accesses to complete the sigma ramp from SIGMA_MAX to SIGMA_MIN.
inline constexpr uint32_t RAMP_STEPS = 10;

/// Compute the per-component sigma value for a given access count.
inline float compute_sigma_component(uint32_t access_count) {
    float alpha = std::min(static_cast<float>(access_count) / RAMP_STEPS, 1.0f);
    return SIGMA_MAX * (1.0f - alpha) + SIGMA_MIN * alpha;
}

/// Fill a sigma vector with uniform sigma values for a given access count.
inline void fill_sigma(std::span<float> sigma, uint32_t access_count) {
    float s = compute_sigma_component(access_count);
    for (auto& v : sigma) {
        v = s;
    }
}

/// A memory node modeled as a diagonal Gaussian distribution.
struct GaussianNode {
    std::span<const float> mu;
    std::span<const float> sigma;
    uint32_t access_count;
};

} // namespace slm::metric
