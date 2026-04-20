#pragma once

#include <cmath>
#include <cstdint>

namespace slm::langevin {

/// Position on the Poincaré disk (radius < 1).
struct DiskPosition {
    float x{0.0f};
    float y{0.0f};

    float radius() const { return std::sqrt(x * x + y * y); }
};

/// State of a memory node on the Poincaré disk.
struct NodeState {
    DiskPosition pos;
    double last_access_time{0.0};
    uint32_t access_count{0};
};

/// Inverse metric tensor g^{-1}(p) = (1 - r^2)^2 / 4.
inline float inverse_metric(const DiskPosition& p) {
    float r2 = p.x * p.x + p.y * p.y;
    float factor = 1.0f - r2;
    return (factor * factor) / 4.0f;
}

/// Project a position back inside the Poincaré disk.
/// If r >= 1, scales to r = 0.999 preserving direction.
inline DiskPosition project_to_disk(DiskPosition p) {
    constexpr float MAX_RADIUS = 0.999f;
    float r = p.radius();
    if (r >= 1.0f) {
        float scale = MAX_RADIUS / r;
        return {p.x * scale, p.y * scale};
    }
    return p;
}

} // namespace slm::langevin
