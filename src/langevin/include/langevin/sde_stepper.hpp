#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include <langevin/poincare_disk.hpp>

namespace slm::langevin {

/// Seconds per day — used to convert age from seconds to days.
inline constexpr float SECONDS_PER_DAY = 86400.0f;

/// Riemannian Langevin integrator on the Poincaré disk.
class LangevinStepper {
public:
    struct Config {
        float dt;                  // tick interval in seconds (e.g., 5.0)
        float lambda_decay;        // outward drift rate (per day²·second)
        float noise_scale;         // diffusion intensity
        float archive_threshold;   // radius threshold for archiving (e.g., 0.95)
        float thermal_kick_radius; // initial offset on activation (e.g., 0.01)
    };

    explicit LangevinStepper(Config config);

    /// Advance all nodes by one tick. Returns indices of newly archived nodes.
    std::vector<uint32_t> step(
        std::span<NodeState> nodes,
        double current_time,
        std::mt19937& rng
    ) const;

    /// Reset a node to the disk center with a small random thermal kick
    /// so it has a drift direction. Kick radius is taken from config.
    void activate(NodeState& node, double current_time,
                  std::mt19937& rng) const;

    const Config& config() const { return config_; }

private:
    Config config_;
};

} // namespace slm::langevin
