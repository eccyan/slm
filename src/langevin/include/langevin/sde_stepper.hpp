#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include <langevin/poincare_disk.hpp>

namespace slm::langevin {

/// Riemannian Langevin integrator on the Poincaré disk.
class LangevinStepper {
public:
    struct Config {
        float dt;                  // tick interval in seconds (e.g., 5.0)
        float lambda_decay;        // outward drift rate
        float noise_scale;         // diffusion intensity
        float archive_threshold;   // radius threshold for archiving (e.g., 0.95)
    };

    explicit LangevinStepper(Config config);

    /// Advance all nodes by one tick. Returns indices of newly archived nodes.
    std::vector<uint32_t> step(
        std::span<NodeState> nodes,
        double current_time,
        std::mt19937& rng
    ) const;

    /// Reset a node to the center of the disk (on access/activation).
    static void activate(NodeState& node, double current_time);

    const Config& config() const { return config_; }

private:
    Config config_;
};

} // namespace slm::langevin
