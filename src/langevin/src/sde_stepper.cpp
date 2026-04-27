#include <langevin/sde_stepper.hpp>
#include <cmath>
#include <numbers>

namespace slm::langevin {

LangevinStepper::LangevinStepper(Config config)
    : config_(config) {}

void LangevinStepper::activate(NodeState& node, uint64_t current_tick,
                                std::mt19937& rng) const {
    // Small thermal kick at random angle so the node has a drift direction
    // and doesn't get stuck at the origin singularity.
    std::uniform_real_distribution<float> angle_dist(
        0.0f, 2.0f * std::numbers::pi_v<float>);
    float theta = angle_dist(rng);
    node.pos = {config_.thermal_kick_radius * std::cos(theta),
                config_.thermal_kick_radius * std::sin(theta)};
    node.last_access_tick = current_tick;
    node.access_count += 1;
}

std::vector<uint32_t> LangevinStepper::step(
    std::span<NodeState> nodes,
    uint64_t current_tick,
    uint64_t delta_ticks,
    std::mt19937& rng
) const {
    std::vector<uint32_t> archived;
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    // Euler-Maruyama scaling by elapsed cognitive ticks
    float dt_eff = config_.dt * static_cast<float>(delta_ticks);

    for (uint32_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        float r = node.pos.radius();

        // Skip nodes at exact origin (no drift direction)
        if (r < 1e-8f) {
            continue;
        }

        float g_inv = inverse_metric(node.pos);

        // Age in cognitive ticks since last access
        float age_ticks = static_cast<float>(
            current_tick - node.last_access_tick);

        // Gradient of U(p) = -lambda * age_ticks * r
        // Drift = g_inv * lambda * age_ticks * (p/r) * dt_eff
        float drift_mag = g_inv * config_.lambda_decay * age_ticks * dt_eff / r;
        float dx_drift = drift_mag * node.pos.x;
        float dy_drift = drift_mag * node.pos.y;

        // Noise: noise_scale * sqrt(2 * g_inv * dt_eff) * xi
        float noise_mag = config_.noise_scale * std::sqrt(2.0f * g_inv * dt_eff);
        float dx_noise = noise_mag * noise_dist(rng);
        float dy_noise = noise_mag * noise_dist(rng);

        // Euler-Maruyama update
        node.pos.x += dx_drift + dx_noise;
        node.pos.y += dy_drift + dy_noise;

        // Project back inside the disk
        node.pos = project_to_disk(node.pos);

        // Check for archival
        if (node.pos.radius() > config_.archive_threshold) {
            archived.push_back(i);
        }
    }

    return archived;
}

} // namespace slm::langevin
