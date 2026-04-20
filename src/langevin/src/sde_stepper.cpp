#include <langevin/sde_stepper.hpp>
#include <cmath>

namespace slm::langevin {

LangevinStepper::LangevinStepper(Config config)
    : config_(config) {}

void LangevinStepper::activate(NodeState& node, double current_time) {
    node.pos = {0.0f, 0.0f};
    node.last_access_time = current_time;
    node.access_count += 1;
}

std::vector<uint32_t> LangevinStepper::step(
    std::span<NodeState> nodes,
    double current_time,
    std::mt19937& rng
) const {
    std::vector<uint32_t> archived;
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        float r = node.pos.radius();

        // Skip nodes at exact origin (no drift direction)
        if (r < 1e-8f) {
            continue;
        }

        float g_inv = inverse_metric(node.pos);

        // Time since last access drives the outward potential
        float delta_t = static_cast<float>(current_time - node.last_access_time);

        // Gradient of U(p) = -lambda * delta_t * r
        // nabla_U = -lambda * delta_t * (p / r)  (radial gradient)
        // Drift = -g_inv * nabla_U * dt = g_inv * lambda * delta_t * (p/r) * dt
        float drift_mag = g_inv * config_.lambda_decay * delta_t * config_.dt / r;
        float dx_drift = drift_mag * node.pos.x;
        float dy_drift = drift_mag * node.pos.y;

        // Noise: sqrt(2 * g_inv * dt) * xi
        float noise_mag = config_.noise_scale * std::sqrt(2.0f * g_inv * config_.dt);
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
