#include <langevin/sde_stepper.hpp>
#include <cmath>
#include <numbers>

namespace slm::langevin {

LangevinStepper::LangevinStepper(Config config)
    : config_(config) {}

void LangevinStepper::activate(NodeState& node, double current_time,
                                std::mt19937& rng) const {
    // Small thermal kick at random angle so the node has a drift direction
    // and doesn't get stuck at the origin singularity.
    std::uniform_real_distribution<float> angle_dist(
        0.0f, 2.0f * std::numbers::pi_v<float>);
    float theta = angle_dist(rng);
    node.pos = {config_.thermal_kick_radius * std::cos(theta),
                config_.thermal_kick_radius * std::sin(theta)};
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

        // Time since last access, converted to days so that drift
        // accumulates over a human-scale working-memory lifespan
        // rather than racing to the boundary in seconds.
        float age_days = static_cast<float>(
            current_time - node.last_access_time) / SECONDS_PER_DAY;

        // Gradient of U(p) = -lambda * age_days * r
        // nabla_U = -lambda * age_days * (p / r)  (radial gradient)
        // Drift = -g_inv * nabla_U * dt = g_inv * lambda * age_days * (p/r) * dt
        float drift_mag = g_inv * config_.lambda_decay * age_days * config_.dt / r;
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
