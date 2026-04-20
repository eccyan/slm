#include <engine/scheduler.hpp>
#include <slab/header.hpp>
#include <metric/gaussian_node.hpp>
#include <cstring>
#include <string>
#include <string_view>

namespace slm::engine {

Scheduler::Scheduler(
    slab::SlabAllocator& slab,
    slab::SPSCRingBuffer<uint32_t, 256>& queue,
    MemoryGraph& graph,
    metric::FisherRaoMetric& metric,
    sheaf::CoboundaryOperator& sheaf,
    langevin::LangevinStepper& langevin,
    persist::Store& persist,
    Config config
)
    : slab_(slab), queue_(queue), graph_(graph), metric_(metric),
      sheaf_(sheaf), langevin_(langevin), persist_(persist), config_(config)
{}

double Scheduler::current_time() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

void Scheduler::run() {
    stop_requested_.store(false, std::memory_order_relaxed);
    start_time_ = std::chrono::steady_clock::now();
    last_tier3_tick_ = start_time_;
    last_checkpoint_ = start_time_;

    while (!stop_requested_.load(std::memory_order_acquire)) {
        process_tier1();

        auto now = std::chrono::steady_clock::now();

        if (!cohomology_pending_.empty()) {
            process_tier2();
        }

        now = std::chrono::steady_clock::now();
        if (now - last_tier3_tick_ >= config_.tier3_tick_interval) {
            process_tier3();
            last_tier3_tick_ = now;
        }

        now = std::chrono::steady_clock::now();
        if (now - last_checkpoint_ >= config_.checkpoint_interval) {
            checkpoint();
            last_checkpoint_ = now;
        }

        if (cohomology_pending_.empty() && !queue_.peek()) {
            std::this_thread::sleep_for(config_.tier1_poll_interval);
        }
    }

    // Graceful shutdown: drain + flush
    process_tier1();
    persist_.flush(graph_);
}

void Scheduler::process_tier1() {
    uint32_t handle;
    while (queue_.try_pop(handle)) {
        uint8_t cmd = slab::decode_command(handle);
        uint32_t slab_idx = slab::decode_slab_index(handle);

        switch (cmd) {
        case slab::CMD_WRITE_COMMIT:
            handle_write_commit(slab_idx);
            break;
        case slab::CMD_READ:
            handle_read(slab_idx);
            break;
        }

        slab_.release(slab_idx);
    }
}

void Scheduler::handle_write_commit(uint32_t slab_idx) {
    auto span = slab_.get(slab_idx);
    const auto& hdr = slab_.header(slab_idx);

    std::string text(
        reinterpret_cast<const char*>(span.data() + hdr.text_offset),
        hdr.text_length
    );

    const float* vec_ptr = reinterpret_cast<const float*>(
        span.data() + hdr.vector_offset);
    std::vector<float> mu(vec_ptr, vec_ptr + hdr.vector_dim);

    std::vector<float> sigma(hdr.vector_dim, metric::SIGMA_MAX);

    langevin::NodeState state{};
    state.pos = {0.0f, 0.0f};
    state.last_access_time = current_time();
    state.access_count = 0;

    auto node_id = graph_.insert(
        std::move(mu), std::move(sigma), std::move(text),
        hdr.parent_id, hdr.depth, state
    );

    cohomology_pending_.push_back(node_id);
}

void Scheduler::handle_read(uint32_t slab_idx) {
    auto span = slab_.get(slab_idx);
    const auto& hdr = slab_.header(slab_idx);

    std::string result;
    for (auto id : graph_.all_ids()) {
        if (graph_.state(id).pos.radius() < config_.active_radius) {
            result += graph_.text(id);
            result += "\n";
            const auto& ann = graph_.annotation(id);
            if (!ann.empty()) {
                result += ann;
                result += "\n";
            }
        }
    }

    if (hdr.vector_dim > 0) {
        const float* query_ptr = reinterpret_cast<const float*>(
            span.data() + hdr.vector_offset);

        std::vector<metric::GaussianNode> candidates;
        std::vector<uint32_t> candidate_ids;
        for (auto id : graph_.all_ids()) {
            candidates.push_back({graph_.mu(id), graph_.sigma(id),
                                  graph_.state(id).access_count});
            candidate_ids.push_back(id);
        }

        if (!candidates.empty()) {
            std::vector<float> query_sigma(hdr.vector_dim, metric::SIGMA_MAX);
            metric::GaussianNode query{
                std::span<const float>(query_ptr, hdr.vector_dim),
                query_sigma, 0
            };

            auto top_indices = metric_.top_k(query, candidates,
                                              config_.search_top_k);

            for (auto idx : top_indices) {
                auto id = candidate_ids[idx];
                langevin::LangevinStepper::activate(graph_.state(id),
                                                     current_time());
                graph_.state(id).access_count++;

                if (graph_.state(id).pos.radius() >= config_.active_radius) {
                    result += graph_.text(id);
                    result += "\n";
                }
            }
        }
    }

    auto* resp_hdr = reinterpret_cast<slab::MemoryFSHeader*>(span.data());
    resp_hdr->magic = slab::MEMFS_DONE;
    resp_hdr->text_offset = 64;
    resp_hdr->text_length = static_cast<uint32_t>(result.size());

    uint32_t max_text = static_cast<uint32_t>(slab_.get(slab_idx).size()) - 64;
    uint32_t copy_len = std::min(resp_hdr->text_length, max_text);
    std::memcpy(span.data() + 64, result.data(), copy_len);
    resp_hdr->text_length = copy_len;
}

void Scheduler::process_tier2() {
    auto deadline = std::chrono::steady_clock::now() + config_.tier2_time_budget;

    while (!cohomology_pending_.empty()
           && std::chrono::steady_clock::now() < deadline) {
        auto node_id = cohomology_pending_.front();
        cohomology_pending_.pop_front();

        if (!graph_.contains(node_id)) continue;

        sheaf::Neighborhood hood;
        hood.new_node_mu = graph_.mu(node_id);
        hood.new_node_text = graph_.text(node_id);

        auto sibs = graph_.siblings(graph_.parent_id(node_id));
        for (auto sib_id : sibs) {
            if (sib_id == node_id) continue;

            uint32_t neighbor_idx = static_cast<uint32_t>(
                hood.neighbor_mus.size());
            auto sib_mu = graph_.mu(sib_id);
            hood.neighbor_mus.emplace_back(sib_mu.begin(), sib_mu.end());
            hood.neighbor_texts.push_back(graph_.text(sib_id));

            std::vector<float> zero_rel(sib_mu.size(), 0.0f);
            hood.edges.push_back({neighbor_idx,
                                  sheaf::EdgeType::Structural,
                                  std::move(zero_rel)});
        }

        auto result = sheaf_.compute_local(hood,
                                            config_.contradiction_threshold);

        if (result.norm > config_.contradiction_threshold) {
            for (auto neighbor_idx : result.conflicting) {
                auto& sibs_list = sibs;
                uint32_t actual_idx = 0;
                for (auto sib_id : sibs_list) {
                    if (sib_id == node_id) continue;
                    if (actual_idx == neighbor_idx) {
                        graph_.state(sib_id).pos = {0.0f, 0.93f};
                        break;
                    }
                    actual_idx++;
                }
            }

            sheaf::Annotation ann;
            if (!result.conflicting.empty()) {
                uint32_t first_conflict = result.conflicting[0];
                if (first_conflict < hood.neighbor_texts.size()) {
                    ann.superseded_text = hood.neighbor_texts[first_conflict];
                }
            }
            ann.superseding_text = graph_.text(node_id);
            ann.delta_norm = result.norm;
            graph_.set_annotation(node_id, sheaf::format_annotation(ann));
        }

        if (queue_.peek()) break;
    }
}

void Scheduler::process_tier3() {
    auto archived = langevin_.step(graph_.all_states(), current_time(), rng_);

    std::vector<uint32_t> archive_ids;
    auto all_ids = graph_.all_ids();
    for (auto state_idx : archived) {
        if (state_idx < all_ids.size()) {
            archive_ids.push_back(all_ids[state_idx]);
        }
    }

    for (auto id : archive_ids) {
        auto snap = graph_.snapshot(id);
        persist_.archive_node(snap);
        graph_.remove(id);
    }
}

void Scheduler::checkpoint() {
    persist_.checkpoint(graph_);
}

} // namespace slm::engine
