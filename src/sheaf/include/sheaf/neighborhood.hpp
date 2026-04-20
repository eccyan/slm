#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace slm::sheaf {

/// Edge type in the local knowledge graph.
enum class EdgeType : uint8_t {
    Structural = 0,  // parent-child or sibling (from scene graph)
    KNN = 1,         // cross-branch k-nearest neighbor
};

/// An edge from the new node to a neighbor in the local neighborhood.
struct Edge {
    uint32_t neighbor_index;          // index into Neighborhood::neighbor_mus
    EdgeType type;
    std::vector<float> relation;      // expected semantic difference r_ij
};

/// Lightweight input to the coboundary operator.
/// Built by the engine scheduler from the MemoryGraph before calling libsheaf.
/// This decouples libsheaf from the MemoryGraph implementation.
struct Neighborhood {
    /// Embedding of the newly ingested node.
    std::span<const float> new_node_mu;
    /// Text of the newly ingested node (for annotation).
    std::string new_node_text;

    /// Embeddings of neighbor nodes.
    std::vector<std::vector<float>> neighbor_mus;
    /// Texts of neighbor nodes (for annotation).
    std::vector<std::string> neighbor_texts;

    /// Edges from the new node to its neighbors.
    std::vector<Edge> edges;
};

} // namespace slm::sheaf
