#include <gtest/gtest.h>
#include <sheaf/neighborhood.hpp>
#include <vector>

using namespace slm::sheaf;

TEST(Edge, Construction) {
    std::vector<float> rel = {0.1f, 0.2f, 0.3f};
    Edge edge{42, EdgeType::Structural, rel};

    EXPECT_EQ(edge.neighbor_index, 42u);
    EXPECT_EQ(edge.type, EdgeType::Structural);
    EXPECT_EQ(edge.relation.size(), 3u);
    EXPECT_FLOAT_EQ(edge.relation[0], 0.1f);
}

TEST(Edge, KnnEdgeType) {
    Edge edge{10, EdgeType::KNN, {}};
    EXPECT_EQ(edge.type, EdgeType::KNN);
}

TEST(Neighborhood, Construction) {
    std::vector<float> new_mu = {1.0f, 2.0f};
    std::vector<float> neighbor_mu = {1.1f, 2.1f};
    std::vector<float> rel = {0.1f, 0.1f};

    Neighborhood hood;
    hood.new_node_mu = new_mu;
    hood.new_node_text = "new memory";
    hood.neighbor_mus.push_back(neighbor_mu);
    hood.neighbor_texts.push_back("old memory");
    hood.edges.push_back({0, EdgeType::Structural, rel});

    EXPECT_EQ(hood.new_node_mu.size(), 2u);
    EXPECT_EQ(hood.neighbor_mus.size(), 1u);
    EXPECT_EQ(hood.edges.size(), 1u);
    EXPECT_EQ(hood.new_node_text, "new memory");
    EXPECT_EQ(hood.neighbor_texts[0], "old memory");
}

TEST(Neighborhood, EmptyNeighborhood) {
    Neighborhood hood;
    hood.new_node_mu = std::vector<float>{1.0f};
    hood.new_node_text = "alone";
    EXPECT_TRUE(hood.edges.empty());
    EXPECT_TRUE(hood.neighbor_mus.empty());
}

#include <sheaf/coboundary.hpp>
#include <cmath>

// --- CoboundaryOperator tests ---

TEST(Coboundary, EmptyNeighborhoodZeroNorm) {
    Neighborhood hood;
    std::vector<float> mu = {1.0f, 0.0f};
    hood.new_node_mu = mu;
    hood.new_node_text = "test";

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);
    EXPECT_FLOAT_EQ(result.norm, 0.0f);
    EXPECT_TRUE(result.conflicting.empty());
}

TEST(Coboundary, ConsistentNodesZeroNorm) {
    std::vector<float> mu_new = {1.0f, 0.0f};
    std::vector<float> mu_neighbor = {0.0f, 0.0f};
    std::vector<float> relation = {1.0f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "new";
    hood.neighbor_mus.push_back(mu_neighbor);
    hood.neighbor_texts.push_back("old");
    hood.edges.push_back({0, EdgeType::Structural, relation});

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);
    EXPECT_NEAR(result.norm, 0.0f, 1e-6f);
    EXPECT_TRUE(result.conflicting.empty());
}

TEST(Coboundary, ContradictionDetected) {
    std::vector<float> mu_new = {1.0f, 0.0f, 0.0f};
    std::vector<float> mu_old = {0.0f, 1.0f, 0.0f};
    std::vector<float> zero_relation = {0.0f, 0.0f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "IP is 10.0.0.5";
    hood.neighbor_mus.push_back(std::vector<float>(mu_old.begin(), mu_old.end()));
    hood.neighbor_texts.push_back("IP is 10.0.0.1");
    hood.edges.push_back({0, EdgeType::Structural, zero_relation});

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);

    EXPECT_GT(result.norm, 0.5f);
    EXPECT_EQ(result.conflicting.size(), 1u);
    EXPECT_EQ(result.conflicting[0], 0u);
}

TEST(Coboundary, BelowThresholdNoConflict) {
    std::vector<float> mu_new = {1.0f, 0.0f};
    std::vector<float> mu_old = {0.9f, 0.0f};
    std::vector<float> relation = {0.1f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "new";
    hood.neighbor_mus.push_back(std::vector<float>(mu_old.begin(), mu_old.end()));
    hood.neighbor_texts.push_back("old");
    hood.edges.push_back({0, EdgeType::Structural, relation});

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);
    EXPECT_LT(result.norm, 0.5f);
    EXPECT_TRUE(result.conflicting.empty());
}

TEST(Coboundary, MultipleNeighborsPartialConflict) {
    std::vector<float> mu_new = {1.0f, 0.0f, 0.0f};

    std::vector<float> mu0 = {0.5f, 0.0f, 0.0f};
    std::vector<float> rel0 = {0.5f, 0.0f, 0.0f};

    std::vector<float> mu1 = {0.0f, 1.0f, 0.0f};
    std::vector<float> rel1 = {0.0f, 0.0f, 0.0f};

    std::vector<float> mu2 = {0.8f, 0.0f, 0.0f};
    std::vector<float> rel2 = {0.2f, 0.0f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "new";
    hood.neighbor_mus = {mu0, mu1, mu2};
    hood.neighbor_texts = {"old0", "old1", "old2"};
    hood.edges = {
        {0, EdgeType::Structural, rel0},
        {1, EdgeType::Structural, rel1},
        {2, EdgeType::KNN, rel2},
    };

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);
    EXPECT_GT(result.norm, 0.0f);

    bool found_1 = false;
    for (auto idx : result.conflicting) {
        if (idx == 1) found_1 = true;
        EXPECT_NE(idx, 0u) << "Neighbor 0 should not conflict";
        EXPECT_NE(idx, 2u) << "Neighbor 2 should not conflict";
    }
    EXPECT_TRUE(found_1) << "Neighbor 1 should be flagged as conflicting";
}

TEST(Coboundary, StructuralEdgesWeightedHigher) {
    std::vector<float> mu_new = {1.0f, 0.0f};

    std::vector<float> mu_s = {0.7f, 0.0f};
    std::vector<float> rel_s = {0.0f, 0.0f};

    std::vector<float> mu_k = {0.7f, 0.0f};
    std::vector<float> rel_k = {0.0f, 0.0f};

    Neighborhood hood_s;
    hood_s.new_node_mu = mu_new;
    hood_s.new_node_text = "new";
    hood_s.neighbor_mus = {mu_s};
    hood_s.neighbor_texts = {"old"};
    hood_s.edges = {{0, EdgeType::Structural, rel_s}};

    Neighborhood hood_k;
    hood_k.new_node_mu = mu_new;
    hood_k.new_node_text = "new";
    hood_k.neighbor_mus = {mu_k};
    hood_k.neighbor_texts = {"old"};
    hood_k.edges = {{0, EdgeType::KNN, rel_k}};

    CoboundaryOperator op;
    auto result_s = op.compute_local(hood_s, 10.0f);
    auto result_k = op.compute_local(hood_k, 10.0f);

    EXPECT_GT(result_s.norm, result_k.norm)
        << "Structural edges should have higher weight in the norm";
}

#include <sheaf/annotation.hpp>

// --- Annotation tests ---

TEST(Annotation, FormatBasic) {
    Annotation ann;
    ann.superseded_text = "proxy IP is 10.0.0.1";
    ann.superseding_text = "proxy IP is 10.0.0.5";
    ann.delta_norm = 0.73f;

    std::string result = format_annotation(ann);
    EXPECT_EQ(result,
        "<!-- cohomology: superseded \"proxy IP is 10.0.0.1\" "
        "by \"proxy IP is 10.0.0.5\", δ-norm=0.73 -->");
}

TEST(Annotation, FormatHighNorm) {
    Annotation ann;
    ann.superseded_text = "old";
    ann.superseding_text = "new";
    ann.delta_norm = 1.41421f;

    std::string result = format_annotation(ann);
    EXPECT_EQ(result,
        "<!-- cohomology: superseded \"old\" by \"new\", δ-norm=1.41 -->");
}

TEST(Annotation, FormatZeroNorm) {
    Annotation ann;
    ann.superseded_text = "a";
    ann.superseding_text = "b";
    ann.delta_norm = 0.0f;

    std::string result = format_annotation(ann);
    EXPECT_EQ(result,
        "<!-- cohomology: superseded \"a\" by \"b\", δ-norm=0.00 -->");
}

TEST(Annotation, FormatLongText) {
    Annotation ann;
    ann.superseded_text = "The deployment uses IP address 192.168.1.100 for the proxy server";
    ann.superseding_text = "The deployment uses IP address 10.0.0.5 for the proxy server";
    ann.delta_norm = 0.55f;

    std::string result = format_annotation(ann);
    EXPECT_NE(result.find("192.168.1.100"), std::string::npos);
    EXPECT_NE(result.find("10.0.0.5"), std::string::npos);
    EXPECT_EQ(result.substr(0, 5), "<!-- ");
    EXPECT_EQ(result.substr(result.size() - 4), " -->");
}

TEST(Annotation, FormatQuotesInText) {
    Annotation ann;
    ann.superseded_text = "said \"hello\"";
    ann.superseding_text = "said \"goodbye\"";
    ann.delta_norm = 0.5f;

    std::string result = format_annotation(ann);
    EXPECT_NE(result.find("said \\\"hello\\\""), std::string::npos);
    EXPECT_NE(result.find("said \\\"goodbye\\\""), std::string::npos);
}

// --- Integration: full contradiction detection flow ---

TEST(SheafIntegration, DetectAndAnnotateContradiction) {
    std::vector<float> mu_new = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> mu_old = {0.1f, 0.9f, 0.0f, 0.0f};
    std::vector<float> zero_rel = {0.0f, 0.0f, 0.0f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "proxy IP is 10.0.0.5";
    hood.neighbor_mus = {std::vector<float>(mu_old.begin(), mu_old.end())};
    hood.neighbor_texts = {"proxy IP is 10.0.0.1"};
    hood.edges = {{0, EdgeType::Structural, zero_rel}};

    // Step 1: Compute coboundary
    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);

    ASSERT_GT(result.norm, 0.5f) << "Should detect contradiction";
    ASSERT_EQ(result.conflicting.size(), 1u);
    EXPECT_EQ(result.conflicting[0], 0u);

    // Step 2: Generate annotation for each conflicting neighbor
    for (auto neighbor_idx : result.conflicting) {
        Annotation ann;
        ann.superseded_text = hood.neighbor_texts[neighbor_idx];
        ann.superseding_text = hood.new_node_text;
        ann.delta_norm = result.norm;

        std::string annotation = format_annotation(ann);

        EXPECT_NE(annotation.find("proxy IP is 10.0.0.1"), std::string::npos);
        EXPECT_NE(annotation.find("proxy IP is 10.0.0.5"), std::string::npos);
        EXPECT_NE(annotation.find("δ-norm="), std::string::npos);
        EXPECT_EQ(annotation.substr(0, 5), "<!-- ");
        EXPECT_EQ(annotation.substr(annotation.size() - 4), " -->");
    }
}

TEST(SheafIntegration, NoContradictionNoAnnotation) {
    std::vector<float> mu_new = {1.0f, 0.0f, 0.0f};
    std::vector<float> mu_old = {0.0f, 1.0f, 0.0f};
    std::vector<float> rel = {1.0f, -1.0f, 0.0f};

    Neighborhood hood;
    hood.new_node_mu = mu_new;
    hood.new_node_text = "new fact";
    hood.neighbor_mus = {std::vector<float>(mu_old.begin(), mu_old.end())};
    hood.neighbor_texts = {"old fact"};
    hood.edges = {{0, EdgeType::Structural, rel}};

    CoboundaryOperator op;
    auto result = op.compute_local(hood, 0.5f);

    EXPECT_NEAR(result.norm, 0.0f, 1e-5f);
    EXPECT_TRUE(result.conflicting.empty())
        << "Consistent nodes should produce no contradictions";
}
