#include <gtest/gtest.h>
#include <metric/gaussian_node.hpp>
#include <metric/simd_ops.hpp>
#include <vector>
#include <cmath>

using namespace slm::metric;

TEST(GaussianNode, SigmaRampConstants) {
    EXPECT_GT(SIGMA_MAX, SIGMA_MIN);
    EXPECT_GT(SIGMA_MIN, 0.0f);
    EXPECT_EQ(RAMP_STEPS, 10u);
}

TEST(GaussianNode, ComputeSigmaAtZeroAccess) {
    float sigma = compute_sigma_component(0);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MAX);
}

TEST(GaussianNode, ComputeSigmaAtMaxAccess) {
    float sigma = compute_sigma_component(10);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MIN);
}

TEST(GaussianNode, ComputeSigmaBeyondMax) {
    float sigma = compute_sigma_component(100);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MIN);
}

TEST(GaussianNode, ComputeSigmaMidpoint) {
    float sigma = compute_sigma_component(5);
    float expected = SIGMA_MAX * 0.5f + SIGMA_MIN * 0.5f;
    EXPECT_FLOAT_EQ(sigma, expected);
}

TEST(GaussianNode, FillSigmaVector) {
    std::vector<float> sigma(4);
    fill_sigma(sigma, 0);
    for (float s : sigma) {
        EXPECT_FLOAT_EQ(s, SIGMA_MAX);
    }

    fill_sigma(sigma, 10);
    for (float s : sigma) {
        EXPECT_FLOAT_EQ(s, SIGMA_MIN);
    }
}

TEST(GaussianNode, StructLayout) {
    std::vector<float> mu = {1.0f, 2.0f, 3.0f};
    std::vector<float> sigma = {0.5f, 0.5f, 0.5f};
    GaussianNode node{mu, sigma, 5};

    EXPECT_EQ(node.mu.size(), 3u);
    EXPECT_EQ(node.sigma.size(), 3u);
    EXPECT_EQ(node.access_count, 5u);
    EXPECT_FLOAT_EQ(node.mu[0], 1.0f);
    EXPECT_FLOAT_EQ(node.sigma[1], 0.5f);
}

// --- SIMD kernel tests ---

TEST(SimdOps, WeightedSqDiffIdentical) {
    std::vector<float> mu = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    float result = slm::metric::simd_weighted_sq_diff(
        mu.data(), mu.data(), sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(SimdOps, WeightedSqDiffSimple) {
    std::vector<float> mu_p = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> mu_q = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(SimdOps, WeightedSqDiffWithSigma) {
    std::vector<float> mu_p = {2.0f};
    std::vector<float> mu_q = {0.0f};
    std::vector<float> sigma_p = {2.0f};
    std::vector<float> sigma_q = {2.0f};
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma_p.data(), sigma_q.data(), 1
    );
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(SimdOps, WeightedSqDiffHighDim) {
    constexpr uint32_t dim = 384;
    std::vector<float> mu_p(dim, 1.0f);
    std::vector<float> mu_q(dim, 0.0f);
    std::vector<float> sigma(dim, 1.0f);
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma.data(), sigma.data(), dim
    );
    EXPECT_NEAR(result, 384.0f, 1e-3f);
}

TEST(SimdOps, VarianceDivergenceIdentical) {
    std::vector<float> sigma = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = slm::metric::simd_variance_divergence(
        sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(SimdOps, VarianceDivergenceSimple) {
    std::vector<float> sigma_p = {1.0f};
    std::vector<float> sigma_q = {std::exp(1.0f)};
    float result = slm::metric::simd_variance_divergence(
        sigma_p.data(), sigma_q.data(), 1
    );
    EXPECT_NEAR(result, 4.0f, 1e-5f);
}

TEST(SimdOps, VarianceDivergenceMultiDim) {
    std::vector<float> sigma_p = {1.0f, 1.0f};
    std::vector<float> sigma_q = {std::exp(1.0f), std::exp(1.0f)};
    float result = slm::metric::simd_variance_divergence(
        sigma_p.data(), sigma_q.data(), 2
    );
    EXPECT_NEAR(result, 8.0f, 1e-4f);
}

TEST(SimdOps, VarianceDivergenceHighDim) {
    constexpr uint32_t dim = 384;
    std::vector<float> sigma_p(dim, 1.0f);
    std::vector<float> sigma_q(dim, 2.0f);
    float result = slm::metric::simd_variance_divergence(
        sigma_p.data(), sigma_q.data(), dim
    );
    float expected = dim * (2.0f * std::log(2.0f)) * (2.0f * std::log(2.0f));
    EXPECT_NEAR(result, expected, 0.5f);
}

TEST(SimdOps, NonAlignedDimension) {
    std::vector<float> mu_p = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> mu_q = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma(7, 1.0f);
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma.data(), sigma.data(), 7
    );
    EXPECT_NEAR(result, 140.0f, 1e-3f);
}

#include <metric/fisher_rao.hpp>

// --- FisherRaoMetric distance tests ---

TEST(FisherRaoDistance, IdenticalNodesZeroDistance) {
    std::vector<float> mu = {1.0f, 2.0f, 3.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f};
    GaussianNode a{mu, sigma, 5};
    GaussianNode b{mu, sigma, 5};

    FisherRaoMetric metric;
    float d = metric.distance(a, b);
    EXPECT_FLOAT_EQ(d, 0.0f);
}

TEST(FisherRaoDistance, SymmetricDistance) {
    std::vector<float> mu_a = {1.0f, 0.0f, 0.0f};
    std::vector<float> mu_b = {0.0f, 1.0f, 0.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f};
    GaussianNode a{mu_a, sigma, 5};
    GaussianNode b{mu_b, sigma, 5};

    FisherRaoMetric metric;
    EXPECT_FLOAT_EQ(metric.distance(a, b), metric.distance(b, a));
}

TEST(FisherRaoDistance, KnownAnalyticalValue) {
    // mu_p = [1, 0], mu_q = [0, 0], sigma = SIGMA_MIN for both
    // Variance term = 0 (identical sigma)
    // weighted_diff = 1/(0.1*0.1) = 100
    // d = sqrt(100) = 10
    std::vector<float> mu_p = {1.0f, 0.0f};
    std::vector<float> mu_q = {0.0f, 0.0f};
    std::vector<float> sigma(2, SIGMA_MIN);
    GaussianNode p{mu_p, sigma, 10};
    GaussianNode q{mu_q, sigma, 10};

    FisherRaoMetric metric;
    float d = metric.distance(p, q);
    float expected_diff = 1.0f / (SIGMA_MIN * SIGMA_MIN);
    EXPECT_NEAR(d, std::sqrt(expected_diff), 1e-3f);
}

TEST(FisherRaoDistance, DifferentSigmaAddsVarianceTerm) {
    std::vector<float> mu = {0.0f};
    std::vector<float> sigma_p = {1.0f};
    std::vector<float> sigma_q = {2.0f};
    GaussianNode p{mu, sigma_p, 10};
    GaussianNode q{mu, sigma_q, 10};

    FisherRaoMetric metric;
    float d = metric.distance(p, q);
    float expected = std::sqrt((2.0f * std::log(2.0f)) * (2.0f * std::log(2.0f)));
    EXPECT_NEAR(d, expected, 1e-3f);
}

TEST(FisherRaoDistance, TriangleInequality) {
    std::vector<float> mu_a = {0.0f, 0.0f};
    std::vector<float> mu_b = {1.0f, 0.0f};
    std::vector<float> mu_c = {1.0f, 1.0f};
    std::vector<float> sigma(2, 1.0f);
    GaussianNode a{mu_a, sigma, 5};
    GaussianNode b{mu_b, sigma, 5};
    GaussianNode c{mu_c, sigma, 5};

    FisherRaoMetric metric;
    float d_ab = metric.distance(a, b);
    float d_bc = metric.distance(b, c);
    float d_ac = metric.distance(a, c);
    EXPECT_LE(d_ac, d_ab + d_bc + 1e-5f);
}

TEST(FisherRaoDistance, HighDim384) {
    constexpr uint32_t dim = 384;
    std::vector<float> mu_p(dim, 0.0f);
    std::vector<float> mu_q(dim, 0.0f);
    mu_q[0] = 1.0f;
    std::vector<float> sigma(dim, 1.0f);
    GaussianNode p{mu_p, sigma, 10};
    GaussianNode q{mu_q, sigma, 10};

    FisherRaoMetric metric;
    float d = metric.distance(p, q);
    EXPECT_GT(d, 0.0f);
    EXPECT_TRUE(std::isfinite(d));
}

// --- FisherRaoMetric top_k tests ---

TEST(FisherRaoTopK, ReturnsCorrectCount) {
    std::vector<float> mu_q = {1.0f, 0.0f};
    std::vector<float> sigma(2, 1.0f);
    GaussianNode query{mu_q, sigma, 5};

    std::vector<float> mu0 = {0.0f, 0.0f};
    std::vector<float> mu1 = {0.5f, 0.0f};
    std::vector<float> mu2 = {2.0f, 0.0f};

    std::vector<GaussianNode> candidates = {
        {mu0, sigma, 5},
        {mu1, sigma, 5},
        {mu2, sigma, 5},
    };

    FisherRaoMetric metric;
    auto result = metric.top_k(query, candidates, 2);
    EXPECT_EQ(result.size(), 2u);
}

TEST(FisherRaoTopK, RankedByAscendingDistance) {
    std::vector<float> mu_q = {1.0f, 0.0f};
    std::vector<float> sigma(2, 1.0f);
    GaussianNode query{mu_q, sigma, 5};

    std::vector<float> mu0 = {0.0f, 0.0f};
    std::vector<float> mu1 = {0.9f, 0.0f};
    std::vector<float> mu2 = {2.0f, 0.0f};

    std::vector<GaussianNode> candidates = {
        {mu0, sigma, 5},
        {mu1, sigma, 5},
        {mu2, sigma, 5},
    };

    FisherRaoMetric metric;
    auto result = metric.top_k(query, candidates, 3);
    ASSERT_EQ(result.size(), 3u);

    EXPECT_EQ(result[0], 1u);

    float prev_d = 0.0f;
    for (auto idx : result) {
        float d = metric.distance(query, candidates[idx]);
        EXPECT_GE(d + 1e-5f, prev_d);
        prev_d = d;
    }
}

TEST(FisherRaoTopK, KLargerThanCandidates) {
    std::vector<float> mu_q = {0.0f};
    std::vector<float> sigma = {1.0f};
    GaussianNode query{mu_q, sigma, 5};

    std::vector<float> mu0 = {1.0f};
    std::vector<GaussianNode> candidates = {{mu0, sigma, 5}};

    FisherRaoMetric metric;
    auto result = metric.top_k(query, candidates, 10);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0u);
}

TEST(FisherRaoTopK, EmptyCandidates) {
    std::vector<float> mu_q = {0.0f};
    std::vector<float> sigma = {1.0f};
    GaussianNode query{mu_q, sigma, 5};

    std::span<const GaussianNode> empty;

    FisherRaoMetric metric;
    auto result = metric.top_k(query, empty, 5);
    EXPECT_TRUE(result.empty());
}

TEST(FisherRaoTopK, KZero) {
    std::vector<float> mu_q = {0.0f};
    std::vector<float> sigma = {1.0f};
    GaussianNode query{mu_q, sigma, 5};

    std::vector<float> mu0 = {1.0f};
    std::vector<GaussianNode> candidates = {{mu0, sigma, 5}};

    FisherRaoMetric metric;
    auto result = metric.top_k(query, candidates, 0);
    EXPECT_TRUE(result.empty());
}

// --- Integration: graduated sigma ramp ---

TEST(FisherRaoIntegration, SigmaRampChangesDistance) {
    constexpr uint32_t dim = 8;
    std::vector<float> mu_p(dim, 0.0f);
    std::vector<float> mu_q(dim, 0.0f);
    mu_q[0] = 0.5f;

    FisherRaoMetric metric;

    float prev_distance = 0.0f;
    for (uint32_t access = 0; access <= 10; access += 2) {
        std::vector<float> sigma(dim);
        fill_sigma(sigma, access);

        GaussianNode p{mu_p, sigma, access};
        GaussianNode q{mu_q, sigma, access};
        float d = metric.distance(p, q);

        if (access > 0) {
            EXPECT_GT(d, prev_distance)
                << "Distance should increase with access_count. "
                << "access=" << access << " d=" << d << " prev=" << prev_distance;
        }
        prev_distance = d;
    }

    std::vector<float> sigma_max(dim, SIGMA_MAX);
    GaussianNode p0{mu_p, sigma_max, 0};
    GaussianNode q0{mu_q, sigma_max, 0};
    float d_at_0 = metric.distance(p0, q0);

    std::vector<float> sigma_min(dim, SIGMA_MIN);
    GaussianNode p10{mu_p, sigma_min, 10};
    GaussianNode q10{mu_q, sigma_min, 10};
    float d_at_10 = metric.distance(p10, q10);

    EXPECT_GT(d_at_10 / d_at_0, 10.0f)
        << "Full geodesic should be >10x larger than cosine-like regime";
}

TEST(FisherRaoIntegration, TopKWithMixedAccessCounts) {
    constexpr uint32_t dim = 4;
    std::vector<float> mu_q = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma_q(dim, SIGMA_MIN);
    GaussianNode query{mu_q, sigma_q, 10};

    std::vector<float> mu0 = {0.9f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma0(dim, SIGMA_MIN);

    std::vector<float> mu1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> sigma1(dim, SIGMA_MAX);

    std::vector<float> mu2 = {0.5f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma2(dim);
    fill_sigma(sigma2, 5);

    std::vector<GaussianNode> candidates = {
        {mu0, sigma0, 10},
        {mu1, sigma1, 0},
        {mu2, sigma2, 5},
    };

    FisherRaoMetric metric;
    auto result = metric.top_k(query, candidates, 3);
    ASSERT_EQ(result.size(), 3u);

    EXPECT_EQ(result[0], 0u);
}
