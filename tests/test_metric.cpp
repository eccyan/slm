#include <gtest/gtest.h>
#include <metric/gaussian_node.hpp>
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
