# libmetric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Fisher-Rao retrieval metric library — the mathematical engine that scores memory similarity using geodesic distance on diagonal Gaussian distributions, with SIMD-optimized inner loops.

**Architecture:** Three focused files: `GaussianNode` struct + sigma ramp logic, SIMD kernels for the two inner-loop operations (weighted squared diff and variance divergence), and `FisherRaoMetric` class composing them into `distance()` and `top_k()`. The library is header + two `.cpp` files, compiled as a CMake static library (`libmetric`) that depends on nothing except the C++ standard library. SIMD dispatch uses compile-time `#ifdef` with a scalar fallback.

**Tech Stack:** C++23, NEON intrinsics (Apple Silicon primary target, with scalar fallback), Google Test

---

## File Map

| File | Responsibility |
|---|---|
| `src/metric/CMakeLists.txt` | libmetric static library target |
| `src/metric/include/metric/gaussian_node.hpp` | `GaussianNode` struct, sigma ramp constants, `compute_sigma()` helper |
| `src/metric/include/metric/simd_ops.hpp` | SIMD kernel declarations |
| `src/metric/src/simd_ops.cpp` | SIMD kernel implementations (NEON/AVX2/SSE4.2/scalar) |
| `src/metric/include/metric/fisher_rao.hpp` | `FisherRaoMetric` class declaration |
| `src/metric/src/fisher_rao.cpp` | `distance()` and `top_k()` implementations |
| `tests/test_metric.cpp` | All libmetric tests |
| `CMakeLists.txt` | Root CMake (add `add_subdirectory(src/metric)`) |
| `tests/CMakeLists.txt` | Add metric_tests executable |

---

### Task 1: CMake Wiring & GaussianNode

**Files:**
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/metric)`)
- Create: `src/metric/CMakeLists.txt`
- Create: `src/metric/include/metric/gaussian_node.hpp`
- Create: `tests/test_metric.cpp`
- Modify: `tests/CMakeLists.txt` (add metric_tests executable)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metric.cpp`:

```cpp
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
    // access_count = 0 → alpha = 0 → sigma = SIGMA_MAX
    float sigma = compute_sigma_component(0);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MAX);
}

TEST(GaussianNode, ComputeSigmaAtMaxAccess) {
    // access_count = 10 → alpha = 1.0 → sigma = SIGMA_MIN
    float sigma = compute_sigma_component(10);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MIN);
}

TEST(GaussianNode, ComputeSigmaBeyondMax) {
    // access_count > 10 → clamped to alpha = 1.0
    float sigma = compute_sigma_component(100);
    EXPECT_FLOAT_EQ(sigma, SIGMA_MIN);
}

TEST(GaussianNode, ComputeSigmaMidpoint) {
    // access_count = 5 → alpha = 0.5 → sigma = midpoint
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
```

- [ ] **Step 2: Create src/metric/CMakeLists.txt**

```cmake
add_library(metric STATIC
    src/simd_ops.cpp
    src/fisher_rao.cpp
)

target_include_directories(metric PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(metric PUBLIC cxx_std_23)
```

- [ ] **Step 3: Add metric to root CMakeLists.txt**

Add this line after `add_subdirectory(src/slab)`:

```cmake
add_subdirectory(src/metric)
```

- [ ] **Step 4: Add metric_tests to tests/CMakeLists.txt**

Append to `tests/CMakeLists.txt`:

```cmake

add_executable(metric_tests
    test_metric.cpp
)

target_link_libraries(metric_tests PRIVATE
    metric
    GTest::gtest_main
)

gtest_discover_tests(metric_tests)
```

- [ ] **Step 5: Create placeholder sources so the library compiles**

Create `src/metric/src/simd_ops.cpp`:

```cpp
#include <metric/simd_ops.hpp>
// Implemented in Task 2
```

Create `src/metric/src/fisher_rao.cpp`:

```cpp
#include <metric/fisher_rao.hpp>
// Implemented in Task 3
```

Create `src/metric/include/metric/simd_ops.hpp`:

```cpp
#pragma once
// SIMD kernels — implemented in Task 2
```

Create `src/metric/include/metric/fisher_rao.hpp`:

```cpp
#pragma once
// FisherRaoMetric — implemented in Task 3
```

- [ ] **Step 6: Implement GaussianNode**

Create `src/metric/include/metric/gaussian_node.hpp`:

```cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace slm::metric {

/// Maximum sigma (high uncertainty, fresh node with 0 accesses).
/// At this value, Fisher-Rao degenerates toward cosine similarity.
inline constexpr float SIGMA_MAX = 10.0f;

/// Minimum sigma (low uncertainty, well-accessed node with 10+ accesses).
/// At this value, full geodesic distance is active.
inline constexpr float SIGMA_MIN = 0.1f;

/// Number of accesses to complete the sigma ramp from SIGMA_MAX to SIGMA_MIN.
inline constexpr uint32_t RAMP_STEPS = 10;

/// Compute the per-component sigma value for a given access count.
/// Linearly interpolates from SIGMA_MAX (access=0) to SIGMA_MIN (access>=10).
inline float compute_sigma_component(uint32_t access_count) {
    float alpha = std::min(static_cast<float>(access_count) / RAMP_STEPS, 1.0f);
    return SIGMA_MAX * (1.0f - alpha) + SIGMA_MIN * alpha;
}

/// Fill a sigma vector with uniform sigma values for a given access count.
inline void fill_sigma(std::span<float> sigma, uint32_t access_count) {
    float s = compute_sigma_component(access_count);
    for (auto& v : sigma) {
        v = s;
    }
}

/// A memory node modeled as a diagonal Gaussian distribution.
/// mu: embedding vector (mean), sigma: per-dimension std dev.
struct GaussianNode {
    std::span<const float> mu;
    std::span<const float> sigma;
    uint32_t access_count;
};

} // namespace slm::metric
```

- [ ] **Step 7: Build and run tests**

Run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build && cd build && ctest --output-on-failure -R metric
```

Expected: All 7 GaussianNode tests PASS.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt src/metric/ tests/test_metric.cpp tests/CMakeLists.txt
git commit -m "feat(metric): add libmetric scaffolding with GaussianNode and sigma ramp"
```

---

### Task 2: SIMD Kernels

**Files:**
- Replace: `src/metric/include/metric/simd_ops.hpp`
- Replace: `src/metric/src/simd_ops.cpp`
- Modify: `tests/test_metric.cpp` (append SIMD tests)

The two core inner-loop operations, each with NEON/scalar implementations.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metric.cpp`:

```cpp
#include <metric/simd_ops.hpp>

// --- SIMD kernel tests ---

TEST(SimdOps, WeightedSqDiffIdentical) {
    // Identical vectors → distance = 0
    std::vector<float> mu = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    float result = slm::metric::simd_weighted_sq_diff(
        mu.data(), mu.data(), sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(SimdOps, WeightedSqDiffSimple) {
    // mu_p = [1, 0, 0, 0], mu_q = [0, 0, 0, 0], sigma = [1, 1, 1, 1]
    // Result = (1-0)^2 / (1*1) = 1.0
    std::vector<float> mu_p = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> mu_q = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma = {1.0f, 1.0f, 1.0f, 1.0f};
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(SimdOps, WeightedSqDiffWithSigma) {
    // mu_p = [2], mu_q = [0], sigma_p = [2], sigma_q = [2]
    // Result = (2-0)^2 / (2*2) = 4/4 = 1.0
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
    // 384 dimensions (MiniLM size), all 1.0 vs all 0.0, sigma = 1.0
    // Result = 384 * (1-0)^2 / (1*1) = 384.0
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
    // Identical sigma → divergence = 0
    std::vector<float> sigma = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = slm::metric::simd_variance_divergence(
        sigma.data(), sigma.data(), 4
    );
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(SimdOps, VarianceDivergenceSimple) {
    // sigma_p = [1], sigma_q = [e] → (2 * ln(e/1))^2 = (2*1)^2 = 4.0
    std::vector<float> sigma_p = {1.0f};
    std::vector<float> sigma_q = {std::exp(1.0f)};
    float result = slm::metric::simd_variance_divergence(
        sigma_p.data(), sigma_q.data(), 1
    );
    EXPECT_NEAR(result, 4.0f, 1e-5f);
}

TEST(SimdOps, VarianceDivergenceMultiDim) {
    // sigma_p = [1, 1], sigma_q = [e, e] → 2 * 4.0 = 8.0
    std::vector<float> sigma_p = {1.0f, 1.0f};
    std::vector<float> sigma_q = {std::exp(1.0f), std::exp(1.0f)};
    float result = slm::metric::simd_variance_divergence(
        sigma_p.data(), sigma_q.data(), 2
    );
    EXPECT_NEAR(result, 8.0f, 1e-4f);
}

TEST(SimdOps, VarianceDivergenceHighDim) {
    // 384 dimensions with sigma_p=1, sigma_q=2
    // Per dim: (2*ln(2/1))^2 = (2*0.6931)^2 = 1.9218^2 ≈ 1.9218
    // Correction: (2*ln2)^2 = (1.3863)^2 = 1.9218
    // Total = 384 * 1.9218 ≈ 737.97
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
    // 7 dimensions — not a multiple of 4 or 8, tests scalar tail handling
    std::vector<float> mu_p = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> mu_q = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma(7, 1.0f);
    float result = slm::metric::simd_weighted_sq_diff(
        mu_p.data(), mu_q.data(), sigma.data(), sigma.data(), 7
    );
    // 1+4+9+16+25+36+49 = 140
    EXPECT_NEAR(result, 140.0f, 1e-3f);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: FAIL — simd_ops.hpp has no declarations.

- [ ] **Step 3: Implement SIMD kernel header**

Replace `src/metric/include/metric/simd_ops.hpp`:

```cpp
#pragma once

#include <cstdint>

namespace slm::metric {

/// Compute sum of (mu_p[i] - mu_q[i])^2 / (sigma_p[i] * sigma_q[i])
/// for i in [0, dim). SIMD-accelerated with scalar tail handling.
float simd_weighted_sq_diff(
    const float* mu_p, const float* mu_q,
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);

/// Compute sum of (2 * ln(sigma_q[i] / sigma_p[i]))^2
/// for i in [0, dim). SIMD-accelerated with scalar tail handling.
float simd_variance_divergence(
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);

} // namespace slm::metric
```

- [ ] **Step 4: Implement SIMD kernels**

Replace `src/metric/src/simd_ops.cpp`:

```cpp
#include <metric/simd_ops.hpp>
#include <cmath>

#if defined(SLM_HAS_NEON)
#include <arm_neon.h>
#elif defined(SLM_HAS_AVX2)
#include <immintrin.h>
#endif

namespace slm::metric {

float simd_weighted_sq_diff(
    const float* mu_p, const float* mu_q,
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
) {
    float sum = 0.0f;
    uint32_t i = 0;

#if defined(SLM_HAS_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        float32x4_t mp = vld1q_f32(mu_p + i);
        float32x4_t mq = vld1q_f32(mu_q + i);
        float32x4_t sp = vld1q_f32(sigma_p + i);
        float32x4_t sq = vld1q_f32(sigma_q + i);

        float32x4_t diff = vsubq_f32(mp, mq);
        float32x4_t diff_sq = vmulq_f32(diff, diff);
        float32x4_t sigma_prod = vmulq_f32(sp, sq);
        // Use approximate reciprocal + Newton-Raphson for division
        float32x4_t inv = vrecpeq_f32(sigma_prod);
        inv = vmulq_f32(vrecpsq_f32(sigma_prod, inv), inv);
        acc = vmlaq_f32(acc, diff_sq, inv);
    }
    // Horizontal sum
    sum = vaddvq_f32(acc);

#elif defined(SLM_HAS_AVX2)
    // AVX2 path — 8 floats at a time
    __m256 acc8 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 mp = _mm256_loadu_ps(mu_p + i);
        __m256 mq = _mm256_loadu_ps(mu_q + i);
        __m256 sp = _mm256_loadu_ps(sigma_p + i);
        __m256 sq = _mm256_loadu_ps(sigma_q + i);

        __m256 diff = _mm256_sub_ps(mp, mq);
        __m256 diff_sq = _mm256_mul_ps(diff, diff);
        __m256 sigma_prod = _mm256_mul_ps(sp, sq);
        __m256 ratio = _mm256_div_ps(diff_sq, sigma_prod);
        acc8 = _mm256_add_ps(acc8, ratio);
    }
    // Horizontal sum of 8 floats
    __m128 lo = _mm256_castps256_ps128(acc8);
    __m128 hi = _mm256_extractf128_ps(acc8, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum = _mm_cvtss_f32(sum4);
#endif

    // Scalar tail (or full scalar fallback)
    for (; i < dim; ++i) {
        float diff = mu_p[i] - mu_q[i];
        sum += (diff * diff) / (sigma_p[i] * sigma_q[i]);
    }

    return sum;
}

float simd_variance_divergence(
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
) {
    float sum = 0.0f;
    uint32_t i = 0;

    // Variance divergence requires log, which has no direct SIMD instruction
    // on NEON. Use scalar loop — this term is typically much cheaper than
    // the weighted_sq_diff (only computed when sigma_p != sigma_q, which
    // is rare in practice since we use uniform sigma per node).
    //
    // For AVX2, _mm256_log_ps is available via SVML but not portable.
    // Keep scalar for correctness and portability.

    for (; i < dim; ++i) {
        float ratio = sigma_q[i] / sigma_p[i];
        float log_ratio = std::log(ratio);
        float term = 2.0f * log_ratio;
        sum += term * term;
    }

    return sum;
}

} // namespace slm::metric
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: All 16 tests PASS (7 GaussianNode + 9 SIMD).

- [ ] **Step 6: Commit**

```bash
git add src/metric/include/metric/simd_ops.hpp src/metric/src/simd_ops.cpp tests/test_metric.cpp
git commit -m "feat(metric): add SIMD-accelerated weighted_sq_diff and variance_divergence kernels"
```

---

### Task 3: FisherRaoMetric — distance()

**Files:**
- Replace: `src/metric/include/metric/fisher_rao.hpp`
- Replace: `src/metric/src/fisher_rao.cpp`
- Modify: `tests/test_metric.cpp` (append distance tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metric.cpp`:

```cpp
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
    // mu_p = [1, 0], mu_q = [0, 0], sigma = [1, 1] (same for both, access=10)
    // Variance term = 0 (identical sigma)
    // Weighted diff = (1-0)^2/(1*1) + (0-0)^2/(1*1) = 1.0
    // d_FR = sqrt(1.0) = 1.0
    std::vector<float> mu_p = {1.0f, 0.0f};
    std::vector<float> mu_q = {0.0f, 0.0f};
    std::vector<float> sigma(2, SIGMA_MIN);  // access_count=10 → SIGMA_MIN
    GaussianNode p{mu_p, sigma, 10};
    GaussianNode q{mu_q, sigma, 10};

    FisherRaoMetric metric;
    float d = metric.distance(p, q);
    // d = sqrt(weighted_diff + variance_div)
    // weighted_diff = 1/(0.1*0.1) = 100
    // d = sqrt(100) = 10
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
    // Weighted diff = 0 (same mu)
    // Variance div = (2*ln(2/1))^2 = (2*0.6931)^2 = 1.9218
    // d = sqrt(1.9218) ≈ 1.3863
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
    // 384-dim vectors (MiniLM size)
    constexpr uint32_t dim = 384;
    std::vector<float> mu_p(dim, 0.0f);
    std::vector<float> mu_q(dim, 0.0f);
    mu_q[0] = 1.0f;  // differ in one dimension only
    std::vector<float> sigma(dim, 1.0f);
    GaussianNode p{mu_p, sigma, 10};
    GaussianNode q{mu_q, sigma, 10};

    FisherRaoMetric metric;
    float d = metric.distance(p, q);
    EXPECT_GT(d, 0.0f);
    EXPECT_TRUE(std::isfinite(d));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: FAIL — `FisherRaoMetric` not defined.

- [ ] **Step 3: Implement FisherRaoMetric header**

Replace `src/metric/include/metric/fisher_rao.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <metric/gaussian_node.hpp>

namespace slm::metric {

/// Fisher-Rao geodesic distance metric on diagonal Gaussian distributions.
///
/// d_FR^2 = sum (2*ln(sigma_q/sigma_p))^2 + sum (mu_p - mu_q)^2 / (sigma_p * sigma_q)
///
/// Uses SIMD-accelerated inner loops for the two summation terms.
class FisherRaoMetric {
public:
    /// Compute the Fisher-Rao geodesic distance between two Gaussian nodes.
    /// Returns sqrt(variance_divergence + weighted_sq_diff).
    float distance(const GaussianNode& p, const GaussianNode& q) const;

    /// Find the top-k nearest candidates to a query, ranked by ascending distance.
    /// Returns a vector of indices into the `candidates` span.
    std::vector<uint32_t> top_k(
        const GaussianNode& query,
        std::span<const GaussianNode> candidates,
        uint32_t k
    ) const;
};

} // namespace slm::metric
```

- [ ] **Step 4: Implement distance()**

Replace `src/metric/src/fisher_rao.cpp`:

```cpp
#include <metric/fisher_rao.hpp>
#include <metric/simd_ops.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace slm::metric {

float FisherRaoMetric::distance(const GaussianNode& p, const GaussianNode& q) const {
    assert(p.mu.size() == q.mu.size());
    assert(p.sigma.size() == q.sigma.size());
    assert(p.mu.size() == p.sigma.size());

    uint32_t dim = static_cast<uint32_t>(p.mu.size());

    float var_div = simd_variance_divergence(
        p.sigma.data(), q.sigma.data(), dim
    );

    float weighted_diff = simd_weighted_sq_diff(
        p.mu.data(), q.mu.data(),
        p.sigma.data(), q.sigma.data(), dim
    );

    return std::sqrt(var_div + weighted_diff);
}

std::vector<uint32_t> FisherRaoMetric::top_k(
    const GaussianNode& query,
    std::span<const GaussianNode> candidates,
    uint32_t k
) const {
    // Placeholder — implemented in Task 4
    return {};
}

} // namespace slm::metric
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: All 22 tests PASS (7 GaussianNode + 9 SIMD + 6 distance).

- [ ] **Step 6: Commit**

```bash
git add src/metric/include/metric/fisher_rao.hpp src/metric/src/fisher_rao.cpp tests/test_metric.cpp
git commit -m "feat(metric): add FisherRaoMetric::distance() with geodesic computation"
```

---

### Task 4: FisherRaoMetric — top_k()

**Files:**
- Modify: `src/metric/src/fisher_rao.cpp` (replace top_k placeholder)
- Modify: `tests/test_metric.cpp` (append top_k tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metric.cpp`:

```cpp
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

    // Distances to query [1,0]:
    // mu0=[0,0] → d=1.0, mu1=[0.9,0] → d=0.1, mu2=[2,0] → d=1.0
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

    // Closest should be index 1 (mu1 is nearest to query)
    EXPECT_EQ(result[0], 1u);

    // Verify distances are non-decreasing
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
    // Should return all available candidates (1)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: FAIL — `top_k` returns empty (placeholder).

- [ ] **Step 3: Implement top_k()**

In `src/metric/src/fisher_rao.cpp`, replace the `top_k` method body:

```cpp
std::vector<uint32_t> FisherRaoMetric::top_k(
    const GaussianNode& query,
    std::span<const GaussianNode> candidates,
    uint32_t k
) const {
    if (k == 0 || candidates.empty()) {
        return {};
    }

    // Compute distances for all candidates
    struct IndexDist {
        uint32_t index;
        float distance;
    };

    std::vector<IndexDist> scored;
    scored.reserve(candidates.size());
    for (uint32_t i = 0; i < candidates.size(); ++i) {
        scored.push_back({i, distance(query, candidates[i])});
    }

    // Partial sort to get top-k
    uint32_t actual_k = std::min(k, static_cast<uint32_t>(scored.size()));
    std::partial_sort(
        scored.begin(),
        scored.begin() + actual_k,
        scored.end(),
        [](const IndexDist& a, const IndexDist& b) {
            return a.distance < b.distance;
        }
    );

    std::vector<uint32_t> result;
    result.reserve(actual_k);
    for (uint32_t i = 0; i < actual_k; ++i) {
        result.push_back(scored[i].index);
    }
    return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: All 27 tests PASS (7 GaussianNode + 9 SIMD + 6 distance + 5 top_k).

- [ ] **Step 5: Commit**

```bash
git add src/metric/src/fisher_rao.cpp tests/test_metric.cpp
git commit -m "feat(metric): add FisherRaoMetric::top_k() with partial sort ranking"
```

---

### Task 5: Integration Test — Graduated Sigma Ramp End-to-End

**Files:**
- Modify: `tests/test_metric.cpp` (append integration test)

This test verifies the complete design intent: as access_count increases from 0 to 10, the distance metric transitions from cosine-like behavior (large sigma → difference matters less) to full geodesic (small sigma → fine-grained discrimination).

- [ ] **Step 1: Write the integration test**

Append to `tests/test_metric.cpp`:

```cpp
// --- Integration: graduated sigma ramp ---

TEST(FisherRaoIntegration, SigmaRampChangesDistance) {
    // Two nodes with the same mu offset but different access counts.
    // As access_count increases (sigma decreases), the distance should INCREASE
    // because lower sigma means higher precision = differences are amplified.
    constexpr uint32_t dim = 8;
    std::vector<float> mu_p(dim, 0.0f);
    std::vector<float> mu_q(dim, 0.0f);
    mu_q[0] = 0.5f;  // small difference in one dimension

    FisherRaoMetric metric;

    float prev_distance = 0.0f;
    for (uint32_t access = 0; access <= 10; access += 2) {
        std::vector<float> sigma(dim);
        fill_sigma(sigma, access);

        GaussianNode p{mu_p, sigma, access};
        GaussianNode q{mu_q, sigma, access};
        float d = metric.distance(p, q);

        if (access > 0) {
            // Distance should increase as sigma shrinks
            EXPECT_GT(d, prev_distance)
                << "Distance should increase with access_count. "
                << "access=" << access << " d=" << d << " prev=" << prev_distance;
        }
        prev_distance = d;
    }

    // At access=0 (SIGMA_MAX=10), distance should be small
    std::vector<float> sigma_max(dim, SIGMA_MAX);
    GaussianNode p0{mu_p, sigma_max, 0};
    GaussianNode q0{mu_q, sigma_max, 0};
    float d_at_0 = metric.distance(p0, q0);

    // At access=10 (SIGMA_MIN=0.1), distance should be much larger
    std::vector<float> sigma_min(dim, SIGMA_MIN);
    GaussianNode p10{mu_p, sigma_min, 10};
    GaussianNode q10{mu_q, sigma_min, 10};
    float d_at_10 = metric.distance(p10, q10);

    // The ratio should be dramatic (SIGMA_MAX/SIGMA_MIN = 100)
    EXPECT_GT(d_at_10 / d_at_0, 10.0f)
        << "Full geodesic should be >10x larger than cosine-like regime";
}

TEST(FisherRaoIntegration, TopKWithMixedAccessCounts) {
    // Query node with high access count (precise)
    constexpr uint32_t dim = 4;
    std::vector<float> mu_q = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma_q(dim, SIGMA_MIN);
    GaussianNode query{mu_q, sigma_q, 10};

    // Candidate 0: close mu, high access → very close (precise match)
    std::vector<float> mu0 = {0.9f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sigma0(dim, SIGMA_MIN);

    // Candidate 1: far mu, low access → moderate (imprecise, far)
    std::vector<float> mu1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> sigma1(dim, SIGMA_MAX);

    // Candidate 2: medium mu, medium access
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

    // Candidate 0 should be closest (small mu diff, same precise sigma)
    EXPECT_EQ(result[0], 0u);
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build && cd build && ctest --output-on-failure -R metric`

Expected: All 29 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_metric.cpp
git commit -m "test(metric): add sigma ramp integration test verifying graduated distance behavior"
```

---

## Summary

After completing all 5 tasks, libmetric provides:

| Component | What it does |
|---|---|
| `GaussianNode` | Struct representing a memory as a diagonal Gaussian; sigma ramp from cosine to geodesic |
| `simd_weighted_sq_diff` | SIMD-accelerated mean-difference term of Fisher-Rao distance |
| `simd_variance_divergence` | Variance-divergence term (scalar, log-based) |
| `FisherRaoMetric::distance()` | Full Fisher-Rao geodesic distance between two Gaussian nodes |
| `FisherRaoMetric::top_k()` | Partial-sort ranking of candidates by ascending distance |

The library has no dependencies on libslab or any other project library — it's pure math. libsheaf will depend on it for k-NN edge discovery, and the engine scheduler will use it for Tier 1 READ processing.
