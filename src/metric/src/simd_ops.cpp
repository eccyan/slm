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
        float32x4_t inv = vrecpeq_f32(sigma_prod);
        inv = vmulq_f32(vrecpsq_f32(sigma_prod, inv), inv);
        inv = vmulq_f32(vrecpsq_f32(sigma_prod, inv), inv);
        acc = vmlaq_f32(acc, diff_sq, inv);
    }
    sum = vaddvq_f32(acc);

#elif defined(SLM_HAS_AVX2)
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

    // Variance divergence requires log — keep scalar for portability
    for (uint32_t i = 0; i < dim; ++i) {
        float ratio = sigma_q[i] / sigma_p[i];
        float log_ratio = std::log(ratio);
        float term = 2.0f * log_ratio;
        sum += term * term;
    }

    return sum;
}

} // namespace slm::metric
