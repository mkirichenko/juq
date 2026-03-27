#include "simd_ops.h"
#include <math.h>
#include <string.h>
#include <float.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define USE_SSE2 1
#endif

/* ---- Dot product ---- */

float vec_dot(const float *a, const float *b, size_t n) {
#if USE_AVX2
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result;
    _mm_store_ss(&result, s);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#elif USE_SSE2
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

/* ---- Matrix-vector multiply ---- */

void mat_vec_mul(const float *mat, const float *vec, float *out,
                 int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = vec_dot(mat + (size_t)i * cols, vec, cols);
    }
}

/* ---- Matrix multiply (A * B^T) ---- */

void mat_mul(const float *A, const float *B_T, float *C,
             int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = vec_dot(A + (size_t)i * k, B_T + (size_t)j * k, k);
        }
    }
}

/* ---- Vector add ---- */

void vec_add(const float *a, const float *b, float *out, size_t n) {
#if USE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

/* ---- Broadcast add ---- */

void broadcast_add(float *out, const float *bias, int seq_len, int hidden) {
    for (int s = 0; s < seq_len; s++) {
        vec_add(out + s * hidden, bias, out + s * hidden, hidden);
    }
}

/* ---- L2 normalize ---- */

void vec_l2_normalize(float *v, size_t n) {
    float norm_sq = vec_dot(v, v, n);
    if (norm_sq <= 0.0f) return;
    float inv_norm = 1.0f / sqrtf(norm_sq);
#if USE_AVX2
    __m256 vn = _mm256_set1_ps(inv_norm);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(v + i, _mm256_mul_ps(va, vn));
    }
    for (; i < n; i++) v[i] *= inv_norm;
#else
    for (size_t i = 0; i < n; i++) v[i] *= inv_norm;
#endif
}

/* ---- Softmax ---- */

void vec_softmax(float *v, size_t n) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (v[i] > max_val) max_val = v[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) v[i] *= inv_sum;
}

/* ---- GELU ---- */

void vec_gelu(float *v, size_t n) {
    /* GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) */
    static const float SQRT_2_INV = 0.7071067811865475f;
    for (size_t i = 0; i < n; i++) {
        v[i] = v[i] * 0.5f * (1.0f + erff(v[i] * SQRT_2_INV));
    }
}

/* ---- Layer normalization ---- */

void layer_norm(const float *input, const float *gamma, const float *beta,
                float *output, int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *row = input + s * hidden;
        float *out_row = output + s * hidden;

        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += row[i];
        mean /= hidden;

        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = row[i] - mean;
            var += diff * diff;
        }
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* Normalize */
#if USE_AVX2
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vistd = _mm256_set1_ps(inv_std);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 vx = _mm256_loadu_ps(row + i);
            __m256 vg = _mm256_loadu_ps(gamma + i);
            __m256 vb = _mm256_loadu_ps(beta + i);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), vistd);
            _mm256_storeu_ps(out_row + i, _mm256_fmadd_ps(norm, vg, vb));
        }
        for (; i < hidden; i++) {
            out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
#endif
    }
}
