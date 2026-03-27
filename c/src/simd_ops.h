#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <stddef.h>

/*
 * SIMD-optimized operations for quantized model inference.
 * Uses SSE2/AVX2 when available, falls back to scalar.
 */

/* Vector dot product: sum(a[i] * b[i]) */
float vec_dot(const float *a, const float *b, size_t n);

/* Matrix-vector multiply: out[i] = sum_j(mat[i*cols + j] * vec[j])
 * mat is row-major [rows x cols], vec is [cols], out is [rows] */
void mat_vec_mul(const float *mat, const float *vec, float *out,
                 int rows, int cols);

/* Full matrix multiply: C[m x n] = A[m x k] * B[k x n] (row-major)
 * B_T is B transposed [n x k] for cache efficiency */
void mat_mul(const float *A, const float *B_T, float *C,
             int m, int k, int n);

/* Vector add: out[i] = a[i] + b[i] */
void vec_add(const float *a, const float *b, float *out, size_t n);

/* Broadcast add: out[seq][i] = A[seq][hidden] + bias[hidden] */
void broadcast_add(float *out, const float *bias, int seq_len, int hidden);

/* L2 normalize in-place */
void vec_l2_normalize(float *v, size_t n);

/* Softmax in-place over contiguous array of size n */
void vec_softmax(float *v, size_t n);

/* GELU activation in-place */
void vec_gelu(float *v, size_t n);

/* Layer normalization:
 * For each row of [seq_len x hidden]:
 *   out = gamma * (x - mean) / sqrt(var + eps) + beta */
void layer_norm(const float *input, const float *gamma, const float *beta,
                float *output, int seq_len, int hidden, float eps);

#endif
