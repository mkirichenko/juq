package dev.juq.inference;

/**
 * Pure Java math operations for BERT inference.
 * Port of c/src/simd_ops.c (without SIMD — JIT will auto-vectorize simple loops).
 */
public final class MathOps {

    private MathOps() {}

    /**
     * Layer normalization: output = gamma * (x - mean) / sqrt(var + eps) + beta.
     * Input is [seqLen * hidden] flat array, normalized per-position.
     */
    public static float[] layerNorm(float[] input, float[] gamma, float[] beta,
                                     int seqLen, int hidden, float eps) {
        float[] output = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            int offset = s * hidden;

            // Compute mean
            float mean = 0;
            for (int i = 0; i < hidden; i++) {
                mean += input[offset + i];
            }
            mean /= hidden;

            // Compute variance
            float var = 0;
            for (int i = 0; i < hidden; i++) {
                float diff = input[offset + i] - mean;
                var += diff * diff;
            }
            var /= hidden;

            // Normalize
            float invStd = (float) (1.0 / Math.sqrt(var + eps));
            for (int i = 0; i < hidden; i++) {
                output[offset + i] = (input[offset + i] - mean) * invStd * gamma[i] + beta[i];
            }
        }
        return output;
    }

    /**
     * GELU activation (tanh approximation), applied in-place.
     * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    public static void gelu(float[] v, int offset, int len) {
        final float SQRT_2_OVER_PI = 0.7978845608028654f;
        for (int i = offset; i < offset + len; i++) {
            float x = v[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
            v[i] = 0.5f * x * (1.0f + (float) Math.tanh(inner));
        }
    }

    /**
     * Softmax over a sub-range, in-place. Numerically stable (subtract max first).
     */
    public static void softmax(float[] v, int offset, int len) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = offset; i < offset + len; i++) {
            if (v[i] > max) max = v[i];
        }
        float sum = 0;
        for (int i = offset; i < offset + len; i++) {
            v[i] = (float) Math.exp(v[i] - max);
            sum += v[i];
        }
        float invSum = 1.0f / sum;
        for (int i = offset; i < offset + len; i++) {
            v[i] *= invSum;
        }
    }

    /**
     * Dot product of two float sub-arrays.
     */
    public static float dotProduct(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    /**
     * L2-normalize a vector in-place.
     */
    public static void l2Normalize(float[] v) {
        float norm = 0;
        for (float x : v) {
            norm += x * x;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            float inv = 1.0f / norm;
            for (int i = 0; i < v.length; i++) {
                v[i] *= inv;
            }
        }
    }

    /**
     * Mean pooling: average token embeddings weighted by attention mask.
     * hidden is [seqLen * hiddenSize] flat, returns [hiddenSize].
     */
    public static float[] meanPool(float[] hidden, int[] attentionMask,
                                    int seqLen, int hiddenSize) {
        float[] pooled = new float[hiddenSize];
        float count = 0;
        for (int s = 0; s < seqLen; s++) {
            if (attentionMask[s] != 0) {
                int offset = s * hiddenSize;
                for (int i = 0; i < hiddenSize; i++) {
                    pooled[i] += hidden[offset + i];
                }
                count++;
            }
        }
        if (count > 0) {
            float inv = 1.0f / count;
            for (int i = 0; i < hiddenSize; i++) {
                pooled[i] *= inv;
            }
        }
        return pooled;
    }

    /**
     * Symmetric per-tensor quantization: float[] → byte[] + scale.
     * scale = absMax / 127. Returns scale.
     */
    public static float symmetricQuantize(float[] input, int inOff, int len, byte[] output) {
        float absMax = 0;
        for (int i = 0; i < len; i++) {
            float abs = Math.abs(input[inOff + i]);
            if (abs > absMax) absMax = abs;
        }
        if (absMax == 0) {
            java.util.Arrays.fill(output, 0, len, (byte) 0);
            return 1.0f;
        }
        float scale = absMax / 127.0f;
        float invScale = 127.0f / absMax;
        for (int i = 0; i < len; i++) {
            int v = Math.round(input[inOff + i] * invScale);
            output[i] = (byte) Math.max(-127, Math.min(127, v));
        }
        return scale;
    }

    /**
     * Int8 dot product: sum of a[i] * b[i] with byte arithmetic, int accumulator.
     */
    public static int dotProductI8(byte[] a, int aOff, byte[] b, int bOff, int len) {
        int sum = 0;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    /**
     * Int8 matrix-vector multiply: out[i] = sum_j(mat[i*cols + j] * vec[j]).
     * mat is [rows x cols] in row-major, vec is [cols], out is [rows].
     */
    public static void matVecMulI8(byte[] mat, byte[] vec, int[] out, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            int sum = 0;
            int mOff = i * cols;
            for (int j = 0; j < cols; j++) {
                sum += mat[mOff + j] * vec[j];
            }
            out[i] = sum;
        }
    }
}
