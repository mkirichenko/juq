package dev.juq.inference;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Pure Java math operations for BERT inference.
 * Port of c/src/simd_ops.c using JDK Incubator Vector API (Java 25).
 *
 * Hot loops (dotProduct, l2Normalize, meanPool, quantize, int8 matmul) are vectorized
 * using {@link FloatVector}, {@link IntVector} and {@link ByteVector}. The Vector API
 * lowers these to platform SIMD (AVX2/AVX-512 on x86, NEON on ARM) at JIT time.
 */
public final class MathOps {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * Byte species whose lane count matches {@link #I_SPECIES}, so that a byte vector
     * can be widened to a single int vector of the same length via convertShape(B2I).
     */
    private static final VectorSpecies<Byte> B_SPECIES =
            VectorSpecies.of(byte.class, VectorShape.forBitSize(I_SPECIES.length() * Byte.SIZE));

    private MathOps() {}

    /**
     * Layer normalization: output = gamma * (x - mean) / sqrt(var + eps) + beta.
     * Input is [seqLen * hidden] flat array, normalized per-position.
     */
    public static float[] layerNorm(float[] input, float[] gamma, float[] beta,
                                     int seqLen, int hidden, float eps) {
        float[] output = new float[seqLen * hidden];
        int upper = F_SPECIES.loopBound(hidden);

        for (int s = 0; s < seqLen; s++) {
            int offset = s * hidden;

            // Compute mean
            FloatVector sumV = FloatVector.zero(F_SPECIES);
            int i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                sumV = sumV.add(FloatVector.fromArray(F_SPECIES, input, offset + i));
            }
            float mean = sumV.reduceLanes(VectorOperators.ADD);
            for (; i < hidden; i++) mean += input[offset + i];
            mean /= hidden;

            // Compute variance
            FloatVector meanV = FloatVector.broadcast(F_SPECIES, mean);
            FloatVector varV = FloatVector.zero(F_SPECIES);
            i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                FloatVector diff = FloatVector.fromArray(F_SPECIES, input, offset + i).sub(meanV);
                varV = diff.fma(diff, varV);
            }
            float var = varV.reduceLanes(VectorOperators.ADD);
            for (; i < hidden; i++) {
                float diff = input[offset + i] - mean;
                var += diff * diff;
            }
            var /= hidden;

            // Normalize: gamma * (x - mean) * invStd + beta
            float invStd = (float) (1.0 / Math.sqrt(var + eps));
            FloatVector invStdV = FloatVector.broadcast(F_SPECIES, invStd);
            i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                FloatVector x = FloatVector.fromArray(F_SPECIES, input, offset + i);
                FloatVector g = FloatVector.fromArray(F_SPECIES, gamma, i);
                FloatVector b = FloatVector.fromArray(F_SPECIES, beta, i);
                // (x - mean) * invStd * gamma + beta
                FloatVector normed = x.sub(meanV).mul(invStdV);
                normed.fma(g, b).intoArray(output, offset + i);
            }
            for (; i < hidden; i++) {
                output[offset + i] = (input[offset + i] - mean) * invStd * gamma[i] + beta[i];
            }
        }
        return output;
    }

    /**
     * GELU activation (tanh approximation), applied in-place.
     * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     *
     * Note: no vectorized tanh in JDK Vector API yet, so this stays scalar — JIT can
     * still partially auto-vectorize the polynomial portion.
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
     * Max-reduce and sum-reduce are vectorized; exp() stays scalar (no vectorized exp).
     */
    public static void softmax(float[] v, int offset, int len) {
        int upper = F_SPECIES.loopBound(len);

        // Max reduce
        float max = Float.NEGATIVE_INFINITY;
        if (upper > 0) {
            FloatVector maxV = FloatVector.fromArray(F_SPECIES, v, offset);
            int i = F_SPECIES.length();
            for (; i < upper; i += F_SPECIES.length()) {
                maxV = maxV.max(FloatVector.fromArray(F_SPECIES, v, offset + i));
            }
            max = maxV.reduceLanes(VectorOperators.MAX);
        }
        for (int i = Math.max(upper, 0); i < len; i++) {
            if (v[offset + i] > max) max = v[offset + i];
        }

        // exp(x - max) — must stay scalar
        float sum = 0;
        for (int i = 0; i < len; i++) {
            float e = (float) Math.exp(v[offset + i] - max);
            v[offset + i] = e;
            sum += e;
        }

        // Normalize by 1/sum — vectorized
        float invSum = 1.0f / sum;
        FloatVector invSumV = FloatVector.broadcast(F_SPECIES, invSum);
        int i = 0;
        for (; i < upper; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, v, offset + i).mul(invSumV).intoArray(v, offset + i);
        }
        for (; i < len; i++) {
            v[offset + i] *= invSum;
        }
    }

    /**
     * Dot product of two float sub-arrays. Vectorized with FMA.
     */
    public static float dotProduct(float[] a, int aOff, float[] b, int bOff, int len) {
        int upper = F_SPECIES.loopBound(len);
        FloatVector accV = FloatVector.zero(F_SPECIES);
        int i = 0;
        for (; i < upper; i += F_SPECIES.length()) {
            FloatVector av = FloatVector.fromArray(F_SPECIES, a, aOff + i);
            FloatVector bv = FloatVector.fromArray(F_SPECIES, b, bOff + i);
            accV = av.fma(bv, accV);
        }
        float sum = accV.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    /**
     * L2-normalize a vector in-place. Vectorized squared-sum reduction.
     */
    public static void l2Normalize(float[] v) {
        int upper = F_SPECIES.loopBound(v.length);
        FloatVector accV = FloatVector.zero(F_SPECIES);
        int i = 0;
        for (; i < upper; i += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, v, i);
            accV = x.fma(x, accV);
        }
        float norm = accV.reduceLanes(VectorOperators.ADD);
        for (; i < v.length; i++) {
            norm += v[i] * v[i];
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            float inv = 1.0f / norm;
            FloatVector invV = FloatVector.broadcast(F_SPECIES, inv);
            i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                FloatVector.fromArray(F_SPECIES, v, i).mul(invV).intoArray(v, i);
            }
            for (; i < v.length; i++) {
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
        int upper = F_SPECIES.loopBound(hiddenSize);
        float count = 0;

        for (int s = 0; s < seqLen; s++) {
            if (attentionMask[s] != 0) {
                int offset = s * hiddenSize;
                int i = 0;
                for (; i < upper; i += F_SPECIES.length()) {
                    FloatVector p = FloatVector.fromArray(F_SPECIES, pooled, i);
                    FloatVector h = FloatVector.fromArray(F_SPECIES, hidden, offset + i);
                    p.add(h).intoArray(pooled, i);
                }
                for (; i < hiddenSize; i++) {
                    pooled[i] += hidden[offset + i];
                }
                count++;
            }
        }
        if (count > 0) {
            float inv = 1.0f / count;
            FloatVector invV = FloatVector.broadcast(F_SPECIES, inv);
            int i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                FloatVector.fromArray(F_SPECIES, pooled, i).mul(invV).intoArray(pooled, i);
            }
            for (; i < hiddenSize; i++) {
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
        int upper = F_SPECIES.loopBound(len);

        // Find absMax — vectorized
        float absMax = 0;
        if (upper > 0) {
            FloatVector maxV = FloatVector.zero(F_SPECIES);
            int i = 0;
            for (; i < upper; i += F_SPECIES.length()) {
                FloatVector x = FloatVector.fromArray(F_SPECIES, input, inOff + i).abs();
                maxV = maxV.max(x);
            }
            absMax = maxV.reduceLanes(VectorOperators.MAX);
        }
        for (int i = Math.max(upper, 0); i < len; i++) {
            float abs = Math.abs(input[inOff + i]);
            if (abs > absMax) absMax = abs;
        }

        if (absMax == 0) {
            java.util.Arrays.fill(output, 0, len, (byte) 0);
            return 1.0f;
        }
        float scale = absMax / 127.0f;
        float invScale = 127.0f / absMax;

        // Quantize — scalar (byte store lane-by-lane path is not faster than JIT loop)
        for (int i = 0; i < len; i++) {
            int v = Math.round(input[inOff + i] * invScale);
            output[i] = (byte) Math.max(-127, Math.min(127, v));
        }
        return scale;
    }

    /**
     * Int8 dot product: sum of a[i] * b[i] with byte arithmetic, int accumulator.
     * Vectorized via byte→int widening.
     */
    public static int dotProductI8(byte[] a, int aOff, byte[] b, int bOff, int len) {
        int laneCount = B_SPECIES.length();
        int upper = len - (len % laneCount);
        IntVector accV = IntVector.zero(I_SPECIES);
        int i = 0;
        for (; i < upper; i += laneCount) {
            IntVector av = (IntVector) ByteVector.fromArray(B_SPECIES, a, aOff + i)
                    .convertShape(VectorOperators.B2I, I_SPECIES, 0);
            IntVector bv = (IntVector) ByteVector.fromArray(B_SPECIES, b, bOff + i)
                    .convertShape(VectorOperators.B2I, I_SPECIES, 0);
            accV = av.mul(bv).add(accV);
        }
        int sum = accV.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    /**
     * Int8 matrix-vector multiply: out[i] = sum_j(mat[i*cols + j] * vec[j]).
     * mat is [rows x cols] in row-major, vec is [cols], out is [rows].
     */
    public static void matVecMulI8(byte[] mat, byte[] vec, int[] out, int rows, int cols) {
        int laneCount = B_SPECIES.length();
        int upper = cols - (cols % laneCount);

        for (int i = 0; i < rows; i++) {
            int mOff = i * cols;
            IntVector accV = IntVector.zero(I_SPECIES);
            int j = 0;
            for (; j < upper; j += laneCount) {
                IntVector mv = (IntVector) ByteVector.fromArray(B_SPECIES, mat, mOff + j)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                IntVector vv = (IntVector) ByteVector.fromArray(B_SPECIES, vec, j)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                accV = mv.mul(vv).add(accV);
            }
            int sum = accV.reduceLanes(VectorOperators.ADD);
            for (; j < cols; j++) {
                sum += mat[mOff + j] * vec[j];
            }
            out[i] = sum;
        }
    }
}
